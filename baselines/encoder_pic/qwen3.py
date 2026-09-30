"""Qwen3 adaptation of COMB's encoder + gated cross-attention architecture.

One bidirectional encoder layer and cross block per four original decoder layers.
The original decoder/embedding is frozen. New layers are initialized from scratch,
with zero cross-attention/MLP gates, following upstream CombLlama.
"""
import copy
import json
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3MLP, Qwen3RMSNorm
from attuner.attention_ops import repeat_kv


class CrossBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads=config.num_attention_heads
        self.kv_heads=config.num_key_value_heads
        self.dim=config.head_dim
        self.q_proj=nn.Linear(config.hidden_size,self.heads*self.dim,bias=False)
        self.o_proj=nn.Linear(self.heads*self.dim,config.hidden_size,bias=False)
        self.q_norm=Qwen3RMSNorm(self.dim,eps=config.rms_norm_eps)
        self.k_norm=Qwen3RMSNorm(self.dim,eps=config.rms_norm_eps)
        self.input_norm=Qwen3RMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        self.post_norm=Qwen3RMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        self.mlp=Qwen3MLP(config)
        self.attention_gate=nn.Parameter(torch.zeros(1))
        self.mlp_gate=nn.Parameter(torch.zeros(1))

    def forward(self, hidden, cache):
        b,t,_=hidden.shape
        q=self.q_norm(self.q_proj(self.input_norm(hidden)).view(b,t,self.heads,self.dim)).transpose(1,2)
        k,v=cache
        out=F.scaled_dot_product_attention(q,repeat_kv(k,self.heads),repeat_kv(v,self.heads),
                                          is_causal=False,dropout_p=0.)
        out=self.o_proj(out.transpose(1,2).reshape(b,t,-1))
        hidden=hidden+self.attention_gate.tanh()*out
        return hidden+self.mlp_gate.tanh()*self.mlp(self.post_norm(hidden))


class Qwen3EncoderPIC(nn.Module):
    def __init__(self, base, insertion_layers=None):
        super().__init__()
        self.base=base
        cfg=copy.deepcopy(base.config)
        self.insertion_layers=list(insertion_layers if insertion_layers is not None else range(3,cfg.num_hidden_layers,4))
        if not self.insertion_layers or any(i<0 or i>=cfg.num_hidden_layers for i in self.insertion_layers):
            raise ValueError('Cross-attention positions must refer to original decoder layers')
        self._states=None
        self.checkpoint_encoder=True
        for p in base.parameters(): p.requires_grad_(False)
        self.encoder=nn.ModuleList([Qwen3DecoderLayer(cfg,i) for i in range(len(self.insertion_layers))])
        self.k_proj=nn.ModuleList([nn.Linear(cfg.hidden_size,cfg.num_key_value_heads*cfg.head_dim,bias=False)
                                  for _ in self.insertion_layers])
        self.v_proj=nn.ModuleList([nn.Linear(cfg.hidden_size,cfg.num_key_value_heads*cfg.head_dim,bias=False)
                                  for _ in self.insertion_layers])
        self.cross=nn.ModuleList([CrossBlock(cfg) for _ in self.insertion_layers])
        for modules in (self.encoder,self.k_proj,self.v_proj,self.cross):
            modules.apply(base._init_weights)
            modules.to(device=base.device,dtype=base.dtype)
        for layer in self.encoder: layer.self_attn.is_causal=False
        self._handles=[]
        for index,position in enumerate(self.insertion_layers):
            def before(module,args,kwargs,index=index):
                if self._states is None: return args,kwargs
                hidden=args[0] if args else kwargs['hidden_states']
                hidden=self.cross[index](hidden,self._states[index])
                if args: return (hidden,*args[1:]),kwargs
                return args,{**kwargs,'hidden_states':hidden}
            self._handles.append(base.model.layers[position].register_forward_pre_hook(before,with_kwargs=True))

    def encode(self, artifacts):
        cfg=self.base.config
        all_states=[]
        for tokens in artifacts:
            ids=torch.tensor([tokens],device=self.base.device)
            hidden=self.base.model.embed_tokens(ids)
            positions=torch.arange(ids.shape[-1],device=ids.device)[None]
            rope=self.base.model.rotary_emb(hidden,positions)
            states=[]
            for index,layer in enumerate(self.encoder):
                def forward(h,layer=layer,rope=rope):
                    result=layer(h,attention_mask=None,position_embeddings=rope,use_cache=False)
                    return result[0] if isinstance(result,tuple) else result
                if self.training and self.checkpoint_encoder:
                    hidden=checkpoint(forward,hidden,use_reentrant=False)
                else: hidden=forward(hidden)
                shape=(1,len(tokens),cfg.num_key_value_heads,cfg.head_dim)
                k=self.cross[index].k_norm(self.k_proj[index](hidden).view(shape)).transpose(1,2)
                v=self.v_proj[index](hidden).view(shape).transpose(1,2)
                states.append((k,v))
            all_states.append(states)
        return tuple(tuple(torch.cat([state[i][j] for state in all_states],-2) for j in range(2))
                     for i in range(len(self.insertion_layers)))

    def clear_context(self): self._states=None

    def forward(self, plan, response):
        artifact_positions={i for s in plan.segments for i in range(s.start,s.end)}
        online=[token for i,token in enumerate(plan.token_ids) if i not in artifact_positions]
        self._states=self.encode([s.token_ids for s in plan.segments])
        ids=torch.tensor([online+list(response)],device=self.base.device)
        labels=ids.clone(); labels[:,:len(online)]=-100
        # Context remains available until backward: checkpoint recomputation calls hooks again.
        return self.base(input_ids=ids,labels=labels,use_cache=False).loss

    def save(self,directory):
        path=Path(directory); path.mkdir(parents=True,exist_ok=True)
        weights={name:value.detach().cpu() for name,value in self.state_dict().items() if not name.startswith('base.')}
        torch.save(weights,path/'encoder_pic.pt')
        (path/'baseline_config.json').write_text(json.dumps(dict(method='encoder_pic',
            insertion_layers=self.insertion_layers,training_objective='response_cross_entropy',
            base_model=self.base.config._name_or_path),indent=2)+'\n')
