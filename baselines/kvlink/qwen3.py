"""Qwen3 KVLink architecture port: document isolation and trainable link tokens.

The mask follows upstream titan_preprocessor.make_segment_mask. Token IDs are
new Qwen tokenizer entries rather than the upstream Llama reserved-token IDs.
All base weights remain trainable, as in KVLink; this is not query LoRA.
"""
from dataclasses import dataclass
import torch
from attuner.plan import PromptPlan, Segment


@dataclass
class LinkedInput:
    plan: PromptPlan
    segment_ids: tuple[int, ...]


def add_links(plan, link_ids):
    tokens, segments, ids = [], [], []
    cursor = 0
    if len(plan.segments) > len(link_ids):
        raise ValueError('More artifacts than reserved link-token groups')
    for index, s in enumerate(plan.segments):
        gap = plan.token_ids[cursor:s.start]
        tokens.extend(gap); ids.extend([0]*len(gap))
        start=len(tokens)
        tokens.extend(s.token_ids); ids.extend([index+1]*len(s.token_ids))
        segments.append(Segment(start,len(tokens),s.token_ids,s.document_id))
        tokens.extend(link_ids[index]); ids.extend([0]*len(link_ids[index]))
        cursor=s.end
    tokens.extend(plan.token_ids[cursor:]); ids.extend([0]*(len(plan.token_ids)-cursor))
    return LinkedInput(PromptPlan(tuple(tokens),tuple(segments),tuple(x==0 for x in ids)),tuple(ids))


def segment_mask(segment_ids, dtype):
    s=segment_ids
    length=s.shape[-1]
    causal=torch.arange(length,device=s.device)[None,:] <= torch.arange(length,device=s.device)[:,None]
    valid=(s[:,:,None]>=0)&(s[:,None,:]>=0)
    allowed=causal[None]&valid&((s[:,:,None]==s[:,None,:])|(s[:,:,None]==0))
    return torch.zeros(allowed.shape,device=s.device,dtype=dtype).masked_fill(~allowed,-torch.inf)[:,None]


class Qwen3KVLink(torch.nn.Module):
    def __init__(self, base, tokenizer, max_artifacts, links_per_artifact=5):
        super().__init__()
        self.base=base
        self.links_per_artifact=links_per_artifact
        self.max_artifacts=max_artifacts
        names=[[f'<|kvlink_{i}_{j}|>' for j in range(links_per_artifact)] for i in range(max_artifacts)]
        tokenizer.add_special_tokens({'additional_special_tokens':[n for row in names for n in row]})
        base.resize_token_embeddings(len(tokenizer),mean_resizing=False)
        self.link_ids=[tokenizer.convert_tokens_to_ids(row) for row in names]
        self.tokenizer=tokenizer
        for p in self.base.parameters(): p.requires_grad_(True)

    def forward(self, plan, response):
        linked=add_links(plan,self.link_ids)
        device=self.base.device
        ids=torch.tensor([list(linked.plan.token_ids)+list(response)],device=device)
        segments=torch.tensor([list(linked.segment_ids)+[0]*len(response)],device=device)
        labels=ids.clone(); labels[:,:len(linked.plan.token_ids)]=-100
        return self.base(input_ids=ids,attention_mask=segment_mask(segments,self.base.dtype),
                         position_ids=torch.arange(ids.shape[1],device=device)[None],
                         labels=labels,use_cache=False).loss

    def save(self, directory):
        from pathlib import Path
        import json
        path=Path(directory); path.mkdir(parents=True,exist_ok=True)
        self.base.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        (path/'baseline_config.json').write_text(json.dumps(dict(method='kvlink',
            links_per_artifact=self.links_per_artifact,max_artifacts=self.max_artifacts,
            link_ids=self.link_ids,training_objective='response_cross_entropy'),indent=2)+'\n')
