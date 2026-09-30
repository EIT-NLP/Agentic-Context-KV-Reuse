"""Paper-based AgentKVShift residual correction; no official code was available.

Scores concatenate KV heads per token. Keys must already share global RoPE
coordinates. K and V probe sets remain separate even when execution uses their union.
"""
import math
from dataclasses import dataclass
import torch


@dataclass
class Probes:
    start: int
    end: int
    keys: torch.Tensor
    values: torch.Tensor
    key_weights: torch.Tensor
    value_weights: torch.Tensor


def select(plan, fresh_k, fresh_v, cached_k, cached_v, ratio):
    if not 0<ratio<=1: raise ValueError('Probe ratio must be in (0,1]')
    result=[]
    for s in plan.segments:
        count=math.ceil(ratio*s.token_count)
        dk=(fresh_k[...,s.start:s.end,:].float()-cached_k[...,s.start:s.end,:].float()).square().sum((0,1,3)).sqrt()
        dv=(fresh_v[...,s.start:s.end,:].float()-cached_v[...,s.start:s.end,:].float()).square().sum((0,1,3)).sqrt()
        k=torch.argsort(dk,descending=True,stable=True)[:count].sort().values+s.start
        v=torch.argsort(dv,descending=True,stable=True)[:count].sort().values+s.start
        result.append(Probes(s.start,s.end,k,v,dk.clamp(max=1),dv.clamp(max=1)))
    return result


def selected_positions(plan, probes, device):
    online=torch.ones(len(plan.token_ids),dtype=torch.bool,device=device)
    for p in probes:
        online[p.start:p.end]=False
        online[p.keys]=True;online[p.values]=True
    return torch.where(online)[0]


def correct(old, fresh, positions, probes, kind):
    output=old.clone()
    local=torch.full((old.shape[-2],),-1,dtype=torch.long,device=old.device)
    local[positions]=torch.arange(len(positions),device=old.device)
    # Refresh online tokens; artifact rows below replace this provisional write.
    output[...,positions,:]=fresh
    for p in probes:
        idx=p.keys if kind=='key' else p.values
        weights=p.key_weights if kind=='key' else p.value_weights
        fresh_probes=fresh[...,local[idx],:]
        residual=(fresh_probes.float()-old[...,idx,:].float()).mean(-2,keepdim=True)
        corrected=old[...,p.start:p.end,:].float()+weights[None,None,:,None]*residual
        output[...,p.start:p.end,:]=corrected.to(old.dtype)
        output[...,idx,:]=fresh_probes
    return output
