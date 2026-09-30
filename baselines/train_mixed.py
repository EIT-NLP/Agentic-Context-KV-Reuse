"""Train Qwen3 KVLink/Encoder-PIC on the agreed mixed12k teacher responses.

Response cross-entropy, not Attuner's KL. Separate process/output from the paper
trainer. Each GPU handles one sample per forward; accumulation is configurable.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import random
import sys
import time
ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT),str(ROOT/'src')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method',choices=['kvlink','comb','encoder_pic'])
    parser.add_argument('--config',type=Path,default=ROOT/'configs/qwen3_4b.yaml')
    parser.add_argument('--data-root',type=Path,default=ROOT)
    parser.add_argument('--data',type=Path,help='Optional prepared training JSONL subset for a small end-to-end test')
    parser.add_argument('--model',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-updates',type=int,help='Optional short validation run; does not change the data selection')
    parser.add_argument('--global-batch-size',type=int,default=16)
    parser.add_argument('--accumulation',type=int,default=4)
    parser.add_argument('--save-every',type=int,default=50,help='Rolling resume checkpoint interval in optimizer updates')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.method == "comb": args.method = "encoder_pic"
    from baselines.progress import Progress
    with Progress(method=args.method, rank=int(os.environ.get('RANK',0)), model=args.model) as progress:
        run(args, parser, progress)


def run(args, parser, progress):
    progress.phase('read_config', path=str(args.config))
    import yaml
    from attuner.config import load
    cfg=load(args.config)
    rank=int(os.environ.get('RANK',0)); world=int(os.environ.get('WORLD_SIZE',1))
    local_rank=int(os.environ.get('LOCAL_RANK',0))
    if args.global_batch_size!=world*args.accumulation:
        parser.error('micro=1 requires global batch = world size * accumulation')
    if args.max_updates is not None and args.max_updates<1: parser.error('--max-updates must be positive')
    if min(args.accumulation,args.global_batch_size,args.save_every)<1: parser.error('Batch and save intervals must be positive')
    lr=5e-6 if args.method=='kvlink' else 5e-5
    protocol=dict(method=args.method,model=args.model,objective='teacher_response_cross_entropy',
        learning_rate=lr,weight_decay=.01,micro_batch_size=1,grad_accumulation=args.accumulation,
        global_batch_size=args.global_batch_size,epochs=1,world_size=world,max_updates=args.max_updates,
        training_data=str(args.data.resolve()) if args.data else 'mixed12k',
        formal_run=args.max_updates is None and args.data is None)
    if rank==0: print(json.dumps(protocol,indent=2),flush=True)
    if args.dry_run: return
    from baselines.run_state import bind_run, final_model_exists, restore_training, save_training, trim_training_log
    from attuner.io import write_json
    progress.phase('import_training_dependencies')
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from attuner.io import read_jsonl
    from attuner.plan import build_plan
    from transformers import AutoModelForCausalLM,AutoTokenizer
    progress.phase('initialize_distributed', rank=rank, world=world)
    torch.cuda.set_device(local_rank)
    if world>1: dist.init_process_group('nccl')
    seed=cfg.get('seed',20260918)
    random.seed(seed)
    torch.manual_seed(seed)
    progress.phase('read_mixed12k')
    data_path=args.data if args.data else args.data_root/cfg['mixed_training']['output_dir']/'train.jsonl'
    examples=read_jsonl(data_path)
    if not examples: raise ValueError('No training examples found')
    if args.data is None and len(examples)!=12000:
        raise ValueError('Expected the agreed 12000 mixed training examples')
    trajectories={}
    for source in ['musique','bigcodebench','personamem_v2']:
        path=args.data_root/cfg['teacher_reuse']['directory']/Path(args.model).name/f'{source}.jsonl'
        progress.phase('read_teacher_responses', dataset=source, path=str(path))
        for row in read_jsonl(path):
            if row['instance_id'] in trajectories: raise ValueError('Duplicate teacher response ID')
            trajectories[row['instance_id']]=row['response_token_ids']
    if any(not trajectories.get(row['instance_id']) for row in examples):
        raise ValueError('Each mixed training example requires a nonempty teacher response')
    if len(examples)%args.global_batch_size:
        raise ValueError('Global batch must divide the training sample count')
    identity={k:v for k,v in protocol.items() if k not in {'max_updates','formal_run'}}
    identity.update(seed=seed,data_root=str(args.data_root.resolve()),
                    mixed_data=cfg['mixed_training']['output_dir'],teacher_directory=cfg['teacher_reuse']['directory'],
                    instance_ids=[row['instance_id'] for row in examples])
    if rank==0: bind_run(args.output,identity,'training_config.json')
    if world>1: dist.barrier()
    if (args.output/'COMPLETE').exists() and final_model_exists(args.output,args.method):
        if rank==0: print('Training already complete: '+str(args.output/'model'),flush=True)
        if world>1: dist.destroy_process_group()
        return
    if (args.output/'COMPLETE').exists() and not (args.output/'resume.pt').exists():
        raise RuntimeError('Final model is missing and no resume.pt is available')
    if rank==0: (args.output/'COMPLETE').unlink(missing_ok=True)
    if world>1: dist.barrier()
    progress.phase('load_tokenizer', source=args.model)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    progress.phase('load_model_weights', source=args.model)
    base=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True)
    progress.phase('move_model_to_device', device=f'cuda:{local_rank}')
    base=base.to(f'cuda:{local_rank}')
    progress.phase('initialize_trainable_modules')
    if args.method=='kvlink':
        from baselines.kvlink.qwen3 import Qwen3KVLink
        count=max(len(row.get('artifacts',row.get('skills',[]))) for row in examples)
        learner=Qwen3KVLink(base,tokenizer,max_artifacts=count)
    else:
        from baselines.encoder_pic.qwen3 import Qwen3EncoderPIC
        learner=Qwen3EncoderPIC(base)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    learner.train()
    progress.phase('initialize_optimizer_and_ddp')
    # Keep gradients in DDP reduction buckets instead of retaining a second
    # model-sized allocation. Preserve these views across accumulated updates.
    model=DDP(learner,device_ids=[local_rank],broadcast_buffers=False,
              gradient_as_bucket_view=True) if world>1 else learner
    params=[p for p in learner.parameters() if p.requires_grad]
    # The foreach implementation may allocate model-sized optimizer temporaries.
    optimizer=torch.optim.AdamW(params,lr=lr,weight_decay=.01,foreach=False)
    order=list(range(len(examples)));random.Random(seed).shuffle(order)
    progress.phase('restore_checkpoint', path=str(args.output/'resume.pt'))
    updates=restore_training(args.output,learner,optimizer,rank,world)
    if rank==0:
        trim_training_log(args.output,updates)
        print(f'Resuming at update {updates}',flush=True)
    total_updates=len(order)//args.global_batch_size
    target=min(total_updates,args.max_updates) if args.max_updates else total_updates
    initial_updates=updates
    training_started=time.monotonic()
    for start in range(updates*args.global_batch_size,target*args.global_batch_size,args.global_batch_size):
        progress.phase('train_update', update=updates+1, total_updates=total_updates, consumed_samples=start)
        batch=order[start:start+args.global_batch_size]
        if len(batch)!=args.global_batch_size:
            raise ValueError('Choose a batch size dividing mixed12k; do not silently drop samples')
        local=batch[rank::world]
        optimizer.zero_grad(set_to_none=False)
        total=0.
        for i,index in enumerate(local):
            row=examples[index]
            progress.update(instance_id=row['instance_id'], micro_sample=i+1, micro_samples=len(local))
            plan=build_plan(tokenizer,row,enable_thinking=False)
            context=model.no_sync() if world>1 and i<len(local)-1 else contextlib.nullcontext()
            with context:
                loss=model(plan,trajectories[row['instance_id']])
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite response CE')
                (loss/len(local)).backward()
            total+=float(loss.detach())/len(local)
            if hasattr(learner,'clear_context'): learner.clear_context()
        norm=torch.nn.utils.clip_grad_norm_(params,1.)
        optimizer.step();updates+=1
        if rank==0:
            args.output.mkdir(parents=True,exist_ok=True)
            with (args.output/'train.jsonl').open('a') as f:
                f.write(json.dumps(dict(update=updates,rank0_mean_loss=total,gradient_norm=float(norm)))+'\n')
            elapsed=time.monotonic()-training_started
            progress.phase('update_done', update=updates, total_updates=total_updates,
                           consumed_samples=updates*args.global_batch_size, rank0_mean_loss=total,
                           eta_s=round(elapsed/(updates-initial_updates)*(target-updates)))
        if updates%args.save_every==0 or updates==target:
            progress.phase('save_checkpoint', update=updates, path=str(args.output/'resume.pt'))
            save_training(args.output,learner,optimizer,updates,rank,world)
    if world>1: dist.barrier()
    if rank==0:
        # A short run retains resume state, but only a full epoch exports the model.
        if updates==total_updates:
            progress.phase('export_model', path=str(args.output/'model'))
            learner.save(args.output/'model')
            if not final_model_exists(args.output,args.method):
                raise RuntimeError('Model export did not produce the required files')
            write_json(args.output/'COMPLETE',dict(completed_updates=updates))
        protocol['completed_updates']=updates
        write_json(args.output/'training_summary.json',protocol)
    if world>1: dist.destroy_process_group()
    progress.phase('training_finished', completed_updates=updates, total_updates=total_updates)

if __name__=='__main__':main()
