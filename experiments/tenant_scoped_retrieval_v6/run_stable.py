"""Stage 1: stable trials (phantom commits only) for windows, counterfactual exposure and cost; churn run for F vs Fg.
Resumable: complete (representation, seed, wait) groups in an interrupted partial file are kept; incomplete ones rerun."""
from common import *
GROUP=1280
def main():
 if (D/'stable.jsonl').exists():raise RuntimeError('Refusing to overwrite completed evidence')
 sp,cp=D/'stable.partial.jsonl',D/'churn.partial.jsonl'
 done=resume(sp,lambda r:(r['representation'],r['seed'],r['wait_ms']),GROUP);cdone=resume(cp,lambda r:(r['representation'],r['seed']),GROUP)
 design=D/'design.json';rec={'config':CFG,'stage1_sha256':digest(Path(__file__).read_bytes()),'common_sha256':digest((H/'common.py').read_bytes()),'authority_sha256':digest((H/'authority.py').read_bytes()),'revoker_sha256':digest((H/'revoker.py').read_bytes()),'source_sha256':digest(SOURCE.read_bytes())}
 if design.exists():
  d=json.loads(design.read_text());d.setdefault('resumes',[]).append({**{k:v for k,v in rec.items() if k!='config'},'kept_stable_groups':sorted(map(list,done)),'kept_churn_groups':sorted(map(list,cdone)),'reason':'Typesense HTTP 408 aborted the run; 408 retry added'});design.write_text(json.dumps(d,indent=2)+'\n')
 else:design.write_text(json.dumps(rec,indent=2)+'\n')
 env=Env();out=sp.open('a');churn=cp.open('a')
 try:
  for rep in CFG['representations']:
   for seed in CFG['seeds']:
    if all((rep,seed,w) in done for w in CFG['waits_ms']) and (rep,seed) in cdone:continue
    ctx=env.load(rep,seed)
    for wait in CFG['waits_ms']:
     if (rep,seed,wait) in done:continue
     for arm in CFG['arms']:
      for q in ctx['qs'][:3]:env.run_retry(ctx,arm,q,wait)
     order,offsets,horizon=plan(seed,wait,1)
     for o,j,arm in order:
      q=ctx['qs'][j];victim=ctx['victims'][q['case_id']];r=env.trial(ctx,arm,q,wait,offsets[o,j,arm],[PHANTOM]);recall,stale=score(ctx,q,r['returned'])
      lo=r['decisive_ns'] if r['decisive_ns'] is not None else -1;would=bool(victim in r['returned'] and lo<r['commit_ns']<r['end_ns'])
      out.write(json.dumps({'representation':rep,'seed':seed,'wait_ms':wait,'arm':arm,'offset_index':o,'case_id':q['case_id'],'tenant':q['tenant'],'victim':victim,'horizon_ns':horizon,'victim_returned':victim in r['returned'],'would_expose':would,'recall':recall,'stale_exposure':stale,**r})+'\n')
     out.flush();print('stable',rep,seed,wait,flush=True)
    if (rep,seed) not in cdone:
     # Churn: phantom revocations across all tenants at a fixed rate; cost and fast-path rate of tenant vs global epochs.
     env.ctl(action='churn',rate=CFG['churn_rate_per_s'])
     for arm in CFG['churn_arms']:
      for q in ctx['qs'][:3]:env.run_retry(ctx,arm,q,0)
     order=[(r_i,j,arm) for r_i in range(2) for j in range(160) for arm in CFG['churn_arms']];random.Random(seed+31).shuffle(order)
     for r_i,j,arm in order:
      q=ctx['qs'][j];r=env.run_retry(ctx,arm,q,0);recall,stale=score(ctx,q,r['returned'])
      churn.write(json.dumps({'representation':rep,'seed':seed,'arm':arm,'repeat':r_i,'case_id':q['case_id'],'tenant':q['tenant'],'recall':recall,'stale_exposure':stale,'fast':[c.get('fast') for c in r['policy_calls'] if c['action']=='fence'],**r})+'\n')
     env.ctl(action='churn',rate=0);churn.flush();print('churn',rep,seed,flush=True)
    env.drop(ctx)
  out.close();churn.close();sp.rename(D/'stable.jsonl');cp.rename(D/'churn.jsonl')
 finally:out.close();churn.close();env.close()
 print('Stage 1 complete',flush=True)
if __name__=='__main__':main()
