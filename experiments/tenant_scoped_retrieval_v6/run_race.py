"""Stage 3: race trials revoking each query's victim; refuses to run unless stage-2 predictions are frozen."""
from common import *
def main():
 pred=D/'predictions.json'
 if not pred.exists():raise RuntimeError('Freeze predictions (predict.py) before running races')
 if (D/'race.jsonl').exists():raise RuntimeError('Refusing to overwrite completed evidence')
 rp=D/'race.partial.jsonl';done=resume(rp,lambda r:(r['representation'],r['seed'],r['wait_ms']),1280);dp=D/'race_design.json'
 rec={'stage3_sha256':digest(Path(__file__).read_bytes()),'common_sha256':digest((H/'common.py').read_bytes()),'predictions_sha256':digest(pred.read_bytes()),'kept_groups':sorted(map(list,done))}
 hist=json.loads(dp.read_text())['runs'] if dp.exists() else [];dp.write_text(json.dumps({'runs':hist+[rec]},indent=2)+'\n')
 env=Env();out=rp.open('a')
 try:
  for rep in CFG['representations']:
   for seed in CFG['seeds']:
    if all((rep,seed,w) in done for w in CFG['waits_ms']):continue
    ctx=env.load(rep,seed)
    for wait in CFG['waits_ms']:
     if (rep,seed,wait) in done:continue
     for arm in CFG['arms']:
      for q in ctx['qs'][:3]:env.run_retry(ctx,arm,q,wait)
     order,offsets,horizon=plan(seed,wait,2)
     for o,j,arm in order:
      q=ctx['qs'][j];victim=ctx['victims'][q['case_id']];r=env.trial(ctx,arm,q,wait,offsets[o,j,arm],[victim]);revoked=r['commit_ns']<r['end_ns'];recall,_=score(ctx,q,r['returned'],{victim} if revoked else ());_,stale=score(ctx,q,[i for i in r['returned'] if i!=victim])
      out.write(json.dumps({'representation':rep,'seed':seed,'wait_ms':wait,'arm':arm,'offset_index':o,'case_id':q['case_id'],'tenant':q['tenant'],'victim':victim,'horizon_ns':horizon,'revoked_before_end':revoked,'race_exposure':bool(victim in r['returned'] and revoked),'stale_exposure':stale,'recall':recall,**r})+'\n')
     out.flush();print('race',rep,seed,wait,flush=True)
    env.drop(ctx)
  out.close();rp.rename(D/'race.jsonl')
 finally:out.close();env.close()
 print('Stage 3 complete',flush=True)
if __name__=='__main__':main()
