"""Independently reconstruct ownership, base revocations and victims; check every fence/race trace against the snapshot contract."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,random
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parents[1];D=H/'data_fence';P=H/'data_prospective'
def main():
 source=json.loads((ROOT/'data/benchmark.json').read_text());records=source['records'];accounts=sorted({r['account_id'] for r in records});support={r['account_id']:i for i,r in enumerate(records) if r['kind']=='support'};queries={q['case_id']:q for q in source['cases'] if q['split']=='test'}
 sets={'lexical':np.load(ROOT/'experiments/tenant_scoped_retrieval/data/vectors.npz')['documents'],'neural':np.load(ROOT/'experiments/tenant_scoped_retrieval_v3/data/neural_vectors.npy')};ctx={}
 for rep,vectors in sets.items():
  for seed in [11,29,47]:
   a=accounts.copy();random.Random(seed).shuffle(a);owner={x:f't{i%20:02}' for i,x in enumerate(a)};tenant=[owner[r['account_id']] for r in records];removed=set(random.Random(seed+9000).sample(range(1430),143))
   for cid,q in queries.items():
    s=vectors@vectors[support[q['account_id']]];eligible=[i for i,t in enumerate(tenant) if t==owner[q['account_id']] and i not in removed];ctx[rep,seed,cid]={'tenant':owner[q['account_id']],'owned':{i for i,t in enumerate(tenant) if t==owner[q['account_id']]},'removed':removed,'victim':max(eligible,key=lambda i:s[i])}
 checked=0
 for line in (D/'race.jsonl').read_text().splitlines():
  r=json.loads(line);c=ctx[r['representation'],r['seed'],r['case_id']];assert r['tenant']==c['tenant'] and r['victim']==c['victim']
  ret=r['returned'];assert set(ret)<=c['owned'];assert r['revoked_before_end']==(r['commit_ns']<r['end_ns']);assert r['race_exposure']==(c['victim'] in ret and r['revoked_before_end']);assert r['stale_exposure']==bool(set(ret)&c['removed'])
  assert r['start_ns']<=r['end_ns'] and r['t0_ns']<=r['start_ns'] and r['target_ns']<=r['commit_ns']
  for call in r['policy_calls']:
   assert r['start_ns']<=call['snapshot_ns']<=r['end_ns'];revoked_now=c['removed']|({c['victim']} if call['snapshot_ns']>r['commit_ns'] else set())
   if call['action']=='list':assert call['revoked']==sorted(i for i in revoked_now if i in c['owned'])
   elif call['action']=='check' or not call.get('fast'):assert call['allowed']==[i for i in call['ids'] if i not in revoked_now]
   else:assert call['epoch']==call['epoch_in'] and call['allowed']==call['ids'] and not (call['snapshot_ns']>r['commit_ns'] and r['commit_epoch']>call['epoch_in'])
  if r['arm']!='A1':
   # Contract: with a stable authority after the last decisive snapshot, exposure occurs only if the commit lands after it.
   last=[x for x in r['policy_calls'] if x['action']=={'P':'list','G20':'check','F':'fence'}[r['arm']]][-1]
   assert r['race_exposure']==(c['victim'] in ret and last['snapshot_ns']<r['commit_ns']<r['end_ns'])
   assert not r['stale_exposure']
  if r['arm']=='F':
   assert 1<=len(r['attempts'])<=2;assert all(not a['dropped'] for a in r['attempts'][-1:]) or len(r['attempts'])==2
  checked+=1
 cost=0
 for line in (D/'cost.jsonl').read_text().splitlines():
  r=json.loads(line);c=ctx[r['representation'],r['seed'],r['case_id']];assert not set(r['returned'])&c['removed'] and set(r['returned'])<=c['owned'];assert not r['exposure'];cost+=1
  if r['arm']=='F':assert len(r['attempts'])==1 and all(x['fast'] for x in r['policy_calls'] if x['action']=='fence')
 report={'status':'passed','race_trials':checked,'cost_rows':cost}
 if (P/'requests.jsonl').exists():
  n=0
  for line in (P/'requests.jsonl').read_text().splitlines():
   r=json.loads(line);seed=r['seed'];removed=set(random.Random(seed+9000).sample(range(1430),r['revoked_count']));assert not set(r['returned'])&removed and not r['exposure'];n+=1
  report['prospective_rows']=n
 (D/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
