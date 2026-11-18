"""Independently reconstruct score oracles and replay every prefilter/refill decision."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
from collections import Counter
import json,random
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parents[1];D=H/'data'
def main():
 source=json.loads((ROOT/'data/benchmark.json').read_text());records=source['records'];accounts=sorted({r['account_id'] for r in records});support={r['account_id']:i for i,r in enumerate(records) if r['kind']=='support'};queries=[q for q in source['cases'] if q['split']=='test'];vectorsets={'lexical':np.load(ROOT/'experiments/tenant_scoped_retrieval/data/vectors.npz')['documents'],'neural':np.load(ROOT/'experiments/tenant_scoped_retrieval_v3/data/neural_vectors.npy')};oracles={}
 for rep,vectors in vectorsets.items():
  for seed in [11,29,47]:
   a=accounts.copy();random.Random(seed).shuffle(a);owner={x:f't{i%20:02}' for i,x in enumerate(a)};tenant=[owner[r['account_id']] for r in records];scores={q['case_id']:vectors@vectors[support[q['account_id']]] for q in queries}
   for rate,n in [(5,72),(10,143),(25,358)]:
    revoked=set(random.Random(seed+9000).sample(range(1430),n))
    for q in queries:
     owned=[i for i,t in enumerate(tenant) if t==owner[q['account_id']]];eligible=[i for i in owned if i not in revoked];s=scores[q['case_id']];cut=sorted(s[eligible],reverse=True)[4];strict=[i for i in eligible if s[i]>cut+1e-5];ties=[i for i in eligible if abs(float(s[i]-cut))<=1e-5];oracles[rep,seed,rate,q['case_id']]={'owned':owned,'eligible':eligible,'strict':strict,'ties':ties,'n':5}
 counts=Counter();rows=0;prefilter=0;refill=0;waitcalls=0
 for line in (D/'requests.jsonl').read_text().splitlines():
  r=json.loads(line);o=oracles[r['representation'],r['seed'],r['rate_percent'],r['case_id']];allowed=set(o['eligible']);owned=set(o['owned']);got=set(r['returned']);assert r['exposure']==bool(got-allowed);assert r['false_denial']==(not got)
  score=(len(got&set(o['strict']))+min(len(got&set(o['ties'])),5-len(o['strict'])))/5;assert abs(score-r['recall'])<1e-7
  for step in r['rounds']:
   assert set(step['candidates'])<=owned
   expected=[i for i in step['candidates'] if i in allowed][:5] if r['arm'].startswith('G') or r['arm']=='R' else step['candidates'][:5];assert step['returned']==expected
  assert r['returned']==r['rounds'][-1]['returned']
  for call in r['policy_calls']:
   assert r['start_ns']<=call['snapshot_ns']<=r['end_ns'];assert call['http_ms']+.02>=r['delay_ms'];waitcalls+=int(r['delay_ms']>0)
   if call['action']=='check':assert call['allowed']==[i for i in call['ids'] if i in allowed]
   else:assert call['revoked']==sorted(owned-allowed)
  if r['arm']=='P':
   assert len(r['policy_calls'])==1 and len(r['rounds'])==1;assert set(r['rounds'][0]['candidates'])<=allowed
   excluded=sorted(owned-allowed);expected=f'tenant:={r["tenant"]} && active:=true'+(' && rid:!=['+','.join(map(str,excluded))+']' if excluded else '');assert r['generated_filter']==expected;prefilter+=1
  elif r['arm']=='R':
   assert [s['budget'] for s in r['rounds']]==[5,10,20,40,80][:len(r['rounds'])];assert len(r['policy_calls'])==len(r['rounds'])
   for step in r['rounds'][:-1]:assert len(step['returned'])<5 and len(step['candidates'])==step['budget']
   last=r['rounds'][-1];assert len(last['returned'])==5 or len(last['candidates'])<last['budget'] or last['budget']==80;refill+=1
  elif r['arm'].startswith('G'):assert len(r['policy_calls'])==1 and r['rounds'][0]['budget']==int(r['arm'][1:])
  else:assert not r['policy_calls']
  counts[r['representation'],r['rate_percent'],r['delay_ms'],r['arm']]+=1;rows+=1
 assert rows==57600 and len(counts)==60 and set(counts.values())=={960}
 with (D/'oracles.jsonl').open('w') as out:
  for key,o in oracles.items():out.write(json.dumps({'key':list(key),**o})+'\n')
 report={'status':'passed','independently_rescored_requests':rows,'conditions':len(counts),'timings_per_condition':960,'case_seed_units_per_condition':480,'prefilter_requests_checked':prefilter,'refill_requests_checked':refill,'delayed_policy_calls_checked':waitcalls,'source_oracles':len(oracles)};(D/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
