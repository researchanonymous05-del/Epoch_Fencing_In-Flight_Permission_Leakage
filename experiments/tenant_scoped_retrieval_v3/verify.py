"""Independent scoring and admission-contract replay; emits content-free scoring oracles."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
from collections import Counter
import json,random,hashlib
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parents[1];D=H/'data'
def main():
 source=json.loads((ROOT/'data/benchmark.json').read_text());records=source['records'];accounts=sorted({r['account_id'] for r in records});queries=[c for c in source['cases'] if c['split']=='test'];lookup={c['case_id']:c for c in queries};support={r['account_id']:i for i,r in enumerate(records) if r['kind']=='support'}
 vectorsets={'lexical':np.load(ROOT/'experiments/tenant_scoped_retrieval/data/vectors.npz')['documents'],'neural':np.load(D/'neural_vectors.npy')};oracles={};targets={};tenants={}
 for rep,vectors in vectorsets.items():
  for seed in [11,29,47]:
   a=accounts.copy();random.Random(seed).shuffle(a);owners={x:f't{i%20:02}' for i,x in enumerate(a)};t=[owners[r['account_id']] for r in records];tenants[rep,seed]=t;selected=set()
   allscores={c['case_id']:vectors@vectors[support[c['account_id']]] for c in queries}
   for c in queries:
    allowed=[i for i,v in enumerate(t) if v==owners[c['account_id']]];scores=allscores[c['case_id']];selected.add(max(allowed,key=lambda i:scores[i]))
   targets[rep,seed]=selected
   for regime,revoked in [('targeted',selected),('random10',set(random.Random(seed+9000).sample(range(1430),143)))]:
    for phase in ['current','stale','applied']:
     for c in queries:
      eligible=[i for i,v in enumerate(t) if v==owners[c['account_id']] and (phase=='current' or i not in revoked)];s=allscores[c['case_id']];n=min(5,len(eligible));cut=sorted(s[eligible],reverse=True)[n-1];strict=[i for i in eligible if s[i]>cut+1e-5];tied=[i for i in eligible if abs(float(s[i]-cut))<=1e-5]
      oracles[rep,seed,regime,phase,c['case_id']]={'eligible':eligible,'strict':strict,'ties':tied,'n':n}
 count=0;groups=Counter()
 for line in (D/'requests.jsonl').read_text().splitlines():
  r=json.loads(line);key=(r['representation'],r['seed'],r['regime'],r['phase'],r['case_id']);o=oracles[key];allowed=set(o['eligible']);ret=set(r['returned']);assert r['exposure']==bool(ret-allowed);assert r['false_denial']==bool(allowed and not ret)
  expected=(len(ret&set(o['strict']))+min(len(ret&set(o['ties'])),o['n']-len(o['strict'])))/o['n'];assert abs(expected-r['recall'])<1e-7
  if r['arm'].startswith('G'):
   assert r['returned']==[i for i in r['candidates'] if i in allowed][:5];assert r['policy_snapshot_ns']>=r['start_ns'];assert not r['gate_error']
  assert all(tenants[r['representation'],r['seed']][i]==r['tenant'] for i in r['returned'])
  groups[r['representation'],r['regime'],r['phase'],r['arm']]+=1;count+=1
 assert count==46080 and len(groups)==48 and set(groups.values())=={960}
 windows=0
 for line in (D/'concurrent.jsonl').read_text().splitlines():
  r=json.loads(line);removed=targets[r['representation'],r['seed']];assert r['exposes_revoked']==bool(set(r['returned'])&removed);assert r['start_ns']>=r['commit_ns']
  if r['arm'].startswith('G'):assert not r['exposes_revoked'];assert r['policy_snapshot_ns']>=r['commit_ns']
  if r['start_ns']>r['all_updates_ack_ns']:assert not r['exposes_revoked']
  windows+=1
 faults=json.loads((D/'faults.json').read_text());assert len(faults)==640 and all(r['gate_error'] and not r['returned'] for r in faults)
 # Content-free oracle supports portable replay without redistributing embeddings or catalog text.
 with (D/'oracles.jsonl').open('w') as f:
  for key,value in oracles.items():f.write(json.dumps({'key':list(key),**value})+'\n')
 report={'status':'passed','independent_scoring_rows':count,'admission_contract_rows':count//2,'concurrent_rows_verified':windows,'fault_rows_verified':len(faults),'groups':len(groups),'requests_per_group':960,'case_seed_units_per_group':480,'privacy':'oracle contains only synthetic IDs and numerical ranks; no catalog text or embeddings'}
 (D/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
