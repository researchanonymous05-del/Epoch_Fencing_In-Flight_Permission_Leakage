"""Independent checks: stage ordering and hashes, reconstructed victims, window contract, phantom isolation, counts."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,random,hashlib
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parents[1];D=H/'data'
def main():
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();pred=json.loads((D/'predictions.json').read_text())
 assert pred['stable_sha256']==sha(D/'stable.jsonl');runs=json.loads((D/'race_design.json').read_text())['runs'];assert all(r['predictions_sha256']==sha(D/'predictions.json') for r in runs)
 src=json.loads((ROOT/'data/benchmark.json').read_text());recs=src['records'];accs=sorted({r['account_id'] for r in recs});sup={r['account_id']:i for i,r in enumerate(recs) if r['kind']=='support'};qs={q['case_id']:q for q in src['cases'] if q['split']=='test'}
 vecs={'lexical':np.load(ROOT/'experiments/tenant_scoped_retrieval/data/vectors.npz')['documents'],'neural':np.load(ROOT/'experiments/tenant_scoped_retrieval_v3/data/neural_vectors.npy')};victim={};own_rec=0
 for rep,v in vecs.items():
  for seed in [11,29,47]:
   a=accs.copy();random.Random(seed).shuffle(a);o={x:f't{i%20:02}' for i,x in enumerate(a)};t=[o[r['account_id']] for r in recs];rm=set(random.Random(seed+9000).sample(range(1430),143))
   for cid,q in qs.items():
    s=v@v[sup[q['account_id']]];vic=max((i for i,x in enumerate(t) if x==o[q['account_id']] and i not in rm),key=lambda i:s[i]);victim[rep,seed,cid]=vic;own_rec+=vic==sup[q['account_id']]
 counts={}
 for name,phantom in [('stable.jsonl',True),('race.jsonl',False)]:
  n=0
  for line in (D/name).read_text().splitlines():
   r=json.loads(line);assert r['victim']==victim[r['representation'],r['seed'],r['case_id']];assert r['t0_ns']<=r['start_ns']<=r['end_ns'] and r['target_ns']<=r['commit_ns']
   inwin=(r['decisive_ns'] if r['decisive_ns'] is not None else -1)<r['commit_ns']<r['end_ns'];ret=r['victim'] in r['returned']
   if phantom:assert r['would_expose']==(ret and inwin);assert r['arm']=='A1' or not r['stale_exposure']
   else:
    assert r['race_exposure']==(ret and r['commit_ns']<r['end_ns'])
    if r['arm']!='A1':assert r['race_exposure']==(ret and inwin) and not r['stale_exposure']
   n+=1
  counts[name]=n
 assert counts=={'stable.jsonl':15360,'race.jsonl':15360}
 ch=sum(1 for _ in (D/'churn.jsonl').read_text().splitlines());assert ch==7680
 rep={'status':'passed','stable':15360,'race':15360,'churn':ch,'victim_is_own_support_record':own_rec/len(victim)};(D/'verification.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep))
if __name__=='__main__':main()
