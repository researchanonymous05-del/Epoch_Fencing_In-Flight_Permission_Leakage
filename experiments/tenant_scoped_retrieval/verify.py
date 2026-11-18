"""Independent replay of saved response scoring from source, vectors and policy ledger."""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,random,hashlib
from collections import Counter
import numpy as np
D=Path(__file__).resolve().parent/'data';ROOT=D.parents[2]
b=json.loads((ROOT/'data/benchmark.json').read_text());records=b['records'];accounts=sorted({r['account_id'] for r in records});vectors=np.load(D/'vectors.npz')['documents'];cases={c['case_id']:c for c in b['cases']};support={r['account_id']:i for i,r in enumerate(records) if r['kind']=='support'}
rows=[json.loads(s) for s in (D/'requests.jsonl').read_text().splitlines()];policies={};revokes={}
for seed in [11,29,47]:
 a=accounts.copy();random.Random(seed).shuffle(a);assignment={x:f't{i%20:02}' for i,x in enumerate(a)};tenant=[assignment[r['account_id']] for r in records];policies[seed]=tenant;removed=set()
 for c in b['cases']:
  if c['split']!='test':continue
  q=vectors[support[c['account_id']]];scores=vectors@q;eligible=[i for i,r in enumerate(records) if assignment[r['account_id']]==assignment[c['account_id']]];removed.add(max(eligible,key=lambda i:scores[i]))
 revokes[seed]=removed
for r in rows:
 seed=r['seed'];tenant=policies[seed];revoked=revokes[seed] if r['phase']!='current' else set();expected=[i for i in range(len(records)) if tenant[i]==r['tenant'] and i not in revoked];assert len(expected)==r['eligible'];scores=vectors@vectors[support[cases[r['case_id']]['account_id']]]
 unauthorized=[i for i in r['returned'] if i not in expected];assert len(unauthorized)==r['unauthorized_hits']
 assert sum(tenant[i]!=r['tenant'] for i in r['returned'])==r['cross_tenant_hits']
 n=min(5,len(expected));cut=sorted(scores[expected],reverse=True)[n-1];strict={i for i in expected if scores[i]>cut+1e-5};tied={i for i in expected if abs(float(scores[i]-cut))<=1e-5};got=set(r['returned']);recall=(len(got&strict)+min(len(got&tied),n-len(strict)))/n
 assert abs(recall-r['recall_at_5'])<1e-8
assert len(rows)==12960
counts=Counter((r['seed'],r['phase'],r['arm']) for r in rows);assert set(counts.values())=={480}
report={'requests_independently_rescored':len(rows),'groups':len(counts),'each_group_requests':480,'source_labels':'constructed not human annotated','revoked_documents_by_seed':{str(k):len(v) for k,v in revokes.items()},'status':'passed'}
(D/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
