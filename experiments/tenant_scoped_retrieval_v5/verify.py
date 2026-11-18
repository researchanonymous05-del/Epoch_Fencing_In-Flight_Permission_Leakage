"""Reconstruct independent oracles; check key/filter sizes, paging and concurrent admission."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
from collections import Counter
import json,random,hashlib
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parents[1];D=H/'data'
def main():
 source=json.loads((ROOT/'data/benchmark.json').read_text());records=source['records'];accounts=sorted({r['account_id'] for r in records});queries=[q for q in source['cases'] if q['split']=='test'];support={r['account_id']:i for i,r in enumerate(records) if r['kind']=='support'};sets={'lexical':np.load(ROOT/'experiments/tenant_scoped_retrieval/data/vectors.npz')['documents'],'neural':np.load(ROOT/'experiments/tenant_scoped_retrieval_v3/data/neural_vectors.npy')};oracles={};targets={}
 for rep,vectors in sets.items():
  for seed in [11,29,47]:
   a=accounts.copy();random.Random(seed).shuffle(a);owner={x:f't{i%20:02}' for i,x in enumerate(a)};tenant=[owner[r['account_id']] for r in records];scores={q['case_id']:vectors@vectors[support[q['account_id']]] for q in queries};selected=set()
   for q in queries:
    owned=[i for i,t in enumerate(tenant) if t==owner[q['account_id']]];selected.add(max(owned,key=lambda i:scores[q['case_id']][i]))
   targets[rep,seed]=selected
   for mode,removed in [('random',set(random.Random(seed+9000).sample(range(1430),143))),('targeted',selected)]:
    for q in queries:
     owned=[i for i,t in enumerate(tenant) if t==owner[q['account_id']]];allowed=[i for i in owned if i not in removed];s=scores[q['case_id']];cut=sorted(s[allowed],reverse=True)[4];oracles[rep,seed,mode,q['case_id']]={'owned':owned,'eligible':allowed,'strict':[i for i in allowed if s[i]>cut+1e-5],'ties':[i for i in allowed if abs(float(s[i]-cut))<=1e-5],'n':5}
 def check(r,mode,padding):
  o=oracles[r['representation'],r['seed'],mode,r['case_id']];owned=set(o['owned']);allowed=set(o['eligible']);ret=set(r['returned']);assert r['success'];assert ret<=allowed if r['arm']!='A1' else ret<=owned
  if r['arm']=='RP':
   # The runner stored the same mutable 'seen' list in every RP call trace, so each
   # call's ids show the final cumulative list. Rebuild the IDs sent at each round.
   seen=[]
   for j,step in enumerate(r['rounds']):
    for i in step['candidates']:
     if i not in seen:seen.append(i)
    assert r['policy_calls'][j]['ids'][:len(seen)]==seen;r['policy_calls'][j]=dict(r['policy_calls'][j],ids=list(seen))
  for call in r['policy_calls']:
   assert r['start_ns']<=call['snapshot_ns']<=r['end_ns']
   if call['action']=='list':
    real=sorted(owned-allowed);first=1_000_000+int(r['tenant'][1:])*10000;ids=real+list(range(first,first+max(0,padding-len(real)))) if padding else real
    assert call['list_count']==len(ids);assert call['list_sha256']==hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
   else:assert call['allowed']==[i for i in call['ids'] if i in allowed]
  if r['arm'] in ['P','Pbody']:
   real=sorted(owned-allowed);first=1_000_000+int(r['tenant'][1:])*10000;ids=real+list(range(first,first+max(0,padding-len(real)))) if padding else real;clause=f'tenant:={r["tenant"]} && active:=true'+(' && rid:!=['+','.join(map(str,ids))+']' if ids else '')
   assert r['filter_bytes']==len(clause.encode());assert r['filter_sha256']==hashlib.sha256(clause.encode()).hexdigest()
   if r['arm']=='P':
    params=json.dumps({'filter_by':clause,'exclude_fields':'vec'},separators=(',',':')).encode();assert r['key_bytes']==4*((48+len(params)+2)//3)
   assert all(i in allowed for i in r['rounds'][0]['candidates'])
  seen=[]
  for j,step in enumerate(r['rounds']):
   assert set(step['candidates'])<=owned
   if r['arm']=='RP':
    assert step['page']==j+1 and step['budget']==80 and len(step['candidates'])<=5
    for i in step['candidates']:
     if i not in seen:seen.append(i)
    assert r['policy_calls'][j]['ids']==seen;expected=[i for i in seen if i in allowed][:5]
   elif r['arm'] in ['R','G20']:assert r['policy_calls'][j]['ids']==step['candidates'];expected=[i for i in step['candidates'] if i in allowed][:5]
   else:expected=step['candidates'][:5]
   assert step['returned']==expected
   if j<len(r['rounds'])-1:assert len(step['returned'])<5
  assert r['returned']==r['rounds'][-1]['returned']
  return (len(ret&set(o['strict']))+min(len(ret&set(o['ties'])),5-len(o['strict'])))/5
 count=0;groups=Counter()
 for line in (D/'requests.jsonl').read_text().splitlines():
  r=json.loads(line);score=check(r,'random',r['requested_list_size']);assert abs(score-r['recall'])<1e-7;assert not r['exposure'];assert r['false_denial']==(not r['returned']);groups[r['suite'],r['representation'],r['requested_list_size'],r['arm']]+=1;count+=1
 assert count==12000 and len(groups)==26
 for key,n in groups.items():assert n==(240 if key[0]=='lists' else 960)
 concurrent=0;protected=0
 for line in (D/'concurrent.jsonl').read_text().splitlines():
  r=json.loads(line);check(r,'targeted',0);assert r['start_ns']>=r['commit_ns'];assert r['exposes_revoked']==bool(set(r['returned'])&targets[r['representation'],r['seed']]);concurrent+=1
  if r['arm']!='A1':assert not r['exposes_revoked'];protected+=1
  if r['start_ns']>r['update_ack_ns']:assert not r['exposes_revoked']
 with (D/'oracles.jsonl').open('w') as out:
  for key,value in oracles.items():out.write(json.dumps({'key':list(key),**value})+'\n')
 report={'status':'passed','serial_rows':count,'conditions':26,'concurrent_rows':concurrent,'protected_concurrent_rows':protected,'source_oracles':len(oracles),'padding':'nonindexed IDs; list representation and transport scaling only','size_limit_observation':'all measured requests completed; no rejection at tested maximum, not an unlimited-size guarantee'};(D/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
