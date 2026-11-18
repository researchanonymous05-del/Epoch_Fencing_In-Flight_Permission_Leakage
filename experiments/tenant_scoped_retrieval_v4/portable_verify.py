"""Source-free replay against numeric scoring oracles and recorded policy decisions."""
from pathlib import Path
import json,sys
D=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent/'evidence'
oracles={tuple(r['key']):r for r in (json.loads(s) for s in (D/'oracles.jsonl').read_text().splitlines())};count=0
for line in (D/'requests.jsonl').read_text().splitlines():
 r=json.loads(line);o=oracles[r['representation'],r['seed'],r['rate_percent'],r['case_id']];owned=set(o['owned']);eligible=set(o['eligible']);ret=set(r['returned']);assert r['exposure']==bool(ret-eligible);assert r['false_denial']==(not ret)
 score=(len(ret&set(o['strict']))+min(len(ret&set(o['ties'])),5-len(o['strict'])))/5;assert abs(score-r['recall'])<1e-7
 for step in r['rounds']:
  assert set(step['candidates'])<=owned
  expect=[i for i in step['candidates'] if i in eligible][:5] if r['arm'].startswith('G') or r['arm']=='R' else step['candidates'][:5];assert step['returned']==expect
 if r['arm']=='P':assert set(r['rounds'][0]['candidates'])<=eligible;assert r['policy_calls'][0]['revoked']==sorted(owned-eligible)
 if r['arm']=='R':
  assert [x['budget'] for x in r['rounds']]==[5,10,20,40,80][:len(r['rounds'])]
  for step in r['rounds'][:-1]:assert len(step['returned'])<5 and len(step['candidates'])==step['budget']
 for call in r['policy_calls']:
  assert call['http_ms']+.02>=r['delay_ms']
  if call['action']=='check':assert call['allowed']==[i for i in call['ids'] if i in eligible]
 assert r['returned']==r['rounds'][-1]['returned'];count+=1
assert count==57600
print('PASS:',count,'numeric scores, placement checks, refill prefixes and response waits. Source oracle reconstruction is a separate local verification.')
