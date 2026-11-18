"""Replay numeric scoring oracles; no source texts, vectors, network or model required."""
from pathlib import Path
import json,sys
D=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent/'evidence'
oracles={tuple(r['key']):r for r in (json.loads(s) for s in (D/'oracles.jsonl').read_text().splitlines())}
count=0
for line in (D/'requests.jsonl').read_text().splitlines():
 r=json.loads(line);o=oracles[(r['representation'],r['seed'],r['regime'],r['phase'],r['case_id'])];eligible=set(o['eligible']);got=set(r['returned']);assert r['exposure']==bool(got-eligible);assert r['false_denial']==bool(eligible and not got)
 recall=(len(got&set(o['strict']))+min(len(got&set(o['ties'])),o['n']-len(o['strict'])))/o['n'];assert abs(recall-r['recall'])<1e-7
 if r['arm'].startswith('G'):assert r['returned']==[i for i in r['candidates'] if i in eligible][:5]
 count+=1
assert count==46080
faults=json.loads((D/'faults.json').read_text());assert len(faults)==640 and all(not r['returned'] and r['gate_error'] for r in faults)
print(f'PASS: {count} numeric response scores and {len(faults)} fault probes. Oracle provenance requires the controlled source; this replay does not regenerate embeddings.')
