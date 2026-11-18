"""Summarize list/header outcomes, pagination work and concurrent admission."""
from pathlib import Path
from collections import defaultdict,Counter
import json,hashlib
import numpy as np
H=Path(__file__).resolve().parent;D=H/'data'
def main():
 rows=[json.loads(s) for s in (D/'requests.jsonl').read_text().splitlines()];groups=defaultdict(list)
 for r in rows:groups[r['suite'],r['representation'],r['requested_list_size'],r['arm']].append(r)
 stats=[]
 for key,rs in sorted(groups.items()):
  units=[r for r in rs if r['repeat']==0];ok=[r for r in rs if r['success']];errors=Counter(str(r['status'])+':'+r.get('error_type','') for r in rs if not r['success']);lat=[r['latency_ms'] for r in ok]
  stats.append(dict(zip(['suite','representation','list_size','arm'],key))|{'units':len(units),'timings':len(rs),'successful_timings':len(ok),'failed_timings':len(rs)-len(ok),'errors':dict(errors),'success_fraction':len(ok)/len(rs),'success_p50_ms':float(np.median(lat)) if lat else None,'success_p95_ms':float(np.percentile(lat,95)) if lat else None,'failure_p50_ms':float(np.median([r['latency_ms'] for r in rs if not r['success']])) if len(ok)<len(rs) else None,'exposed_units':sum(r['exposure'] for r in units),'false_denials':sum(r['false_denial'] for r in units),'mean_recall_including_failures':float(np.mean([r['recall'] for r in units])),'mean_ann_calls':float(np.mean([len(r['rounds']) for r in units])),'mean_policy_calls':float(np.mean([len(r['policy_calls']) for r in units])),'mean_delivered_candidates':float(np.mean([sum(len(step['candidates']) for step in r['rounds']) for r in units])),'max_key_bytes':max(r['key_bytes'] for r in rs),'max_filter_bytes':max(r['filter_bytes'] for r in rs)})
 paired=[];lookup={(r['suite'],r['representation'],r['seed'],r['requested_list_size'],r['case_id'],r['repeat'],r['arm']):r for r in rows}
 for key,rs in sorted(groups.items()):
  suite,rep,n,arm=key
  if arm=='G20':continue
  ds=[]
  for r in rs:
   comparator=lookup[suite,rep,r['seed'],n,r['case_id'],r['repeat'],'G20']
   if r['success'] and comparator['success']:ds.append(r['latency_ms']-comparator['latency_ms'])
  paired.append({'suite':suite,'representation':rep,'list_size':n,'arm':arm,'successful_pairs':len(ds),'median_vs_G20_ms':float(np.median(ds)) if ds else None})
 concurrent=[json.loads(s) for s in (D/'concurrent.jsonl').read_text().splitlines()];ws=[]
 for rep in ['lexical','neural']:
  for arm in ['A1','G20','P','R','RP']:
   rs=[r for r in concurrent if (r['representation'],r['arm'])==(rep,arm)];exposed=[r for r in rs if r['exposes_revoked']];ws.append({'representation':rep,'arm':arm,'responses':len(rs),'successful':sum(r['success'] for r in rs),'exposed':len(exposed),'last_observed_exposure_ms':max((r['end_ns']-r['commit_ns'])/1e6 for r in exposed) if exposed else None,'after_ack_started':sum(r['start_ns']>r['update_ack_ns'] for r in rs),'after_ack_exposed':sum(r['start_ns']>r['update_ack_ns'] and r['exposes_revoked'] for r in rs)})
 result={'serial_timings':len(rows),'groups':stats,'paired':paired,'concurrent_responses':len(concurrent),'concurrent_groups':ws,'frozen_runner_hash_matches':hashlib.sha256((H/'run.py').read_bytes()).hexdigest()==json.loads((D/'design.json').read_text())['runner_sha256']};(D/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 for g in stats:print(g['suite'],g['representation'],g['list_size'],g['arm'],'ok',g['successful_timings'],'/',g['timings'],'p50',g['success_p50_ms'],'key',g['max_key_bytes'],'filter',g['max_filter_bytes'],'errors',g['errors'],'recall',g['mean_recall_including_failures'])
 print('Concurrent',ws)
if __name__=='__main__':main()
