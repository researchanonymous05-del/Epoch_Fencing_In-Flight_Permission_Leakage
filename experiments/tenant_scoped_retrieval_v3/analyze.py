"""Deduplicate security outcomes; retain repeated timings and observed concurrent windows."""
from pathlib import Path
from collections import defaultdict
import json,hashlib
import numpy as np
H=Path(__file__).resolve().parent;D=H/'data'
def read(name):return [json.loads(s) for s in (D/name).read_text().splitlines()]
def main():
 rows=read('requests.jsonl');groups=defaultdict(list)
 for r in rows:groups[r['representation'],r['regime'],r['phase'],r['arm']].append(r)
 summary=[]
 for key,rs in sorted(groups.items()):
  unique=[r for r in rs if r['repeat']==0];assert len(unique)==480
  summary.append(dict(zip(['representation','regime','phase','arm'],key))|{'case_seed_units':len(unique),'queries':160,'timed_requests':len(rs),'exposed_units':sum(r['exposure'] for r in unique),'false_denials':sum(r['false_denial'] for r in unique),'mean_recall':float(np.mean([r['recall'] for r in unique])),'p50_ms':float(np.median([r['latency_ms'] for r in rs])),'p95_ms':float(np.percentile([r['latency_ms'] for r in rs],95)),'policy_p50_ms':float(np.median([r['policy_http_ms'] for r in rs])),'mean_candidates':float(np.mean([len(r['candidates']) for r in unique]))})
 bykey={(r['representation'],r['seed'],r['regime'],r['phase'],r['case_id'],r['repeat'],r['arm']):r for r in rows};pairs=[]
 for rep in ['lexical','neural']:
  for regime in ['targeted','random10']:
   for phase in ['current','stale','applied']:
    for arm in ['G5','G20']:
     ds=[]
     for r in rows:
      if (r['representation'],r['regime'],r['phase'],r['arm'])!=(rep,regime,phase,arm):continue
      key=(rep,r['seed'],regime,phase,r['case_id'],r['repeat'],'A1');ds.append(r['latency_ms']-bykey[key]['latency_ms'])
     pairs.append({'representation':rep,'regime':regime,'phase':phase,'arm':arm,'pairs':len(ds),'median_added_ms':float(np.median(ds)),'p95_added_ms':float(np.percentile(ds,95))})
 crows=read('concurrent.jsonl');windows=[]
 for trial in json.loads((D/'windows.json').read_text()):
  for arm in ['A1','A2','G5','G20']:
   rs=sorted([r for r in crows if r['representation']==trial['representation'] and r['seed']==trial['seed'] and r['arm']==arm],key=lambda r:r['end_ns']);exposed=[r for r in rs if r['exposes_revoked']];start=trial['commit_ns'];ts=[r['end_ns']/1e6 for r in rs]
   windows.append({'representation':trial['representation'],'seed':trial['seed'],'arm':arm,'delay_ms':trial['injected_delay_ms'],'observations':len(rs),'exposed':len(exposed),'last_observed_exposure_ms':max((r['end_ns']-start)/1e6 for r in exposed) if exposed else None,'all_updates_ack_ms':(trial['all_updates_ack_ns']-start)/1e6,'after_ack_started':sum(r['start_ns']>trial['all_updates_ack_ns'] for r in rs),'after_ack_exposed':sum(r['start_ns']>trial['all_updates_ack_ns'] and r['exposes_revoked'] for r in rs),'max_interresponse_gap_ms':max(np.diff(ts)) if len(ts)>1 else None})
 faults=json.loads((D/'faults.json').read_text());fs=[]
 for rep in ['lexical','neural']:
  for arm in ['G5','G20']:
   rs=[r for r in faults if (r['representation'],r['arm'])==(rep,arm)];fs.append({'representation':rep,'arm':arm,'requests':len(rs),'empty_results':sum(not r['returned'] for r in rs),'gate_errors':sum(r['gate_error'] for r in rs)})
 result={'groups':summary,'paired_cost':pairs,'windows':windows,'faults':fs,'timed_requests':len(rows),'security_units_per_group':480,'concurrent_observations':len(crows),'security_rule':'repeat zero only; repeat one used only for timing; same 160 queries across assignments and representations','runner_sha256_matches':hashlib.sha256((H/'run.py').read_bytes()).hexdigest()==json.loads((D/'design.json').read_text())['runner_sha256']}
 (D/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 for g in summary:
  if g['phase']=='stale':print(g['representation'],g['regime'],g['arm'],g['exposed_units'],'/480','recall',round(g['mean_recall'],3),'FD',g['false_denials'],'p50/p95',round(g['p50_ms'],2),round(g['p95_ms'],2))
 print('Concurrent windows:',windows)
if __name__=='__main__':main()
