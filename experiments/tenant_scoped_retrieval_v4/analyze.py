"""Summarize sweeps over case-seed units and paired timing repetitions."""
from pathlib import Path
from collections import defaultdict
import json,hashlib
import numpy as np
H=Path(__file__).resolve().parent;D=H/'data'
def main():
 rows=[json.loads(s) for s in (D/'requests.jsonl').read_text().splitlines()];groups=defaultdict(list)
 for r in rows:groups[r['representation'],r['rate_percent'],r['delay_ms'],r['arm']].append(r)
 result=[]
 for key,rs in sorted(groups.items()):
  units=[r for r in rs if r['repeat']==0];assert len(units)==480 and len(rs)==960
  result.append(dict(zip(['representation','rate_percent','delay_ms','arm'],key))|{'security_units':480,'timed_requests':960,'exposed_units':sum(r['exposure'] for r in units),'false_denials':sum(r['false_denial'] for r in units),'mean_recall':float(np.mean([r['recall'] for r in units])),'p50_ms':float(np.median([r['latency_ms'] for r in rs])),'p95_ms':float(np.percentile([r['latency_ms'] for r in rs],95)),'mean_ann_calls':float(np.mean([len(r['rounds']) for r in units])),'mean_policy_calls':float(np.mean([len(r['policy_calls']) for r in units])),'max_ann_calls':max(len(r['rounds']) for r in units),'max_budget':max(r['rounds'][-1]['budget'] for r in units),'mean_policy_http_ms':float(np.mean([r['policy_http_ms'] for r in rs])),'short_units':sum(len(r['returned'])<5 for r in units)})
 bykey={(r['representation'],r['seed'],r['rate_percent'],r['delay_ms'],r['case_id'],r['repeat'],r['arm']):r for r in rows};pairs=[]
 for key,rs in sorted(groups.items()):
  rep,rate,delay,arm=key
  if arm in ['A1','A2']:continue
  ds=[];which='same-strategy zero-delay' if delay else 'A1 zero-delay'
  for r in rs:
   comparator=bykey[rep,r['seed'],rate,0,r['case_id'],r['repeat'],arm if delay else 'A1'];ds.append(r['latency_ms']-comparator['latency_ms'])
  pairs.append({'representation':rep,'rate_percent':rate,'delay_ms':delay,'arm':arm,'comparator':which,'pairs':len(ds),'median_added_ms':float(np.median(ds)),'p95_added_ms':float(np.percentile(ds,95))})
 output={'timed_requests':len(rows),'groups':result,'paired_cost':pairs,'cases':160,'security_units_per_condition':480,'timings_per_condition':960,'frozen_runner_hash_matches':hashlib.sha256((H/'run.py').read_bytes()).hexdigest()==json.loads((D/'design.json').read_text())['runner_sha256']}
 (D/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
 for g in result:
  if g['rate_percent']==10 and g['delay_ms']==0 or g['arm']=='R':print(g['representation'],g['rate_percent'],g['delay_ms'],g['arm'],'exp',g['exposed_units'],'rec',round(g['mean_recall'],4),'p50',round(g['p50_ms'],3),'p95',round(g['p95_ms'],3),'calls',round(g['mean_policy_calls'],3))
 print('Total timed requests',len(rows),'groups',len(result))
if __name__=='__main__':main()
