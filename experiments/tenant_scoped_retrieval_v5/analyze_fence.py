"""Summarize the fence/race study and the prospective budget run; recompute paired costs against G20."""
from pathlib import Path
from collections import defaultdict
import json,hashlib
import numpy as np
H=Path(__file__).resolve().parent;D=H/'data_fence';P=H/'data_prospective'
def rows(path):return [json.loads(s) for s in path.read_text().splitlines()]
def main():
 race=rows(D/'race.jsonl');cost=rows(D/'cost.jsonl');out={'race_trials':len(race),'cost_timings':len(cost)}
 g=defaultdict(list)
 for r in race:g[r['representation'],r['wait_ms'],r['arm']].append(r)
 out['race']=[{'representation':k[0],'wait_ms':k[1],'arm':k[2],'trials':len(v),'revoked_before_end':sum(r['revoked_before_end'] for r in v),'race_exposed':sum(r['race_exposure'] for r in v),'stale_exposed':sum(r['stale_exposure'] for r in v),'retried':sum(len(r['attempts'])>1 for r in v),'mean_recall':float(np.mean([r['recall'] for r in v])),'short_results':sum(len(r['returned'])<5 for r in v),'p50_ms':float(np.median([r['latency_ms'] for r in v])),'commit_lag_p50_us':float(np.median([(r['commit_ns']-r['target_ns'])/1e3 for r in v]))} for k,v in sorted(g.items())]
 g=defaultdict(list);ref={(r['representation'],r['seed'],r['wait_ms'],r['case_id'],r['repeat']):r['latency_ms'] for r in cost if r['arm']=='G20'}
 for r in cost:g[r['representation'],r['wait_ms'],r['arm']].append(r)
 out['cost']=[{'representation':k[0],'wait_ms':k[1],'arm':k[2],'units':sum(r['repeat']==0 for r in v),'timings':len(v),'exposed_units':sum(r['exposure'] for r in v if r['repeat']==0),'mean_recall':float(np.mean([r['recall'] for r in v if r['repeat']==0])),'p50_ms':float(np.median([r['latency_ms'] for r in v])),'p95_ms':float(np.percentile([r['latency_ms'] for r in v],95)),'mean_policy_calls':float(np.mean([len(r['policy_calls']) for r in v])),'fence_fast_fraction':float(np.mean([all(c.get('fast',False) for c in r['policy_calls'] if c['action']=='fence') for r in v])) if k[2]=='F' else None,'median_vs_G20_ms':float(np.median([r['latency_ms']-ref[r['representation'],r['seed'],r['wait_ms'],r['case_id'],r['repeat']] for r in v]))} for k,v in sorted(g.items())]
 design=json.loads((D/'design.json').read_text());out['frozen_runner_hash_matches']=hashlib.sha256((H/'run_fence.py').read_bytes()).hexdigest()==design['runner_sha256']
 if (P/'requests.jsonl').exists():
  pr=rows(P/'requests.jsonl');pref={(r['representation'],r['seed'],r['rate_percent'],r['case_id'],r['repeat']):r['latency_ms'] for r in pr if r['arm']=='G20'};g=defaultdict(list)
  for r in pr:g[r['representation'],r['rate_percent'],r['arm']].append(r)
  pdesign=json.loads((P/'design.json').read_text())
  out['prospective']={'frozen_runner_hash_matches':hashlib.sha256((H/'run_prospective.py').read_bytes()).hexdigest()==pdesign['runner_sha256'],'frozen_predictions_hash_matches':hashlib.sha256((H/'data_model/prospective_predictions.json').read_bytes()).hexdigest()==pdesign['predictions_sha256'],'groups':[{'representation':k[0],'rate_percent':k[1],'arm':k[2],'units':sum(r['repeat']==0 for r in v),'exposed_units':sum(r['exposure'] for r in v if r['repeat']==0),'mean_recall':float(np.mean([r['recall'] for r in v if r['repeat']==0])),'full_recall_units':sum(r['recall']==1.0 for r in v if r['repeat']==0),'false_denials':sum(r['false_denial'] for r in v if r['repeat']==0),'p50_ms':float(np.median([r['latency_ms'] for r in v])),'median_vs_G20_ms':float(np.median([r['latency_ms']-pref[r['representation'],r['seed'],r['rate_percent'],r['case_id'],r['repeat']] for r in v]))} for k,v in sorted(g.items())]}
 (D/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
 for x in out['race']:print('race',x)
 for x in out['cost']:print('cost',x)
 for x in out.get('prospective',{}).get('groups',[]):print('prospective',x)
 print('hash',out['frozen_runner_hash_matches'],out.get('prospective',{}).get('frozen_runner_hash_matches'),out.get('prospective',{}).get('frozen_predictions_hash_matches'))
if __name__=='__main__':main()
