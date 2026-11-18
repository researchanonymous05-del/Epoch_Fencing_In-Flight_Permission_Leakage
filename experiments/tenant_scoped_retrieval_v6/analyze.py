"""Compare frozen stable-run predictions with race outcomes for every cell; summarize stable cost and churn fast paths."""
from pathlib import Path
from collections import defaultdict
import json,hashlib
import numpy as np
H=Path(__file__).resolve().parent;D=H/'data'
def rows(name):return [json.loads(s) for s in (D/name).read_text().splitlines()]
def main():
 pred=json.loads((D/'predictions.json').read_text());assert hashlib.sha256((D/'stable.jsonl').read_bytes()).hexdigest()==pred['stable_sha256']
 runs=json.loads((D/'race_design.json').read_text())['runs'];assert all(r['predictions_sha256']==hashlib.sha256((D/'predictions.json').read_bytes()).hexdigest() for r in runs)
 race=defaultdict(list);stable=defaultdict(list)
 for r in rows('race.jsonl'):race[r['representation'],r['wait_ms'],r['arm']].append(r)
 for r in rows('stable.jsonl'):stable[r['representation'],r['wait_ms'],r['arm']].append(r)
 cells=[]
 for c in pred['cells']:
  k=(c['representation'],c['wait_ms'],c['arm']);rs=race[k];m=sum(r['race_exposure'] for r in rs);p=c['predicted_count_960']
  ref={(r['seed'],r['case_id'],r['offset_index']):r['latency_ms'] for r in stable[c['representation'],c['wait_ms'],'G20']}
  cells.append({**c,'race_trials':len(rs),'measured':m,'z':(m-p)/c['predicted_sd'] if c['predicted_sd'] else None,'stale':sum(r['stale_exposure'] for r in rs),'race_recall':float(np.mean([r['recall'] for r in rs])),'retried':sum(len(r['attempts'])>1 for r in rs),'race_p50_ms':float(np.median([r['latency_ms'] for r in rs])),
   'stable_p50_ms':float(np.median([r['latency_ms'] for r in stable[k]])),'stable_recall':float(np.mean([r['recall'] for r in stable[k]])),'stable_vs_G20_ms':float(np.median([r['latency_ms']-ref[r['seed'],r['case_id'],r['offset_index']] for r in stable[k]])),'retries_408':sum(r.get('retries_408',0) for r in rs)+sum(r.get('retries_408',0) for r in stable[k])})
 ratios=[]
 for rep in ['lexical','neural']:
  for wait in [0,5]:
   g={c['arm']:c for c in cells if (c['representation'],c['wait_ms'])==(rep,wait)}
   for a,b in [('P','F'),('P','G20'),('F','G20')]:ratios.append({'representation':rep,'wait_ms':wait,'pair':f'{a}/{b}','predicted':g[a]['predicted_count_960']/max(g[b]['predicted_count_960'],1e-9),'measured':g[a]['measured']/max(g[b]['measured'],1)})
 ch=defaultdict(list)
 for r in rows('churn.jsonl'):ch[r['representation'],r['arm']].append(r)
 ref={(r['representation'],r['seed'],r['case_id'],r['repeat']):r['latency_ms'] for r in rows('churn.jsonl') if r['arm']=='G20'}
 churn=[{'representation':k[0],'arm':k[1],'timings':len(v),'p50_ms':float(np.median([r['latency_ms'] for r in v])),'vs_G20_ms':float(np.median([r['latency_ms']-ref[k[0],r['seed'],r['case_id'],r['repeat']] for r in v])),'fast_fraction':float(np.mean([all(r['fast']) for r in v])) if k[1] in ['F','Fg'] else None,'retry_fraction':float(np.mean([len(r['attempts'])>1 for r in v])),'recall':float(np.mean([r['recall'] for r in v])),'stale':sum(r['stale_exposure'] for r in v)} for k,v in sorted(ch.items())]
 lag=[(r['commit_ns']-r['target_ns'])/1e3 for v in race.values() for r in v]
 out={'cells':cells,'ratios':ratios,'churn':churn,'race_trials':sum(len(v) for v in race.values()),'stable_trials':sum(len(v) for v in stable.values()),'commit_lag_us_p50':float(np.median(lag)),'commit_lag_us_p95':float(np.percentile(lag,95))}
 (D/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
 for c in cells:print(c['representation'][:3],c['wait_ms'],c['arm'],'pred',round(c['predicted_count_960'],1),'+-',round(c['predicted_sd'],1),'meas',c['measured'],'z',None if c['z'] is None else round(c['z'],2),'win',round(c['mean_window_ms'],2),'recall',round(c['race_recall'],3),'retry',c['retried'],'stable dG20',round(c['stable_vs_G20_ms'],2),'408s',c['retries_408'])
 for r in ratios:print(r)
 for c in churn:print('churn',c)
 print('lag us',out['commit_lag_us_p50'],out['commit_lag_us_p95'])
if __name__=='__main__':main()
