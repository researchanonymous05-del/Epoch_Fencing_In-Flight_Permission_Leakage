"""Stage 2: freeze race-exposure predictions computed only from stage-1 stable trials (no race data exists yet)."""
from pathlib import Path
from collections import defaultdict
import json,hashlib,datetime
import numpy as np
H=Path(__file__).resolve().parent;D=H/'data'
def main():
 out=D/'predictions.json'
 if out.exists():raise RuntimeError('Predictions already frozen')
 assert not (D/'race.jsonl').exists() and not (D/'race.partial.jsonl').exists(),'race data must not exist when predicting'
 raw=(D/'stable.jsonl').read_bytes();g=defaultdict(list)
 for line in raw.decode().splitlines():r=json.loads(line);g[r['representation'],r['wait_ms'],r['arm']].append(r)
 cells=[]
 for (rep,wait,arm),rs in sorted(g.items()):
  n=len(rs);T=rs[0]['horizon_ns'];p=float(np.mean([r['would_expose'] for r in rs]))
  # Analytic form: window share of the horizon, times the share of requests that return the victim.
  win=[(r['end_ns']-(r['decisive_ns'] if r['decisive_ns'] is not None else r['t0_ns']))/1e6 for r in rs];ret=float(np.mean([r['victim_returned'] for r in rs]))
  cells.append({'representation':rep,'wait_ms':wait,'arm':arm,'stable_trials':n,'counterfactual_rate':p,'predicted_count_960':960*p,'predicted_sd':float(np.sqrt(960*p*(1-p)+960**2*p*(1-p)/n)),'mean_window_ms':float(np.mean(win)),'victim_returned_rate':ret,'analytic_rate':ret*float(np.mean(win))/(T/1e6)})
 obj={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stable_sha256':hashlib.sha256(raw).hexdigest(),'predict_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rule':'race exposure for each cell = 960 x share of stable trials whose phantom commit fell after the decisive snapshot and before delivery while the victim was returned; sd includes binomial noise of both stages','cells':cells}
 out.write_text(json.dumps(obj,indent=2)+'\n')
 for c in cells:print(c['representation'][:3],c['wait_ms'],c['arm'],round(c['predicted_count_960'],1),'+-',round(c['predicted_sd'],1),'win',round(c['mean_window_ms'],2),'ret',round(c['victim_returned_rate'],3),'analytic',round(960*c['analytic_rate'],1))
if __name__=='__main__':main()
