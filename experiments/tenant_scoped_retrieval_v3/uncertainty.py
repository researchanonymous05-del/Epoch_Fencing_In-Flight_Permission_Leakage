"""Post-hoc case-block bootstrap of paired gate costs; no new measurements."""
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
D=Path(__file__).resolve().parent/'data';rows=[json.loads(x) for x in (D/'requests.jsonl').read_text().splitlines()];cases=sorted({r['case_id'] for r in rows});index={(r['representation'],r['seed'],r['regime'],r['phase'],r['case_id'],r['repeat'],r['arm']):r for r in rows};results=[]
for rep in ['lexical','neural']:
 for arm in ['G5','G20']:
  blocks=[]
  for case in cases:
   blocks.append([index[rep,s,'random10','stale',case,r,arm]['latency_ms']-index[rep,s,'random10','stale',case,r,'A1']['latency_ms'] for s in [11,29,47] for r in [0,1]])
  blocks=np.array(blocks);rng=np.random.default_rng(20261101);boots=np.median(blocks[rng.integers(0,len(cases),size=(2000,len(cases)))].reshape(2000,-1),axis=1)
  results.append({'representation':rep,'arm':arm,'median_added_ms':float(np.median(blocks)),'case_block_percentile_95_low_ms':float(np.percentile(boots,2.5)),'case_block_percentile_95_high_ms':float(np.percentile(boots,97.5))})
report={'method':'Post-hoc sensitivity: 2000 case-block resamples; 160 cases, preserving all ownership assignments and repeats within each case. Conditional on this corpus, three assignments and one timing run; not independent deployment or security population bounds.','seed':20261101,'paired_cost_intervals':results}
(D/'uncertainty.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
for rep in ['lexical','neural']:
 for reg in ['random10','targeted']:
  for arm in ['G5','G20']:
   rs=[r for r in rows if (r['representation'],r['regime'],r['phase'],r['arm'],r['repeat'])==(rep,reg,'stale',arm,0)];print(rep,reg,arm,'short',sum(len(r['returned'])<5 for r in rs),'mean_return',float(np.mean([len(r['returned']) for r in rs])))
