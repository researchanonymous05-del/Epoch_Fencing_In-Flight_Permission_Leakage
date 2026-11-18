"""Case-block resampling of fixed-run paired costs; no additional measurements."""
from pathlib import Path
import json
import numpy as np
D=Path(__file__).resolve().parent/'data';rows=[json.loads(s) for s in (D/'requests.jsonl').read_text().splitlines()];cases=sorted({r['case_id'] for r in rows});lookup={(r['representation'],r['seed'],r['rate_percent'],r['delay_ms'],r['case_id'],r['repeat'],r['arm']):r for r in rows};result=[]
for rep in ['lexical','neural']:
 for delay in [0,5]:
  for arm in ['G20','P','R']:
   blocks=[]
   for c in cases:
    blocks.append([lookup[rep,s,10,delay,c,r,arm]['latency_ms']-lookup[rep,s,10,0,c,r,arm if delay else 'A1']['latency_ms'] for s in [11,29,47] for r in [0,1]])
   blocks=np.array(blocks);rng=np.random.default_rng(20261101);samples=np.median(blocks[rng.integers(0,160,size=(2000,160))].reshape(2000,-1),axis=1);result.append({'representation':rep,'delay_ms':delay,'arm':arm,'median_added_ms':float(np.median(blocks)),'low_ms':float(np.percentile(samples,2.5)),'high_ms':float(np.percentile(samples,97.5))})
report={'method':'2000 case-block percentile resamples retaining all ownership seeds and repetitions within each of 160 cases; conditional sensitivity, not independent deployment confidence','seed':20261101,'results':result};(D/'uncertainty.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
