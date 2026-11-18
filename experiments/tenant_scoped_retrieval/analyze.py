"""Analyze saved requests, keeping repeated measurements distinct from case-seed units."""
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
HERE=Path(__file__).resolve().parent;D=HERE/'data'
def main():
 rows=[json.loads(s) for s in (D/'requests.jsonl').read_text().splitlines()];probes=[json.loads(s) for s in (D/'probes.jsonl').read_text().splitlines()]
 groups=defaultdict(list)
 for row in rows:groups[(row['arm'],row['phase'])].append(row)
 result=[]
 for (arm,phase),rs in sorted(groups.items()):
  units=defaultdict(list)
  for r in rs:units[(r['seed'],r['case_id'])].append(r)
  unit_exposures=sum(any(r['unauthorized_hits'] for r in us) for us in units.values())
  seed_stats=[]
  for seed in [11,29,47]:
   sr=[r for r in rs if r['seed']==seed]
   seed_stats.append({'seed':seed,'exposure_requests':sum(bool(r['unauthorized_hits']) for r in sr),'requests':len(sr),'mean_recall':float(np.mean([r['recall_at_5'] for r in sr])),'p50_ms':float(np.median([r['latency_ms'] for r in sr])),'p95_ms':float(np.percentile([r['latency_ms'] for r in sr],95))})
  result.append({'arm':arm,'phase':phase,'requests':len(rs),'case_seed_units':len(units),'units_exposed_any_repeat':unit_exposures,'cross_tenant_exposed_requests':sum(bool(r['cross_tenant_hits']) for r in rs),'revoked_exposed_requests':sum(bool(r['revoked_hits']) for r in rs),'unauthorized_exposed_requests':sum(bool(r['unauthorized_hits']) for r in rs),'mean_recall_at_5':float(np.mean([r['recall_at_5'] for r in rs])),'false_denials':sum(bool(r['eligible'] and not r['returned']) for r in rs),'no_authorized_evidence':sum(r['false_denial'] for r in rs),'p50_ms':float(np.median([r['latency_ms'] for r in rs])),'p95_ms':float(np.percentile([r['latency_ms'] for r in rs],95)),'seed_stats':seed_stats})
 pgroup=defaultdict(list)
 for p in probes:pgroup[(p['arm'],p['probe'])].append(p)
 ps=[{'arm':arm,'probe':probe,'attempts':len(rs),'unauthorized_hits':sum(r['unauthorized_hits'] for r in rs),'nonempty':sum(r['hits']>0 for r in rs),'authorization_errors':sum(r['http_error'] in [401,403] for r in rs)} for (arm,probe),rs in sorted(pgroup.items())]
 summary={'groups':result,'probes':ps,'total_requests':len(rows),'total_probes':len(probes),'scoring_note':'Raw runner field false_denial records no authorized evidence, including nonempty unauthorized-only responses. Summary false_denials uses eligible>0 and zero returned hits; raw field is reported separately as no_authorized_evidence. Naming correction; no re-execution or filtering.', 'uncertainty':'3 tenant assignments; per-seed ranges reported; repetitions not independent. No population confidence guarantee.'}
 (D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 for r in result:print(r['arm'],r['phase'],'exposure',r['unauthorized_exposed_requests'],'/',r['requests'],'recall',round(r['mean_recall_at_5'],4),'denials',r['false_denials'],'p50/p95',round(r['p50_ms'],3),round(r['p95_ms'],3))
 print('Probes',ps)
if __name__=='__main__':main()
