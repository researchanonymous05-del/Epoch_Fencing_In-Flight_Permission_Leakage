"""Predictive placement model: budget recall, refill rounds, race exposure and latency, validated on measured runs."""
from pathlib import Path
from collections import defaultdict
import json,random
import numpy as np
from scipy.stats import hypergeom
H=Path(__file__).resolve().parent;ROOT=H.parents[1];OUT=H/'data_model';OUT.mkdir(exist_ok=True)
V4=ROOT/'experiments/tenant_scoped_retrieval_v4/data/requests.jsonl';V5=H/'data/requests.jsonl';FENCE=H/'data_fence'
SOURCE=ROOT/'data/benchmark.json'
REFILL=[5,10,20,40,80]
def rows(path):return [json.loads(s) for s in path.read_text().splitlines()]
def ownership():
 records=json.loads(SOURCE.read_text())['records'];accounts=sorted({r['account_id'] for r in records});out={}
 for seed in [11,29,47]:
  a=accounts.copy();random.Random(seed).shuffle(a);owner={x:f't{i%20:02}' for i,x in enumerate(a)};out[seed]=[owner[r['account_id']] for r in records]
 return out
def tenant_counts(tenant,removed,t):
 owned=[i for i,x in enumerate(tenant) if x==t];return len(owned),sum(i in removed for i in owned)
def budget_recall(M,K,B,k=5):
 """Expected recall@k of a B-candidate gate when K of the tenant's M documents are revoked independently of rank."""
 x=np.arange(0,min(K,B)+1);return float(np.sum(hypergeom.pmf(x,M,K,B)*np.minimum(k,B-x))/k)
def refill_rounds(M,K,k=5):
 """Distribution of refill rounds over budgets 5/10/20/40/80: stop at the first budget with at least k eligible."""
 done=[float(hypergeom.cdf(B-k,M,K,min(B,M))) if B<=M else 1.0 for B in REFILL];probs=[];prev=0.0
 for p in done:probs.append(max(0.0,p-prev));prev=max(prev,p)
 probs[-1]+=1-sum(probs);return probs
def recall_model(own):
 """M1: zero-parameter hypergeometric predictions vs the v4 budget/rate sweep."""
 groups=defaultdict(lambda:{'measured':[],'predicted':[]});rounds=defaultdict(lambda:{'measured':[],'predicted':np.zeros(len(REFILL))})
 for r in rows(V4):
  if r['suite']!='sweep' or r['repeat']!=0 or not (r['arm'].startswith('G') or r['arm']=='R'):continue
  removed=set(random.Random(r['seed']+9000).sample(range(1430),r['revoked_count']));M,K=tenant_counts(own[r['seed']],removed,r['tenant'])
  if r['arm']=='R':
   key=(r['representation'],r['rate_percent']);rounds[key]['measured'].append(len(r['rounds']));rounds[key]['predicted']+=refill_rounds(M,K)
  else:
   key=(r['representation'],r['rate_percent'],int(r['arm'][1:]));groups[key]['measured'].append(r['recall']);groups[key]['predicted'].append(budget_recall(M,K,int(r['arm'][1:])))
 recall=[{'representation':k[0],'rate_percent':k[1],'budget':k[2],'units':len(v['measured']),'measured':float(np.mean(v['measured'])),'predicted':float(np.mean(v['predicted']))} for k,v in sorted(groups.items())]
 refill=[]
 for k,v in sorted(rounds.items()):
  n=len(v['measured']);hist=[sum(x==j+1 for x in v['measured']) for j in range(len(REFILL))]
  refill.append({'representation':k[0],'rate_percent':k[1],'units':n,'measured_rounds':hist,'predicted_rounds':[float(x) for x in v['predicted']],'measured_mean':float(np.mean(v['measured'])),'predicted_mean':float(np.dot(v['predicted'],range(1,len(REFILL)+1))/n)})
 return recall,refill
def smallest_budget(M,K,target):
 for B in range(5,M+1):
  if budget_recall(M,K,B)>=target:return B
 return M
def race_window(r):
 """Interval in which an authority commit would expose the victim: after the last snapshot that could exclude it, before delivery."""
 if r['arm']=='A1':return r['t0_ns'],r['end_ns']
 calls=r['policy_calls'];action={'P':'list','G20':'check','F':'fence'}[r['arm']]
 return [c for c in calls if c['action']==action][-1]['snapshot_ns'],r['end_ns']
def race_model():
 """M2: predict in-flight exposure from each trial's measured window and the horizon, without using its commit time."""
 data=rows(FENCE/'race.jsonl');groups=defaultdict(list);consistent=0;checked=0
 # The revoker's sleep overshoot (commit - target) is harness timing. It depends on what the client thread is doing at
 # the target instant, so absolute predictions use one median shift per representation/wait and are a secondary check;
 # the primary rate check is the ratio of exposures between arms, where the shared revoker's timing error cancels.
 lag=defaultdict(list)
 for r in data:lag[r['representation'],r['wait_ms']].append(r['commit_ns']-r['target_ns'])
 shift={k:float(np.median(v)) for k,v in lag.items()}
 for r in data:
  lo,hi=race_window(r);s=shift[r['representation'],r['wait_ms']];T=float(r['horizon_ns'])
  # Commit c = t0 + U + s with U~Uniform[0,T]; exposure needs lo < c < hi.
  a=max(0.0,lo-r['t0_ns']-s);b=min(T,hi-r['t0_ns']-s);p=max(0.0,b-a)/T
  inside=lo<r['commit_ns']<hi;expected=bool(inside and r['victim'] in r['attempts'][-1]['returned'])
  if r['arm']!='A1':checked+=1;consistent+=expected==r['race_exposure']
  groups[r['representation'],r['wait_ms'],r['arm']].append((p,r['race_exposure'],(hi-lo)/1e6,r['latency_ms'],len(r['attempts'])))
 out=[]
 for k,v in sorted(groups.items()):
  p=np.array([x[0] for x in v]);m=sum(x[1] for x in v);sd=float(np.sqrt(np.sum(p*(1-p))))
  out.append({'representation':k[0],'wait_ms':k[1],'arm':k[2],'trials':len(v),'measured_exposed':int(m),'predicted_exposed':float(p.sum()),'predicted_sd':sd,'z':float((m-p.sum())/sd) if sd else None,'mean_window_ms':float(np.mean([x[2] for x in v])),'p50_latency_ms':float(np.median([x[3] for x in v])),'retried':sum(x[4]>1 for x in v)})
 ratios=[]
 for rep in ['lexical','neural']:
  for wait in [0,5]:
   g={x['arm']:x for x in out if x['representation']==rep and x['wait_ms']==wait}
   for num,den in [('P','F'),('P','G20'),('F','G20')]:
    ratios.append({'representation':rep,'wait_ms':wait,'pair':f'{num}/{den}','exposure_ratio':g[num]['measured_exposed']/max(1,g[den]['measured_exposed']),'window_ratio':g[num]['mean_window_ms']/g[den]['mean_window_ms']})
 return out,{'trials_checked':checked,'trace_consistent':consistent,'commit_lag_median_us':{'/'.join(map(str,k)):v/1e3 for k,v in shift.items()},'ratios':ratios}
def features(r,source):
 """Work counts per request: candidates requested, ANN calls, policy calls, policy calls x wait, filter IDs."""
 if source=='fence':
  cand=sum(len(a['candidates']) if r['arm']!='G20' else 20 for a in r['attempts']);ann=len(r['attempts']);lst=sum(len(c.get('revoked',[])) if c['action']=='list' else 0 for c in r['policy_calls']);wait=r['wait_ms']
 else:
  cand=sum(s['budget'] for s in r['rounds']);ann=len(r['rounds']);wait=r.get('delay_ms',0);lst=r.get('list_count') or 0
  if source=='v4' and r['arm']=='P':lst=len([c for c in r['policy_calls'] if c['action']=='list'][0]['revoked'])
 calls=len(r['policy_calls']);keyed=int(r['arm'] in ['P','Pbody','F'])
 return np.array([cand,ann,calls,calls*wait,lst,keyed],dtype=float)
NAMES=['per_candidate_ms','per_ann_call_ms','per_policy_call_ms','per_waited_call_x_wait','per_filter_id_ms','per_scoped_key_ms']
def cost_model():
 """M3: differential work model. Each arm's latency is paired with G20 on the same run, seed, condition, case and
 repetition, which cancels run-level load drift; coefficients are fitted to condition medians of those differences."""
 sources=[('v4',r) for r in rows(V4) if r['suite'] in ['sweep','delay'] and r['arm'] not in ['A2']]
 sources+=[('v5',r) for r in rows(V5) if r['success'] and r['arm'] in ['P','G20','R']]
 if (FENCE/'cost.jsonl').exists():sources+=[('fence',r) for r in rows(FENCE/'cost.jsonl')]
 def cond(src,r):return (src,r['representation'],r['seed'],r.get('suite'),r.get('rate_percent'),r.get('delay_ms',r.get('wait_ms',0)),r.get('requested_list_size'))
 ref={(cond(s,r),r['case_id'],r['repeat']):(features(r,s),r['latency_ms']) for s,r in sources if r['arm']=='G20'}
 cells=defaultdict(list)
 for s,r in sources:
  if r['arm']=='G20':continue
  fr,yr=ref[cond(s,r),r['case_id'],r['repeat']];cells[cond(s,r)+(r['arm'],)].append((features(r,s)-fr,r['latency_ms']-yr))
 table=[(k,np.median([f for f,_ in v],axis=0),float(np.median([y for _,y in v]))) for k,v in cells.items()]
 def fit(train):
  X=np.array([x for _,x,_ in train]);y=np.array([y for _,_,y in train]);coef,*_=np.linalg.lstsq(X,y,rcond=None);return coef
 def row(k,x,y,coef):return {'source':k[0],'seed':k[2],'suite':k[3],'rate':k[4],'wait':k[5],'list':k[6],'arm':k[7],'measured_ms':y,'predicted_ms':float(x@coef)}
 def err(v):return {'median_abs_ms':float(np.median([abs(e['predicted_ms']-e['measured_ms']) for e in v])),'max_abs_ms':float(max(abs(e['predicted_ms']-e['measured_ms']) for e in v)),'sign_agreement':float(np.mean([np.sign(round(e['predicted_ms'],1))==np.sign(round(e['measured_ms'],1)) for e in v])),'rows':len(v)} if v else None
 result={}
 for rep in ['lexical','neural']:
  mine=[t for t in table if t[0][1]==rep];loso=[]
  for held in [11,29,47]:
   coef=fit([t for t in mine if t[0][2]!=held]);loso+=[row(k,x,y,coef) for k,x,y in mine if k[2]==held]
  coef_v4=fit([t for t in mine if t[0][0]=='v4']);cross=[row(k,x,y,coef_v4) for k,x,y in mine if k[0]!='v4']
  # Held-out arm: fit without any fence-run rows, then predict the never-fitted F arm (and P/G20-relative cost there).
  coef_nf=fit([t for t in mine if t[0][0]!='fence']);unseen=[row(k,x,y,coef_nf) for k,x,y in mine if k[0]=='fence']
  full=fit(mine)
  result[rep]={'coefficients':dict(zip(NAMES,map(float,full))),'conditions':len(mine),'loso':err(loso),'cross_run_from_v4':err(cross),'held_out_fence_run':err(unseen),'held_out_fence_rows':unseen,'cross_run_rows':cross,'loso_rows':loso}
 return result
def decision(costs):
 """Predicted cheapest placement meeting recall>=0.99, relative to G20, over filter-list size and revocation rate."""
 grid=[];M=72
 for rep,c in costs.items():
  w=c['coefficients']
  def delta(cand,ann,calls,lst,keyed):return w['per_candidate_ms']*(cand-20)+w['per_ann_call_ms']*(ann-1)+w['per_policy_call_ms']*(calls-1)+w['per_filter_id_ms']*lst+w['per_scoped_key_ms']*keyed
  for rate in [5,10,25]:
   K=round(M*rate/100);B=smallest_budget(M,K,0.99)
   for L in [10,100,1000,10000]:
    g=delta(B,1,1,0,0);p=delta(5,1,1,L,1);f=delta(5,1,2,L,1)
    grid.append({'representation':rep,'rate_percent':rate,'list_ids':L,'gate_budget_for_0.99':B,'G_vs_G20_ms':g,'P_vs_G20_ms':p,'F_vs_G20_ms':f,'stable_contract_choice':'P' if p<g else f'G{B}','request_end_choice':'F' if f<g else f'G{B}'})
  # Break-even filter size: list IDs at which P (or F) costs the same as the cheapest sufficient gate.
  for rate in [5,10,25]:
   K=round(M*rate/100);B=smallest_budget(M,K,0.99);g=delta(B,1,1,0,0)
   base_p=delta(5,1,1,0,1);base_f=delta(5,1,2,0,1)
   grid.append({'representation':rep,'rate_percent':rate,'breakeven':True,'gate_budget_for_0.99':B,'P_breakeven_ids':(g-base_p)/w['per_filter_id_ms'],'F_breakeven_ids':(g-base_f)/w['per_filter_id_ms']})
 return grid
def main():
 own=ownership();recall,refill=recall_model(own);result={'recall_model':recall,'refill_model':refill}
 if (FENCE/'race.jsonl').exists():result['race_model'],result['race_trace_check']=race_model()
 result['cost_model']=cost_model();result['decision_grid']=decision(result['cost_model'])
 slim={k:v for k,v in result.items() if k!='cost_model'};slim['cost_model']={rep:{k:v for k,v in c.items() if not k.endswith('_rows')} for rep,c in result['cost_model'].items()}
 (OUT/'model.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(slim,indent=1))
if __name__=='__main__':main()
