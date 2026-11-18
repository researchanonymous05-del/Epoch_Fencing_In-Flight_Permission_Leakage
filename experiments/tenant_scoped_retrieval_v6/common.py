"""Shared harness: separate-process authority and revoker, isolated Typesense, placements A1/P/G20/F/Fg, trial loop."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,time,random,secrets,subprocess,hashlib,importlib.util,http.client,socket
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parents[1];D=H/'data';D.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('admission_v4',ROOT/'experiments/tenant_scoped_retrieval_v4/run.py');v4=importlib.util.module_from_spec(spec);spec.loader.exec_module(v4);base=v4.base;base.CONFIG['port']=8118
SOURCE=ROOT/'data/benchmark.json'
CFG={'seeds':[11,29,47],'representations':['lexical','neural'],'revoked_count':143,'arms':['A1','P','G20','F'],'churn_arms':['P','G20','F','Fg'],'waits_ms':[0,5],'horizon_ms':{'0':20,'5':40},'offsets_per_case':2,'lead_ms':1.0,'churn_rate_per_s':200,'search_port':8118,'authority_port':8128,'revoker_port':8138,'image':'typesense/typesense:30.2',
 'authority':'separate OS process; serialized ledger; global and per-tenant epochs; policy wait split half before and half after the snapshot',
 'revoker':'separate OS process; spins to an absolute CLOCK_UPTIME_RAW target, then commits at the authority',
 'phantom':'stable trials commit a non-indexed ID owned by tenant t99, never queried: identical schedule/load to race trials, no effect on any queried tenant',
 'fence':'F: P then a fence over returned IDs with per-tenant epoch fast path; Fg: same with global epoch; if a fence drops an ID, rerun P once and fence again; if the second fence drops, return the remaining IDs',
 'victim':"query's top eligible document under the base 10% revocation (usually the query's own support record)"}
PHANTOM=1_000_000+99*10000
def now():return time.clock_gettime_ns(time.CLOCK_UPTIME_RAW)
def digest(b):return hashlib.sha256(b).hexdigest()
class Policy:
 # Bytes body (one write with the headers) and TCP_NODELAY: a str body is written separately and, without NODELAY,
 # Nagle plus delayed ACK stalled larger requests by ~125 ms in the first (discarded) stage-1 run.
 def __init__(self,port):
  self.c=http.client.HTTPConnection('127.0.0.1',port,timeout=10);self.c.connect();self.c.sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
 def __call__(self,**q):
  self.c.request('POST','/policy',body=json.dumps(q).encode(),headers={'Content-Type':'application/json'});r=self.c.getresponse();obj=json.loads(r.read());assert r.status==200;return obj
class Env:
 """Starts Typesense, authority and revoker; loads one representation/seed at a time."""
 def __init__(self):
  self.key=secrets.token_hex(24);self.name='v6-'+secrets.token_hex(4);self.procs=[]
  subprocess.run(['docker','run','-d','--rm','--name',self.name,'--cpus','2','--memory','2g','-p',f"127.0.0.1:{CFG['search_port']}:8108",'--tmpfs','/data:rw,size=512m','-e',f'TYPESENSE_API_KEY={self.key}',CFG['image'],'--data-dir=/data','--api-port=8108'],check=True,capture_output=True)
  for script,args in [('authority.py',[str(CFG['authority_port'])]),('revoker.py',[str(CFG['revoker_port']),str(CFG['authority_port'])])]:
   p=subprocess.Popen([sys.executable,'-I',str(H/script),*args],stdout=subprocess.PIPE,text=True);assert p.stdout.readline().strip()=='ready';self.procs.append(p)
  self.ts=base.TS(self.key)
  for _ in range(80):
   try:
    if self.ts.call('GET','/health')['ok']:break
   except Exception:self.ts.conn.close();self.ts=base.TS(self.key);time.sleep(.2)
  else:raise RuntimeError('search startup failed')
  self.auth=Policy(CFG['authority_port']);self.rev=Policy(CFG['revoker_port']);self.ctl=Policy(CFG['authority_port'])
  src=json.loads(SOURCE.read_text());self.records=src['records'];self.accounts=sorted({r['account_id'] for r in self.records});self.support={r['account_id']:i for i,r in enumerate(self.records) if r['kind']=='support'};self.queries=[q for q in src['cases'] if q['split']=='test']
  self.sets={'lexical':np.load(ROOT/'experiments/tenant_scoped_retrieval/data/vectors.npz')['documents'],'neural':np.load(ROOT/'experiments/tenant_scoped_retrieval_v3/data/neural_vectors.npy')}
 def load(self,rep,seed):
  vectors=self.sets[rep];a=self.accounts.copy();random.Random(seed).shuffle(a);owner={x:f't{i%20:02}' for i,x in enumerate(a)};tenant=[owner[r['account_id']] for r in self.records];col=f'v6{rep[0]}{seed}'
  self.ts.call('POST','/collections',{'name':col,'fields':[{'name':'tenant','type':'string','facet':True},{'name':'active','type':'bool','facet':True},{'name':'rid','type':'int32','facet':True},{'name':'vec','type':'float[]','num_dim':vectors.shape[1],'hnsw_params':{'M':16,'ef_construction':200}}]})
  raw=self.ts.call('POST',f'/collections/{col}/documents/import?action=create','\n'.join(json.dumps({'id':str(i),'rid':i,'tenant':tenant[i],'active':True,'vec':vectors[i].tolist()}) for i in range(1430)),raw=True);assert all(json.loads(s)['success'] for s in raw.splitlines())
  parent=self.ts.createkey([col]);removed=sorted(random.Random(seed+9000).sample(range(1430),CFG['revoked_count']));self.ctl(action='setup',tenants=tenant,revoked=removed)
  qs=[{'case_id':q['case_id'],'tenant':owner[q['account_id']],'source_index':self.support[q['account_id']]} for q in self.queries];scores={q['case_id']:vectors@vectors[q['source_index']] for q in qs};rm=set(removed);tn=np.array(tenant)
  victims={q['case_id']:int(max(np.flatnonzero((tn==q['tenant'])&np.array([i not in rm for i in range(1430)])),key=lambda i:scores[q['case_id']][i])) for q in qs}
  keys={t:base.scoped(parent,{'filter_by':f'tenant:={t} && active:=true','exclude_fields':'vec'}) for t in sorted(set(tenant))}
  return {'collection':col,'parent':parent,'keys':keys,'vectors':vectors,'tenant':tn,'removed':rm,'qs':qs,'scores':scores,'victims':victims}
 def drop(self,ctx):self.ts.call('DELETE',f"/collections/{ctx['collection']}")
 def close(self):
  for p in self.procs:p.terminate()
  subprocess.run(['docker','stop',self.name],capture_output=True)
 def run(self,ctx,arm,q,wait):
  """One request; returns trace with decisive snapshot and delivery time."""
  t=q['tenant'];calls=[];attempts=[];start=now()
  def call(**kw):
   b=now();obj=self.auth(tenant=t,wait_ms=wait,**kw);calls.append({**kw,**obj,'http_ms':(now()-b)/1e6});return obj
  def search(key,k):
   p={'collection':ctx['collection'],'q':'*','vector_query':'vec:(['+','.join(format(float(x),'.8g') for x in ctx['vectors'][q['source_index']])+f'],k:{k},ef:100,flat_search_cutoff:0)','per_page':k,'exclude_fields':'vec'}
   res=self.ts.call('POST','/multi_search',{'searches':[p]},key=key)['results'][0];assert 'hits' in res,res;return [int(h['document']['id']) for h in res['hits']]
  if arm in ['A1','G20']:
   c=search(ctx['keys'][t],20 if arm=='G20' else 5)
   if arm=='G20':allowed=set(call(action='check',ids=c)['allowed']);ret=[i for i in c if i in allowed][:5]
   else:ret=c[:5]
   attempts.append({'candidates':c,'returned':ret})
  else:
   for n in range(2 if arm in ['F','Fg'] else 1):
    st=call(action='list');clause=f'tenant:={t} && active:=true'
    if st['revoked']:clause+=' && rid:!=['+','.join(map(str,st['revoked']))+']'
    c=search(base.scoped(ctx['parent'],{'filter_by':clause,'exclude_fields':'vec'}),5);ret=c[:5]
    if arm in ['F','Fg']:
     scope='tenant' if arm=='F' else 'global';fe=call(action='fence',ids=ret,scope=scope,epoch=st['tenant_epoch'] if scope=='tenant' else st['epoch']);kept=set(fe['allowed']);dropped=[i for i in ret if i not in kept];ret=[i for i in ret if i in kept]
     attempts.append({'candidates':c,'dropped':dropped,'returned':ret})
     if not dropped:break
    else:attempts.append({'candidates':c,'returned':ret})
  end=now();decisive=None
  if arm!='A1':decisive=[x for x in calls if x['action']=={'P':'list','G20':'check','F':'fence','Fg':'fence'}[arm]][-1]['snapshot_ns']
  return {'returned':ret,'attempts':attempts,'policy_calls':calls,'start_ns':start,'end_ns':end,'latency_ms':(end-start)/1e6,'decisive_ns':decisive}
 def run_retry(self,ctx,arm,q,wait):
  """A search-engine 408 is a transient failure, not an outcome: reconnect and repeat the request (at most 3 times)."""
  for n in range(4):
   try:return {**self.run(ctx,arm,q,wait),'retries_408':n}
   except base.APIError as e:
    if e.status!=408 or n==3:raise
    self.ts.conn.close();self.ts=base.TS(self.key)
 def trial(self,ctx,arm,q,wait,offset_ns,ids):
  """Schedule a commit of `ids` at start+offset, run one request, wait for the commit, then restore.
  On a search 408 the whole trial (schedule, request, commit, restore) is repeated, at most 3 times."""
  for n in range(4):
   t0=now()+int(CFG['lead_ms']*1e6);self.rev(action='schedule',target_ns=t0+offset_ns,ids=ids)
   while now()<t0:pass
   try:r=self.run(ctx,arm,q,wait);failed=None
   except base.APIError as e:
    if e.status!=408 or n==3:raise
    failed=e
   c=self.rev(action='result');self.ctl(action='restore',ids=ids)
   if failed:self.ts.conn.close();self.ts=base.TS(self.key);continue
   return {**r,'t0_ns':t0,'target_ns':c['target_ns'],'fired_ns':c['fired_ns'],'commit_ns':c['commit_ns'],'retries_408':n}
def resume(path,key,size):
 """Keep only complete groups from an interrupted partial file; return the kept rows' group keys."""
 if not path.exists():return set()
 rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()];count={}
 for r in rows:count[key(r)]=count.get(key(r),0)+1
 done={k for k,n in count.items() if n==size};path.write_text(''.join(json.dumps(r)+'\n' for r in rows if key(r) in done));return done
def score(ctx,q,returned,extra_revoked=()):
 mask=(ctx['tenant']==q['tenant'])&np.array([i not in ctx['removed'] and i not in extra_revoked for i in range(1430)]);strict,ties,k=base.reference(ctx['scores'][q['case_id']],mask,5);ret=set(returned)
 return (len(ret&strict)+min(len(ret&ties),k-len(strict)))/k,bool(any(not mask[i] for i in ret))
def plan(seed,wait,salt):
 """Deterministic trial order and offsets shared by the stable and race stages."""
 horizon=CFG['horizon_ms'][str(wait)]*1_000_000;order=[(o,j,arm) for o in range(CFG['offsets_per_case']) for j in range(160) for arm in CFG['arms']];random.Random(seed*1000+wait+salt).shuffle(order);rng=random.Random(seed*100+wait+salt)
 return order,{key:rng.randrange(horizon) for key in sorted(order)},horizon
