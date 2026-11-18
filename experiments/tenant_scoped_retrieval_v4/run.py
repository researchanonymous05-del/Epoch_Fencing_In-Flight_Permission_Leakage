"""Measure current-list prefiltering, prefix refill, budget/rate sweeps and policy waits."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,time,random,secrets,subprocess,threading,http.client,socket,hashlib,importlib.util
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parents[1];D=H/'data';D.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('admission_v3',ROOT/'experiments/tenant_scoped_retrieval_v3/run.py');v3=importlib.util.module_from_spec(spec);spec.loader.exec_module(v3);base=v3.base;base.CONFIG['port']=8114
CFG={'seeds':[11,29,47],'representations':['lexical','neural'],'revoked_counts':[72,143,358],'rates_percent':[5,10,25],'budgets':[5,10,20,40],'strategies':['A1','A2','G5','G10','G20','G40','P','R'],'repeats':2,'delay_ms':[0,1,5],'delay_strategies':['G20','P','R'],'refill_budgets':[5,10,20,40,80],'port':8114,'image':'typesense/typesense:30.2','policy_delay':'response wait after authority snapshot; added once per actual HTTP call; not a WAN simulator','prefilter':'tenant-scoped current revoked-ID exclusion signed into per-request key before ANN; no post-search recheck','refill':'successively requery entire prefix 5/10/20/40/80; recheck full prefix each round; stop after five admitted or exhaustion','race':'no revocations during these serial requests; in-flight race excluded and stated in abstract'}
SOURCE=ROOT/'data/benchmark.json'
def dump(name,data):(D/name).write_text(json.dumps(data,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class Handler(BaseHTTPRequestHandler):
 protocol_version='HTTP/1.1'
 def setup(self):super().setup();self.connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
 def log_message(self,*args):pass
 def do_POST(self):
  req=json.loads(self.rfile.read(int(self.headers['Content-Length'])));ledger=self.server.ledger
  with ledger.lock:
   timestamp=time.perf_counter_ns();epoch=ledger.epoch
   if req['action']=='list':body={'revoked':[i for i in sorted(ledger.revoked) if ledger.tenants[i]==req['tenant']]}
   else:body={'allowed':[i for i in req['ids'] if ledger.tenants[i]==req['tenant'] and i not in ledger.revoked]}
  if req['delay_ms']:time.sleep(req['delay_ms']/1000)
  body.update({'epoch':epoch,'snapshot_ns':timestamp});raw=json.dumps(body).encode();self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
class Client:
 def __init__(self,ts,parent,shared,prefix,keys,vectors,policy_port):self.ts=ts;self.parent=parent;self.shared=shared;self.prefix=prefix;self.keys=keys;self.vectors=vectors;self.policy=http.client.HTTPConnection('127.0.0.1',policy_port,timeout=5)
 def run(self,arm,q,delay):
  start=time.perf_counter_ns();t=q['tenant'];policy_calls=[];rounds=[];generated_filter=None
  def check(action,ids=None):
   before=time.perf_counter_ns();payload={'action':action,'tenant':t,'ids':ids or [],'delay_ms':delay};self.policy.request('POST','/policy',body=json.dumps(payload),headers={'Content-Type':'application/json'});r=self.policy.getresponse();data=json.loads(r.read());assert r.status==200;policy_calls.append({'action':action,'ids':ids or [],**data,'http_ms':(time.perf_counter_ns()-before)/1e6});return data
  key=self.keys['A2' if arm=='A2' else 'A1',t];collection=self.prefix+t if arm=='A2' else self.shared
  if arm=='P':
   state=check('list');generated_filter=f'tenant:={t} && active:=true'
   if state['revoked']:generated_filter+=' && rid:!=['+','.join(str(i) for i in state['revoked'])+']'
   key=base.scoped(self.parent,{'filter_by':generated_filter,'exclude_fields':'vec'})
  budgets=CFG['refill_budgets'] if arm=='R' else [int(arm[1:]) if arm.startswith('G') else 5]
  for budget in budgets:
   params={'collection':collection,'q':'*','vector_query':'vec:(['+','.join(format(float(x),'.8g') for x in self.vectors[q['source_index']])+f'],k:{budget},ef:100,flat_search_cutoff:0)','per_page':budget,'exclude_fields':'vec'}
   result=self.ts.call('POST','/multi_search',{'searches':[params]},key=key)['results'][0];assert 'hits' in result,result
   candidates=[int(h['document']['id']) for h in result['hits']]
   if arm.startswith('G') or arm=='R':
    state=check('check',candidates);allowed=set(state['allowed']);returned=[i for i in candidates if i in allowed][:5]
   else:returned=candidates[:5]
   rounds.append({'budget':budget,'candidates':candidates,'returned':returned})
   if arm!='R' or len(returned)>=5 or len(candidates)<budget:break
  end=time.perf_counter_ns();return {'returned':returned,'rounds':rounds,'policy_calls':policy_calls,'generated_filter':generated_filter,'latency_ms':(end-start)/1e6,'policy_http_ms':sum(c['http_ms'] for c in policy_calls),'start_ns':start,'end_ns':end}
def main():
 if (D/'requests.jsonl').exists():raise RuntimeError('Refusing to overwrite completed sweep')
 source=json.loads(SOURCE.read_text());records=source['records'];accounts=sorted({r['account_id'] for r in records});support={r['account_id']:i for i,r in enumerate(records) if r['kind']=='support'};queries=[c for c in source['cases'] if c['split']=='test'];dev=next(c for c in source['cases'] if c['split']=='development')
 sets={'lexical':np.load(ROOT/'experiments/tenant_scoped_retrieval/data/vectors.npz')['documents'],'neural':np.load(ROOT/'experiments/tenant_scoped_retrieval_v3/data/neural_vectors.npy')}
 dump('design.json',{'config':CFG,'runner_sha256':sha(Path(__file__)),'v3_helper_sha256':sha(ROOT/'experiments/tenant_scoped_retrieval_v3/run.py'),'source_sha256':sha(SOURCE),'frozen_before_measurement':True,'security_denominator':480,'timing_per_condition':960,'query_population':160,'random_rule':'random.Random(seed+9000).sample(range(1430), n); rate-specific sets; 143 uses previous sampling rule','scope':'only stale metadata; never updates active flags during these serial requests'})
 key=secrets.token_hex(24);name='retrieval-sweep-'+secrets.token_hex(4);ts=base.TS(key);ledger=v3.Ledger();server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.daemon_threads=True;server.ledger=ledger;threading.Thread(target=server.serve_forever,daemon=True).start();port=server.server_address[1]
 subprocess.run(['docker','run','-d','--rm','--name',name,'--cpus','2','--memory','2g','-p','127.0.0.1:8114:8108','--tmpfs','/data:rw,size=512m','-e',f'TYPESENSE_API_KEY={key}',CFG['image'],'--data-dir=/data','--api-port=8108'],check=True,capture_output=True)
 out=(D/'requests.partial.jsonl').open('w')
 try:
  for attempt in range(80):
   try:
    if ts.call('GET','/health')['ok']:break
   except Exception:ts.conn.close();ts=base.TS(key);time.sleep(.2)
  else:raise RuntimeError('search startup failed')
  dump('environment.json',{'typesense':ts.call('GET','/debug'),'search_cpu_limit':2,'search_memory_gib':2,'authority':'host outside search CPU limit','injected_wait':'monotonic wall-clock response wait after snapshot, no transport emulator'})
  for rep,vectors in sets.items():
   for seed in CFG['seeds']:
    a=accounts.copy();random.Random(seed).shuffle(a);assignment={account:f't{i%20:02}' for i,account in enumerate(a)};tenant=np.array([assignment[r['account_id']] for r in records]);ledger.tenants=tenant.tolist();prefix=f's{rep[0]}{seed}_';shared=prefix+'shared';cols=[shared]+[prefix+t for t in sorted(set(tenant))]
    for col in cols:
     ts.call('POST','/collections',{'name':col,'fields':[{'name':'tenant','type':'string','facet':True},{'name':'active','type':'bool','facet':True},{'name':'rid','type':'int32','facet':True},{'name':'vec','type':'float[]','num_dim':vectors.shape[1],'hnsw_params':{'M':16,'ef_construction':200}}]})
     ids=range(1430) if col==shared else [i for i,t in enumerate(tenant) if prefix+t==col];docs=[{'id':str(i),'rid':i,'tenant':tenant[i],'active':True,'vec':vectors[i].tolist()} for i in ids];response=ts.call('POST',f'/collections/{col}/documents/import?action=create','\n'.join(json.dumps(x) for x in docs),raw=True);assert all(json.loads(s)['success'] for s in response.splitlines())
    parent=ts.createkey([shared]);keys={}
    for t in sorted(set(tenant)):
     keys['A1',t]=base.scoped(parent,{'filter_by':f'tenant:={t} && active:=true','exclude_fields':'vec'});p=ts.createkey([prefix+t]);keys['A2',t]=base.scoped(p,{'filter_by':'active:=true','exclude_fields':'vec'})
    client=Client(ts,parent,shared,prefix,keys,vectors,port);qs=[{'case_id':q['case_id'],'tenant':assignment[q['account_id']],'source_index':support[q['account_id']]} for q in queries];scores=[vectors@vectors[q['source_index']] for q in qs]
    # Development-only check of exclusion syntax before measured queries.
    pilot={'tenant':assignment[dev['account_id']],'source_index':support[dev['account_id']]};removed=set(random.Random(seed+9000).sample(range(1430),143));ledger.commit(removed);response=client.run('P',pilot,0);assert all(i not in removed and tenant[i]==pilot['tenant'] for i in response['returned'])
    conditions=[('sweep',rate,n,0,CFG['strategies']) for rate,n in zip(CFG['rates_percent'],CFG['revoked_counts'])]+[('delay',10,143,wait,CFG['delay_strategies']) for wait in [1,5]]
    for suite,rate,n,wait,arms in conditions:
     removed=set(random.Random(seed+9000).sample(range(1430),n));ledger.commit(removed);active=np.array([i not in removed for i in range(1430)])
     for arm in arms:
      for q in qs[:3]:client.run(arm,q,wait)
     order=[(repeat,j,arm) for repeat in range(2) for j in range(160) for arm in arms];random.Random(seed+rate+wait).shuffle(order)
     for repeat,j,arm in order:
      q=qs[j];r=client.run(arm,q,wait);mask=(tenant==q['tenant'])&active;strict,ties,k=base.reference(scores[j],mask,5);returned=set(r['returned']);recall=(len(returned&strict)+min(len(returned&ties),k-len(strict)))/k
      row={'suite':suite,'representation':rep,'seed':seed,'rate_percent':rate,'revoked_count':n,'delay_ms':wait,'arm':arm,'repeat':repeat,'case_id':q['case_id'],'tenant':q['tenant'],'eligible':int(mask.sum()),'exposure':bool(any(not mask[i] for i in returned)),'false_denial':not returned,'recall':recall,**r};out.write(json.dumps(row)+'\n')
     out.flush();print('Measured',rep,seed,suite,rate,wait,flush=True)
    for col in cols:ts.call('DELETE',f'/collections/{col}')
    client.policy.close()
  out.close();(D/'requests.partial.jsonl').rename(D/'requests.jsonl')
 finally:out.close();server.shutdown();subprocess.run(['docker','stop',name],capture_output=True)
 print('Sweep and alternatives completed',flush=True)
if __name__=='__main__':main()
