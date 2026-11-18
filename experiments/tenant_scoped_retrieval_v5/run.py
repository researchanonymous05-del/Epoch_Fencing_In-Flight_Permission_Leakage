"""List representation scaling, page refill and post-commit concurrent validation."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,time,random,secrets,threading,socket,http.client,subprocess,hashlib,importlib.util
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parents[1];D=H/'data';D.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('admission_v4',ROOT/'experiments/tenant_scoped_retrieval_v4/run.py');v4=importlib.util.module_from_spec(spec);spec.loader.exec_module(v4);base=v4.base;base.CONFIG['port']=8115
CFG={'seeds':[11,29,47],'representations':['lexical','neural'],'list_sizes':[100,1000,10000],'list_cases':40,'list_case_seed':777,'list_arms':['P','Pbody','G20'],'refill_arms':['G20','P','R','RP'],'page_size':5,'page_horizon':80,'repeats':2,'revoked_count':143,'concurrent_workers':['A1','G20','P','R','RP'],'concurrent_seed':11,'concurrent_update_wait_ms':75,'concurrent_post_ack_ms':150,'port':8115,'image':'typesense/typesense:30.2','filter_by_max_ops':100,'padding':'per-tenant nonexistent numeric IDs >=1000000; selectivity held fixed; not 10k indexed revocations','body_route':'diagnostic only, fresh filter in trusted request body; not an immutable scoped-filter substitute','page_contract':'k=80 per_page=5 page increments; revalidate accumulated unique IDs each round; no cursor/snapshot claim','race':'concurrent requests start after authority commit; in-flight authority changes excluded'}
SOURCE=ROOT/'data/benchmark.json'
def dump(name,obj):(D/name).write_text(json.dumps(obj,indent=2)+'\n')
def digest(b):return hashlib.sha256(b).hexdigest()
def padded(real,tenant,count):
 if not count:return sorted(real)
 first=1_000_000+int(tenant[1:])*10000
 return sorted(real)+list(range(first,first+max(0,count-len(real))))
class Handler(BaseHTTPRequestHandler):
 protocol_version='HTTP/1.1'
 def setup(self):super().setup();self.connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
 def log_message(self,*args):pass
 def do_POST(self):
  req=json.loads(self.rfile.read(int(self.headers['Content-Length'])));state=self.server.ledger
  with state.lock:
   snap=time.perf_counter_ns();epoch=state.epoch
   if req['action']=='list':
    real=[i for i in sorted(state.revoked) if state.tenants[i]==req['tenant']];obj={'revoked':padded(real,req['tenant'],self.server.padding_count)}
   else:obj={'allowed':[i for i in req['ids'] if state.tenants[i]==req['tenant'] and i not in state.revoked]}
  obj.update({'epoch':epoch,'snapshot_ns':snap});raw=json.dumps(obj).encode();self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
class Client:
 def __init__(self,ts,parent,collection,keys,vectors,port):self.ts=ts;self.parent=parent;self.collection=collection;self.keys=keys;self.vectors=vectors;self.policy=http.client.HTTPConnection('127.0.0.1',port,timeout=10)
 def close(self):self.ts.conn.close();self.policy.close()
 def run(self,arm,q):
  start=time.perf_counter_ns();t=q['tenant'];calls=[];rounds=[];filter_bytes=0;key_bytes=0;filter_sha=None;list_count=0
  def check(action,ids=None):
   begin=time.perf_counter_ns();self.policy.request('POST','/policy',body=json.dumps({'action':action,'tenant':t,'ids':ids or []}),headers={'Content-Type':'application/json'});r=self.policy.getresponse();raw=r.read();assert r.status==200;obj=json.loads(raw);trace={'action':action,'epoch':obj['epoch'],'snapshot_ns':obj['snapshot_ns'],'http_ms':(time.perf_counter_ns()-begin)/1e6,'response_bytes':len(raw)}
   if action=='list':trace.update({'list_count':len(obj['revoked']),'list_sha256':digest(json.dumps(obj['revoked'],separators=(',',':')).encode())})
   else:trace.update({'ids':ids or [],'allowed':obj['allowed']})
   calls.append(trace);return obj
  key=self.keys[t];extra=None
  try:
   if arm in ['P','Pbody']:
    state=check('list');list_count=len(state['revoked']);clause=f'tenant:={t} && active:=true'
    if state['revoked']:clause+=' && rid:!=['+','.join(map(str,state['revoked']))+']'
    filter_bytes=len(clause.encode());filter_sha=digest(clause.encode())
    if arm=='P':key=base.scoped(self.parent,{'filter_by':clause,'exclude_fields':'vec'})
    else:extra=clause
   key_bytes=len(key.encode());seen=[]
   budgets=[5,10,20,40,80] if arm=='R' else ([80]*16 if arm=='RP' else [20 if arm=='G20' else 5])
   for index,budget in enumerate(budgets):
    params={'collection':self.collection,'q':'*','vector_query':'vec:(['+','.join(format(float(x),'.8g') for x in self.vectors[q['source_index']])+f'],k:{budget},ef:100,flat_search_cutoff:0)','per_page':5 if arm=='RP' else budget,'exclude_fields':'vec'}
    if arm=='RP':params['page']=index+1
    if extra:params['filter_by']=extra
    result=self.ts.call('POST','/multi_search',{'searches':[params]},key=key)['results'][0]
    if 'hits' not in result:raise base.APIError(result.get('code',500),'search result error')
    candidates=[int(h['document']['id']) for h in result['hits']]
    if arm=='RP':
     for i in candidates:
      if i not in seen:seen.append(i)
     state=check('check',seen);allowed=set(state['allowed']);returned=[i for i in seen if i in allowed][:5]
    elif arm in ['G20','R']:
     state=check('check',candidates);allowed=set(state['allowed']);returned=[i for i in candidates if i in allowed][:5]
    else:returned=candidates[:5]
    rounds.append({'budget':budget,'page':index+1 if arm=='RP' else None,'candidates':candidates,'returned':returned})
    if arm not in ['R','RP'] or len(returned)>=5 or len(candidates)<(5 if arm=='RP' else budget):break
   outcome={'success':True,'status':200,'returned':returned}
  except (base.APIError,OSError,http.client.HTTPException) as error:
   self.ts.conn.close();outcome={'success':False,'status':getattr(error,'status',None),'error_type':type(error).__name__,'returned':[]}
  end=time.perf_counter_ns();return {**outcome,'rounds':rounds,'policy_calls':calls,'list_count':list_count,'filter_bytes':filter_bytes,'filter_sha256':filter_sha,'key_bytes':key_bytes,'latency_ms':(end-start)/1e6,'start_ns':start,'end_ns':end}
def main():
 if (D/'requests.jsonl').exists():raise RuntimeError('Refusing to overwrite completed evidence')
 source=json.loads(SOURCE.read_text());records=source['records'];accounts=sorted({r['account_id'] for r in records});support={r['account_id']:i for i,r in enumerate(records) if r['kind']=='support'};queries=[q for q in source['cases'] if q['split']=='test'];subset=random.Random(777).sample(sorted(q['case_id'] for q in queries),40);sets={'lexical':np.load(ROOT/'experiments/tenant_scoped_retrieval/data/vectors.npz')['documents'],'neural':np.load(ROOT/'experiments/tenant_scoped_retrieval_v3/data/neural_vectors.npy')}
 dump('design.json',{'config':CFG,'runner_sha256':digest(Path(__file__).read_bytes()),'source_sha256':digest(SOURCE.read_bytes()),'list_subset_cases':subset,'frozen_before_measurement':True,'list_security_units_per_condition':120,'list_timings_per_condition':240,'refill_security_units_per_condition':480,'refill_timings_per_condition':960})
 key=secrets.token_hex(24);name='list-page-study-'+secrets.token_hex(4);ts=base.TS(key);ledger=v4.v3.Ledger();server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.daemon_threads=True;server.ledger=ledger;server.padding_count=0;threading.Thread(target=server.serve_forever,daemon=True).start();port=server.server_address[1]
 subprocess.run(['docker','run','-d','--rm','--name',name,'--cpus','2','--memory','2g','-p','127.0.0.1:8115:8108','--tmpfs','/data:rw,size=512m','-e',f'TYPESENSE_API_KEY={key}',CFG['image'],'--data-dir=/data','--api-port=8108','--filter-by-max-ops=100'],check=True,capture_output=True)
 out=(D/'requests.partial.jsonl').open('w');con=(D/'concurrent.partial.jsonl').open('w');windows=[]
 try:
  for attempt in range(80):
   try:
    if ts.call('GET','/health')['ok']:break
   except Exception:ts.conn.close();ts=base.TS(key);time.sleep(.2)
  else:raise RuntimeError('search startup failed')
  dump('environment.json',{'typesense':ts.call('GET','/debug'),'filter_by_max_ops':100,'max_per_page_default':250,'no_proxy':True,'routes':'direct localhost HTTP/1.1 to container','search_cpu_limit':2,'search_memory_gib':2})
  for rep,vectors in sets.items():
   for seed in [11,29,47]:
    a=accounts.copy();random.Random(seed).shuffle(a);owners={x:f't{i%20:02}' for i,x in enumerate(a)};tenant=np.array([owners[r['account_id']] for r in records]);ledger.tenants=tenant.tolist();collection=f's{rep[0]}{seed}';ts.call('POST','/collections',{'name':collection,'fields':[{'name':'tenant','type':'string','facet':True},{'name':'active','type':'bool','facet':True},{'name':'rid','type':'int32','facet':True},{'name':'vec','type':'float[]','num_dim':vectors.shape[1],'hnsw_params':{'M':16,'ef_construction':200}}]})
    docs=[{'id':str(i),'rid':i,'tenant':tenant[i],'active':True,'vec':vectors[i].tolist()} for i in range(1430)];raw=ts.call('POST',f'/collections/{collection}/documents/import?action=create','\n'.join(map(json.dumps,docs)),raw=True);assert all(json.loads(s)['success'] for s in raw.splitlines());parent=ts.createkey([collection]);keys={t:base.scoped(parent,{'filter_by':f'tenant:={t} && active:=true','exclude_fields':'vec'}) for t in sorted(set(tenant))};client=Client(ts,parent,collection,keys,vectors,port);qs=[{'case_id':q['case_id'],'tenant':owners[q['account_id']],'source_index':support[q['account_id']]} for q in queries];scores={q['case_id']:vectors@vectors[q['source_index']] for q in qs};removed=set(random.Random(seed+9000).sample(range(1430),143));ledger.commit(removed)
    conditions=[('lists',n,CFG['list_arms'],[q for q in qs if q['case_id'] in subset]) for n in CFG['list_sizes']]+[('refill',0,CFG['refill_arms'],qs)]
    for suite,n,arms,cohort in conditions:
     server.padding_count=n
     for arm in arms:
      for q in cohort[:2]:client.run(arm,q)
     order=[(repeat,j,arm) for repeat in [0,1] for j in range(len(cohort)) for arm in arms];random.Random(seed+n).shuffle(order)
     for repeat,j,arm in order:
      q=cohort[j];response=client.run(arm,q);mask=np.array([t==q['tenant'] and i not in removed for i,t in enumerate(tenant)]);strict,ties,k=base.reference(scores[q['case_id']],mask,5);ret=set(response['returned']);recall=(len(ret&strict)+min(len(ret&ties),k-len(strict)))/k;out.write(json.dumps({'suite':suite,'representation':rep,'seed':seed,'arm':arm,'requested_list_size':n,'repeat':repeat,'case_id':q['case_id'],'tenant':q['tenant'],'exposure':bool(any(not mask[i] for i in ret)),'false_denial':not ret,'recall':recall,**response})+'\n')
     out.flush();print('Measured',rep,seed,suite,n,flush=True)
    if seed==11:
     server.padding_count=0;target=set()
     for q in qs:
      eligible=np.flatnonzero(tenant==q['tenant']);target.add(int(eligible[np.argmax(scores[q['case_id']][eligible])]))
     commit,epoch=ledger.commit(target);barrier=threading.Barrier(6);stop=threading.Event();observations=[];errors=[]
     def worker(arm):
      c=Client(base.TS(key),parent,collection,keys,vectors,port);j=0
      try:
       barrier.wait()
       while not stop.is_set():
        q=qs[j%160];r=c.run(arm,q);r.update({'representation':rep,'seed':seed,'arm':arm,'case_id':q['case_id'],'tenant':q['tenant'],'exposes_revoked':bool(set(r['returned'])&target)});observations.append(r);j+=1
      except Exception as error:errors.append(type(error).__name__)
      finally:c.close()
     threads=[threading.Thread(target=worker,args=(arm,)) for arm in CFG['concurrent_workers']]
     for thread in threads:thread.start()
     barrier.wait();time.sleep(.075);begin=time.perf_counter_ns();raw=ts.call('POST',f'/collections/{collection}/documents/import?action=update','\n'.join(json.dumps({'id':str(i),'active':False}) for i in sorted(target)),raw=True);assert all(json.loads(s)['success'] for s in raw.splitlines());ack=time.perf_counter_ns();time.sleep(.15);stop.set()
     for thread in threads:thread.join()
     assert not errors,errors
     for row in observations:row.update({'commit_ns':commit,'update_start_ns':begin,'update_ack_ns':ack});con.write(json.dumps(row)+'\n')
     con.flush();windows.append({'representation':rep,'seed':seed,'commit_ns':commit,'update_start_ns':begin,'update_ack_ns':ack,'observations':len(observations)});print('Concurrent complete',rep,flush=True)
    ts.call('DELETE',f'/collections/{collection}');client.policy.close()
  out.close();con.close();(D/'requests.partial.jsonl').rename(D/'requests.jsonl');(D/'concurrent.partial.jsonl').rename(D/'concurrent.jsonl');dump('windows.json',windows)
 finally:out.close();con.close();server.shutdown();subprocess.run(['docker','stop',name],capture_output=True)
 print('Completed list, page and concurrent extensions',flush=True)
if __name__=='__main__':main()
