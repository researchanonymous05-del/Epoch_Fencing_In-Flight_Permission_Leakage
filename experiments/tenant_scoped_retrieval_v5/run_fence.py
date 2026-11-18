"""Epoch-fenced prefilter (F) and in-flight revocation race against P, G20 and A1."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,time,random,secrets,threading,socket,http.client,subprocess,hashlib,importlib.util
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parents[1];D=H/'data_fence';D.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('admission_v4',ROOT/'experiments/tenant_scoped_retrieval_v4/run.py');v4=importlib.util.module_from_spec(spec);spec.loader.exec_module(v4);base=v4.base;base.CONFIG['port']=8116
CFG={'seeds':[11,29,47],'representations':['lexical','neural'],'revoked_count':143,'race_arms':['A1','P','G20','F'],'cost_arms':['P','G20','F'],'waits_ms':[0,5],'race_horizon_ms':{'0':20,'5':40},'race_offsets_per_case':2,'cost_repeats':2,'fence_retries':1,'port':8116,'image':'typesense/typesense:30.2',
 'fence':'F = P prefilter at list epoch e0, then one policy call over the <=5 IDs about to be returned; fast path returns IDs unchanged when the authority epoch still equals e0; if the fence drops any ID, rerun P once with a fresh list and fence again',
 'race':'victim = query top eligible document under the base 10% random revocation; a revoker thread commits base+victim at start+U, U~Uniform[0,horizon]; authority restored after each trial; index flags never change',
 'exposure':'victim returned and authority commit_ns < client end_ns (request-end freshness); stale base exposure scored separately',
 'wait':'policy response wait after the authority snapshot, once per HTTP call; not a WAN model'}
SOURCE=ROOT/'data/benchmark.json'
def dump(name,obj):(D/name).write_text(json.dumps(obj,indent=2)+'\n')
def digest(b):return hashlib.sha256(b).hexdigest()
class Handler(BaseHTTPRequestHandler):
 protocol_version='HTTP/1.1'
 def setup(self):super().setup();self.connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
 def log_message(self,*args):pass
 def do_POST(self):
  req=json.loads(self.rfile.read(int(self.headers['Content-Length'])));state=self.server.ledger
  with state.lock:
   snap=time.perf_counter_ns();epoch=state.epoch
   if req['action']=='list':obj={'revoked':[i for i in sorted(state.revoked) if state.tenants[i]==req['tenant']]}
   elif req['action']=='fence' and req['epoch']==epoch:obj={'allowed':req['ids'],'fast':True}
   else:obj={'allowed':[i for i in req['ids'] if state.tenants[i]==req['tenant'] and i not in state.revoked],'fast':False}
  if req['wait_ms']:time.sleep(req['wait_ms']/1000)
  obj.update({'epoch':epoch,'snapshot_ns':snap});raw=json.dumps(obj).encode();self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
class Client:
 def __init__(self,ts,parent,collection,keys,vectors,port):self.ts=ts;self.parent=parent;self.collection=collection;self.keys=keys;self.vectors=vectors;self.policy=http.client.HTTPConnection('127.0.0.1',port,timeout=10)
 def close(self):self.policy.close()
 def run(self,arm,q,wait):
  start=time.perf_counter_ns();t=q['tenant'];calls=[];attempts=[]
  def call(action,ids=None,epoch=None):
   begin=time.perf_counter_ns();self.policy.request('POST','/policy',body=json.dumps({'action':action,'tenant':t,'ids':list(ids or []),'epoch':epoch,'wait_ms':wait}),headers={'Content-Type':'application/json'});r=self.policy.getresponse();obj=json.loads(r.read());assert r.status==200
   calls.append({'action':action,'ids':list(ids or []),'epoch_in':epoch,**obj,'http_ms':(time.perf_counter_ns()-begin)/1e6});return obj
  def search(key,budget):
   params={'collection':self.collection,'q':'*','vector_query':'vec:(['+','.join(format(float(x),'.8g') for x in self.vectors[q['source_index']])+f'],k:{budget},ef:100,flat_search_cutoff:0)','per_page':budget,'exclude_fields':'vec'}
   result=self.ts.call('POST','/multi_search',{'searches':[params]},key=key)['results'][0];assert 'hits' in result,result;return [int(h['document']['id']) for h in result['hits']]
  if arm in ['A1','G20']:
   candidates=search(self.keys[t],20 if arm=='G20' else 5)
   if arm=='G20':allowed=set(call('check',candidates)['allowed']);returned=[i for i in candidates if i in allowed][:5]
   else:returned=candidates[:5]
   attempts.append({'candidates':candidates,'returned':returned})
  else:
   for attempt in range(1+(CFG['fence_retries'] if arm=='F' else 0)):
    state=call('list');clause=f'tenant:={t} && active:=true'
    if state['revoked']:clause+=' && rid:!=['+','.join(map(str,state['revoked']))+']'
    candidates=search(base.scoped(self.parent,{'filter_by':clause,'exclude_fields':'vec'}),5);returned=candidates[:5]
    if arm=='F':
     fence=call('fence',returned,state['epoch']);kept=set(fence['allowed']);dropped=[i for i in returned if i not in kept];returned=[i for i in returned if i in kept]
     attempts.append({'list_epoch':state['epoch'],'candidates':candidates,'dropped':dropped,'returned':returned})
     if not dropped:break
    else:attempts.append({'list_epoch':state['epoch'],'candidates':candidates,'returned':returned})
  end=time.perf_counter_ns();return {'returned':returned,'attempts':attempts,'policy_calls':calls,'latency_ms':(end-start)/1e6,'start_ns':start,'end_ns':end}
def main():
 if (D/'race.jsonl').exists():raise RuntimeError('Refusing to overwrite completed evidence')
 source=json.loads(SOURCE.read_text());records=source['records'];accounts=sorted({r['account_id'] for r in records});support={r['account_id']:i for i,r in enumerate(records) if r['kind']=='support'};queries=[q for q in source['cases'] if q['split']=='test'];dev=next(c for c in source['cases'] if c['split']=='development')
 sets={'lexical':np.load(ROOT/'experiments/tenant_scoped_retrieval/data/vectors.npz')['documents'],'neural':np.load(ROOT/'experiments/tenant_scoped_retrieval_v3/data/neural_vectors.npy')}
 dump('design.json',{'config':CFG,'runner_sha256':digest(Path(__file__).read_bytes()),'v4_helper_sha256':digest((ROOT/'experiments/tenant_scoped_retrieval_v4/run.py').read_bytes()),'source_sha256':digest(SOURCE.read_bytes()),'frozen_before_measurement':True,'race_trials_per_condition':960,'cost_timings_per_condition':960,'query_population':160})
 key=secrets.token_hex(24);name='fence-race-'+secrets.token_hex(4);ts=base.TS(key);ledger=v4.v3.Ledger();server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.daemon_threads=True;server.ledger=ledger;threading.Thread(target=server.serve_forever,daemon=True).start();port=server.server_address[1]
 subprocess.run(['docker','run','-d','--rm','--name',name,'--cpus','2','--memory','2g','-p','127.0.0.1:8116:8108','--tmpfs','/data:rw,size=512m','-e',f'TYPESENSE_API_KEY={key}',CFG['image'],'--data-dir=/data','--api-port=8108'],check=True,capture_output=True)
 race=(D/'race.partial.jsonl').open('w');cost=(D/'cost.partial.jsonl').open('w')
 try:
  for attempt in range(80):
   try:
    if ts.call('GET','/health')['ok']:break
   except Exception:ts.conn.close();ts=base.TS(key);time.sleep(.2)
  else:raise RuntimeError('search startup failed')
  dump('environment.json',{'typesense':ts.call('GET','/debug'),'search_cpu_limit':2,'search_memory_gib':2,'authority':'host process, outside search CPU limit','revoker':'separate thread; time.sleep to target then ledger commit under the authority lock'})
  for rep,vectors in sets.items():
   for seed in CFG['seeds']:
    a=accounts.copy();random.Random(seed).shuffle(a);owner={x:f't{i%20:02}' for i,x in enumerate(a)};tenant=np.array([owner[r['account_id']] for r in records]);ledger.tenants=tenant.tolist();collection=f'f{rep[0]}{seed}'
    ts.call('POST','/collections',{'name':collection,'fields':[{'name':'tenant','type':'string','facet':True},{'name':'active','type':'bool','facet':True},{'name':'rid','type':'int32','facet':True},{'name':'vec','type':'float[]','num_dim':vectors.shape[1],'hnsw_params':{'M':16,'ef_construction':200}}]})
    docs=[{'id':str(i),'rid':i,'tenant':tenant[i],'active':True,'vec':vectors[i].tolist()} for i in range(1430)];raw=ts.call('POST',f'/collections/{collection}/documents/import?action=create','\n'.join(map(json.dumps,docs)),raw=True);assert all(json.loads(s)['success'] for s in raw.splitlines())
    parent=ts.createkey([collection]);keys={t:base.scoped(parent,{'filter_by':f'tenant:={t} && active:=true','exclude_fields':'vec'}) for t in sorted(set(tenant))};client=Client(ts,parent,collection,keys,vectors,port)
    qs=[{'case_id':q['case_id'],'tenant':owner[q['account_id']],'source_index':support[q['account_id']]} for q in queries];scores={q['case_id']:vectors@vectors[q['source_index']] for q in qs}
    removed=set(random.Random(seed+9000).sample(range(1430),CFG['revoked_count']));ledger.commit(removed);active=np.array([i not in removed for i in range(1430)])
    victims={q['case_id']:int(max(np.flatnonzero((tenant==q['tenant'])&active),key=lambda i:scores[q['case_id']][i])) for q in qs}
    pilot={'tenant':owner[dev['account_id']],'source_index':support[dev['account_id']]};r=client.run('F',pilot,0);assert all(i not in removed and tenant[i]==pilot['tenant'] for i in r['returned'])
    for wait in CFG['waits_ms']:
     # Serial cost without in-flight changes.
     for arm in CFG['cost_arms']:
      for q in qs[:3]:client.run(arm,q,wait)
     order=[(rep_i,j,arm) for rep_i in range(CFG['cost_repeats']) for j in range(160) for arm in CFG['cost_arms']];random.Random(seed*10+wait).shuffle(order)
     for rep_i,j,arm in order:
      q=qs[j];r=client.run(arm,q,wait);mask=(tenant==q['tenant'])&active;strict,ties,k=base.reference(scores[q['case_id']],mask,5);ret=set(r['returned']);recall=(len(ret&strict)+min(len(ret&ties),k-len(strict)))/k
      cost.write(json.dumps({'representation':rep,'seed':seed,'wait_ms':wait,'arm':arm,'repeat':rep_i,'case_id':q['case_id'],'tenant':q['tenant'],'exposure':bool(any(not mask[i] for i in ret)),'false_denial':not ret,'recall':recall,**r})+'\n')
     cost.flush()
     # In-flight revocation race.
     horizon=CFG['race_horizon_ms'][str(wait)]*1_000_000;rng=random.Random(seed*100+wait+7)
     order=[(o,j,arm) for o in range(CFG['race_offsets_per_case']) for j in range(160) for arm in CFG['race_arms']];random.Random(seed*1000+wait).shuffle(order);offsets={(o,j,arm):rng.randrange(horizon) for o,j,arm in sorted(order)}
     for o,j,arm in order:
      q=qs[j];victim=victims[q['case_id']];box={}
      def revoke(target):
       delay=(target-time.perf_counter_ns())/1e9
       if delay>0:time.sleep(delay)
       box['commit_ns'],box['epoch']=ledger.commit(removed|{victim})
      t0=time.perf_counter_ns();target=t0+offsets[o,j,arm];thread=threading.Thread(target=revoke,args=(target,));thread.start()
      r=client.run(arm,q,wait);thread.join();ledger.commit(removed)
      revoked_at_end=r['end_ns']>box['commit_ns'];mask=(tenant==q['tenant'])&active
      if revoked_at_end:mask[victim]=False
      strict,ties,k=base.reference(scores[q['case_id']],mask,5);ret=set(r['returned']);recall=(len(ret&strict)+min(len(ret&ties),k-len(strict)))/k
      race.write(json.dumps({'representation':rep,'seed':seed,'wait_ms':wait,'arm':arm,'offset_index':o,'case_id':q['case_id'],'tenant':q['tenant'],'victim':victim,'horizon_ns':horizon,'t0_ns':t0,'target_ns':target,'commit_ns':box['commit_ns'],'commit_epoch':box['epoch'],'revoked_before_end':revoked_at_end,'race_exposure':bool(victim in ret and revoked_at_end),'stale_exposure':bool(any(i in removed for i in ret)),'recall':recall,**r})+'\n')
     race.flush();print('Measured',rep,seed,'wait',wait,flush=True)
    ts.call('DELETE',f'/collections/{collection}');client.close()
  race.close();cost.close();(D/'race.partial.jsonl').rename(D/'race.jsonl');(D/'cost.partial.jsonl').rename(D/'cost.jsonl')
 finally:race.close();cost.close();server.shutdown();subprocess.run(['docker','stop',name],capture_output=True)
 print('Completed fence and race study',flush=True)
if __name__=='__main__':main()
