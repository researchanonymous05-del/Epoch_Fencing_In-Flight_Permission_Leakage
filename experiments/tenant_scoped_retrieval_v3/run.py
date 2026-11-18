"""Fresh admission checks, random revocations and concurrent propagation measurements."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import sys,json,hashlib,random,secrets,subprocess,time,threading,http.client,socket
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parents[1];D=H/'data';D.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'experiments/tenant_scoped_retrieval'));import run as base
base.CONFIG['port']=8113
CFG={'seeds':[11,29,47],'tenants':20,'arms':['A1','A2','G5','G20'],'representations':['lexical','neural'],'regimes':['targeted','random10'],'phases':['current','stale','applied'],'repeats':2,'k':5,'random_revoke_fraction':.10,'ef':100,'image':'typesense/typesense:30.2','port':8113,'concurrent_delays_ms':[0,25,100],'concurrent_post_ack_ms':150,'authority':'single-process HTTP batch service, serialized immutable ledger snapshots; no cache','gate':'search top 5 or 20 in shared scoped index; batch HTTP current-policy check; keep first 5 allowed; fail closed on HTTP errors or old epoch'}
SOURCE=ROOT/'data/benchmark.json'
def dump(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class Ledger:
 def __init__(self):self.lock=threading.Lock();self.tenants=[];self.revoked=set();self.epoch=0;self.fail=False
 def commit(self,removed):
  with self.lock:self.revoked=set(removed);self.epoch+=1;return time.perf_counter_ns(),self.epoch
class PolicyHandler(BaseHTTPRequestHandler):
 protocol_version='HTTP/1.1'
 def setup(self):
  super().setup();self.connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
 def log_message(self,*args):pass
 def do_POST(self):
  data=json.loads(self.rfile.read(int(self.headers['Content-Length'])));state=self.server.state
  with state.lock:
   snapshot_ns=time.perf_counter_ns();epoch=state.epoch;fail=state.fail
   allowed=[i for i in data['ids'] if state.tenants[i]==data['tenant'] and i not in state.revoked]
  body=json.dumps({'allowed':allowed,'epoch':epoch,'snapshot_ns':snapshot_ns}).encode();self.send_response(503 if fail else 200);self.send_header('Content-Length',str(len(body)));self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(body)
class Client:
 def __init__(self,ts,key,shared,prefix,keys,tenant,vectors,ledger,policy_port):
  self.ts=ts;self.key=key;self.shared=shared;self.prefix=prefix;self.keys=keys;self.tenant=tenant;self.vectors=vectors;self.ledger=ledger;self.policy=http.client.HTTPConnection('127.0.0.1',policy_port,timeout=5)
 def search(self,arm,q):
  start=time.perf_counter_ns();t=q['tenant'];budget=20 if arm=='G20' else 5;key=self.keys[('A2' if arm=='A2' else 'A1',t)];col=self.prefix+t if arm=='A2' else self.shared
  required=self.ledger.epoch;vec=self.vectors[q['source_index']]
  params={'collection':col,'q':'*','vector_query':'vec:(['+','.join(format(float(v),'.8g') for v in vec)+f'],k:{budget},ef:100,flat_search_cutoff:0)','per_page':budget,'exclude_fields':'vec'}
  result=self.ts.call('POST','/multi_search',{'searches':[params]},key=key)['results'][0]
  if 'hits' not in result:raise RuntimeError(result)
  candidates=[int(h['document']['id']) for h in result['hits']];retrieved=time.perf_counter_ns();policy_ms=0;gate_error=False;epoch=None;snapshot=None
  if arm in ['G5','G20']:
   req=json.dumps({'tenant':t,'ids':candidates});begin=time.perf_counter_ns();self.policy.request('POST','/check',body=req,headers={'Content-Type':'application/json'});response=self.policy.getresponse();data=json.loads(response.read());policy_ms=(time.perf_counter_ns()-begin)/1e6;epoch=data['epoch'];snapshot=data['snapshot_ns'];gate_error=response.status!=200 or epoch<required;allowed=set(data['allowed']) if not gate_error else set();returned=[i for i in candidates if i in allowed][:5]
  else:returned=candidates[:5]
  end=time.perf_counter_ns()
  return {'returned':returned,'candidates':candidates,'start_ns':start,'retrieved_ns':retrieved,'end_ns':end,'policy_snapshot_ns':snapshot,'policy_epoch':epoch,'gate_error':gate_error,'latency_ms':(end-start)/1e6,'policy_http_ms':policy_ms}
 def close(self):self.ts.conn.close();self.policy.close()
def main():
 if (D/'requests.jsonl').exists():raise RuntimeError('Refusing to overwrite a completed run')
 source=json.loads(SOURCE.read_text());records=source['records'];accounts=sorted({r['account_id'] for r in records});support={r['account_id']:i for i,r in enumerate(records) if r['kind']=='support'};test=[c for c in source['cases'] if c['split']=='test'];texts=[base.text_record(r) for r in records]
 lexical=np.load(ROOT/'experiments/tenant_scoped_retrieval/data/vectors.npz')['documents']
 if (D/'neural_vectors.npy').exists():neural=np.load(D/'neural_vectors.npy')
 else:raise RuntimeError('Run prepare_neural.py separately before the measurement process')
 assert neural.shape==(1430,384)
 vectorsets={'lexical':lexical,'neural':neural}
 dump(D/'design.json',{'config':CFG,'runner_sha256':sha(Path(__file__)),'source_sha256':sha(SOURCE),'neural_revision':'1110a243fdf4706b3f48f1d95db1a4f5529b4d41','vector_sha256':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in vectorsets.items()},'frozen_before_new_measured_run':True,'queries':'160 source-support vectors reused; three ownership assignments, not independent queries','security_denominator':'480 case-seed units per representation/regime/phase/arm; repetitions reserved for timing','policy_contract':'check against current ledger snapshot, not time of client receipt; no distributed-consistency claim'})
 key=secrets.token_hex(24);name='admission-study-'+secrets.token_hex(4);admin=base.TS(key);ledger=Ledger();server=ThreadingHTTPServer(('127.0.0.1',0),PolicyHandler);server.daemon_threads=True;server.state=ledger;threading.Thread(target=server.serve_forever,daemon=True).start();policy_port=server.server_address[1]
 subprocess.run(['docker','run','-d','--rm','--name',name,'--cpus','2','--memory','2g','-p','127.0.0.1:8113:8108','--tmpfs','/data:rw,size=512m','-e',f'TYPESENSE_API_KEY={key}',CFG['image'],'--data-dir=/data','--api-port=8108'],check=True,capture_output=True)
 outfile=(D/'requests.partial.jsonl').open('w');windowfile=(D/'concurrent.partial.jsonl').open('w');faults=[];updates=[];windows=[]
 try:
  for attempt in range(80):
   try:
    if admin.call('GET','/health')['ok']:break
   except Exception:admin.conn.close();admin=base.TS(key);time.sleep(.2)
  else:raise RuntimeError('search startup failed')
  debug=admin.call('GET','/debug');dump(D/'environment.json',{'typesense_debug':debug,'cpu_limit':2,'memory_gib':2,'policy_port_randomized':True,'neural_inference':'offline CPU; embedding time excluded from retrieval latency'})
  for representation,vectors in vectorsets.items():
   for seed in CFG['seeds']:
    rng=random.Random(seed);shuffled=accounts.copy();rng.shuffle(shuffled);assignment={a:f't{i%20:02}' for i,a in enumerate(shuffled)};tenant=[assignment[r['account_id']] for r in records];tenantarr=np.array(tenant);ledger.tenants=tenant
    queries=[{'case_id':c['case_id'],'tenant':assignment[c['account_id']],'source_index':support[c['account_id']]} for c in test];scores=[vectors@vectors[q['source_index']] for q in queries]
    prefix=f'x{representation[0]}{seed}_';shared=prefix+'shared';cols=[shared]+[prefix+t for t in sorted(set(tenant))];keys={}
    for col in cols:
     admin.call('POST','/collections',{'name':col,'fields':[{'name':'tenant','type':'string','facet':True},{'name':'active','type':'bool','facet':True},{'name':'vec','type':'float[]','num_dim':vectors.shape[1],'hnsw_params':{'M':16,'ef_construction':200}}]})
     ids=list(range(len(records))) if col==shared else [i for i,t in enumerate(tenant) if prefix+t==col]
     docs=[{'id':str(i),'tenant':tenant[i],'active':True,'vec':vectors[i].tolist()} for i in ids];response=admin.call('POST',f'/collections/{col}/documents/import?action=create','\n'.join(json.dumps(x) for x in docs),raw=True);assert all(json.loads(s)['success'] for s in response.splitlines())
    parent=admin.createkey([shared])
    for t in sorted(set(tenant)):
     keys['A1',t]=base.scoped(parent,{'filter_by':f'tenant:={t} && active:=true','exclude_fields':'vec'})
     p=admin.createkey([prefix+t]);keys['A2',t]=base.scoped(p,{'filter_by':'active:=true','exclude_fields':'vec'})
    client=Client(admin,key,shared,prefix,keys,tenant,vectors,ledger,policy_port)
    def update(ids,active):
     start=time.perf_counter_ns();acks={}
     # Alternate organization order to disclose and reduce fixed propagation-order bias.
     ordered=cols if seed!=29 else cols[1:]+[shared]
     for col in ordered:
      values=[{'id':str(i),'active':active} for i in sorted(ids) if col==shared or col==prefix+tenant[i]]
      if values:
       response=admin.call('POST',f'/collections/{col}/documents/import?action=update','\n'.join(json.dumps(x) for x in values),raw=True);assert all(json.loads(s)['success'] for s in response.splitlines());acks[col]=time.perf_counter_ns()
     return start,time.perf_counter_ns(),acks
    target=set()
    for q,s in zip(queries,scores):
     eligible=np.flatnonzero(tenantarr==q['tenant']);target.add(int(eligible[np.argmax(s[eligible])]))
    randomset=set(random.Random(seed+9000).sample(range(len(records)),143))
    for regime,removed in [('targeted',target),('random10',randomset)]:
     update(set(range(len(records))),True);ledger.commit(set())
     for phase in CFG['phases']:
      if phase=='stale':ledger.commit(removed)
      if phase=='applied':
       start,end,acks=update(removed,False);updates.append({'representation':representation,'seed':seed,'regime':regime,'revoked':len(removed),'batch_ms':(end-start)/1e6})
      active=np.ones(len(records),bool)
      if phase!='current':active[list(removed)]=False
      for arm in CFG['arms']:
       for q in queries[:3]:client.search(arm,q)
      order=[(repeat,j,arm) for repeat in range(CFG['repeats']) for j in range(160) for arm in CFG['arms']];random.Random(seed+len(phase)+len(regime)).shuffle(order)
      for repeat,j,arm in order:
       q=queries[j];response=client.search(arm,q);mask=(tenantarr==q['tenant'])&active;strict,ties,n=base.reference(scores[j],mask,5);returned=set(response['returned']);recall=(len(returned&strict)+min(len(returned&ties),n-len(strict)))/n if n else None
       row={'representation':representation,'seed':seed,'regime':regime,'phase':phase,'arm':arm,'repeat':repeat,'case_id':q['case_id'],'tenant':q['tenant'],'eligible':int(mask.sum()),'exposure':any(not mask[i] for i in returned),'recall':recall,'false_denial':bool(n and not returned),**response};outfile.write(json.dumps(row)+'\n')
      outfile.flush();print('Measured',representation,seed,regime,phase,flush=True)
    # Concurrent visibility trial: targeted revocation, four independent worker connections.
    update(set(range(len(records))),True);ledger.commit(set());barrier=threading.Barrier(5);done=threading.Event();concurrent_rows=[];errors=[];commit_ns=None
    def worker(arm):
     ts=base.TS(key);c=Client(ts,key,shared,prefix,keys,tenant,vectors,ledger,policy_port);j=0
     try:
      barrier.wait()
      while not done.is_set():
       q=queries[j%160];row=c.search(arm,q);row.update({'representation':representation,'seed':seed,'arm':arm,'case_id':q['case_id'],'exposes_revoked':bool(set(row['returned'])&target)});concurrent_rows.append(row);j+=1
     except Exception as error:errors.append(str(error))
     finally:c.close()
    commit_ns,epoch=ledger.commit(target)
    threads=[threading.Thread(target=worker,args=(arm,)) for arm in CFG['arms']]
    for t in threads:t.start()
    barrier.wait();delay=CFG['concurrent_delays_ms'][CFG['seeds'].index(seed)];time.sleep(delay/1000);start,end,acks=update(target,False);time.sleep(CFG['concurrent_post_ack_ms']/1000);done.set()
    for t in threads:t.join()
    assert not errors,errors
    for row in concurrent_rows:row['commit_ns']=commit_ns;row['all_updates_ack_ns']=end;row['update_start_ns']=start;row['injected_delay_ms']=delay;windowfile.write(json.dumps(row)+'\n')
    windowfile.flush();windows.append({'representation':representation,'seed':seed,'injected_delay_ms':delay,'commit_ns':commit_ns,'update_start_ns':start,'all_updates_ack_ns':end,'acks_ns':acks,'observations':len(concurrent_rows)})
    # Explicit policy-service failure injection (HTTP 503), not a network outage claim.
    if seed==11:
     ledger.fail=True
     for arm in ['G5','G20']:
      for q in queries:
       response=client.search(arm,q);faults.append({'representation':representation,'arm':arm,'case_id':q['case_id'],**response})
     ledger.fail=False
    for col in cols:admin.call('DELETE',f'/collections/{col}')
    print('Completed concurrent/fault tests',representation,seed,flush=True)
  outfile.close();windowfile.close();(D/'requests.partial.jsonl').rename(D/'requests.jsonl');(D/'concurrent.partial.jsonl').rename(D/'concurrent.jsonl');dump(D/'updates.json',updates);dump(D/'windows.json',windows);dump(D/'faults.json',faults)
 finally:
  outfile.close();windowfile.close();server.shutdown();subprocess.run(['docker','stop',name],capture_output=True)
 print('Run complete',flush=True)
if __name__=='__main__':main()
