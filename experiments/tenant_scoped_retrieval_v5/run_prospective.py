"""Prospective test of the placement model: measure the model-chosen budgets G6/G7/G9 after freezing predictions."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,time,random,secrets,threading,subprocess,hashlib,importlib.util
from pathlib import Path
from http.server import ThreadingHTTPServer
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parents[1];D=H/'data_prospective';D.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('admission_v4',ROOT/'experiments/tenant_scoped_retrieval_v4/run.py');v4=importlib.util.module_from_spec(spec);spec.loader.exec_module(v4);base=v4.base;base.CONFIG['port']=8117
CFG={'seeds':[11,29,47],'representations':['lexical','neural'],'conditions':[[5,72,'G6'],[10,143,'G7'],[25,358,'G9']],'reference':'G20','repeats':2,'port':8117,'image':'typesense/typesense:30.2','predictions':'data_model/prospective_predictions.json (frozen before this run)'}
SOURCE=ROOT/'data/benchmark.json'
def digest(b):return hashlib.sha256(b).hexdigest()
def main():
 if (D/'requests.jsonl').exists():raise RuntimeError('Refusing to overwrite completed evidence')
 frozen=H/'data_model/prospective_predictions.json';assert frozen.exists()
 source=json.loads(SOURCE.read_text());records=source['records'];accounts=sorted({r['account_id'] for r in records});support={r['account_id']:i for i,r in enumerate(records) if r['kind']=='support'};queries=[q for q in source['cases'] if q['split']=='test']
 sets={'lexical':np.load(ROOT/'experiments/tenant_scoped_retrieval/data/vectors.npz')['documents'],'neural':np.load(ROOT/'experiments/tenant_scoped_retrieval_v3/data/neural_vectors.npy')}
 (D/'design.json').write_text(json.dumps({'config':CFG,'runner_sha256':digest(Path(__file__).read_bytes()),'predictions_sha256':digest(frozen.read_bytes()),'source_sha256':digest(SOURCE.read_bytes())},indent=2)+'\n')
 key=secrets.token_hex(24);name='prospective-'+secrets.token_hex(4);ts=base.TS(key);ledger=v4.v3.Ledger();server=ThreadingHTTPServer(('127.0.0.1',0),v4.Handler);server.daemon_threads=True;server.ledger=ledger;threading.Thread(target=server.serve_forever,daemon=True).start();port=server.server_address[1]
 subprocess.run(['docker','run','-d','--rm','--name',name,'--cpus','2','--memory','2g','-p','127.0.0.1:8117:8108','--tmpfs','/data:rw,size=512m','-e',f'TYPESENSE_API_KEY={key}',CFG['image'],'--data-dir=/data','--api-port=8108'],check=True,capture_output=True)
 out=(D/'requests.partial.jsonl').open('w')
 try:
  for attempt in range(80):
   try:
    if ts.call('GET','/health')['ok']:break
   except Exception:ts.conn.close();ts=base.TS(key);time.sleep(.2)
  else:raise RuntimeError('search startup failed')
  for rep,vectors in sets.items():
   for seed in CFG['seeds']:
    a=accounts.copy();random.Random(seed).shuffle(a);owner={x:f't{i%20:02}' for i,x in enumerate(a)};tenant=np.array([owner[r['account_id']] for r in records]);ledger.tenants=tenant.tolist();shared=f'p{rep[0]}{seed}'
    ts.call('POST','/collections',{'name':shared,'fields':[{'name':'tenant','type':'string','facet':True},{'name':'active','type':'bool','facet':True},{'name':'rid','type':'int32','facet':True},{'name':'vec','type':'float[]','num_dim':vectors.shape[1],'hnsw_params':{'M':16,'ef_construction':200}}]})
    docs=[{'id':str(i),'rid':i,'tenant':tenant[i],'active':True,'vec':vectors[i].tolist()} for i in range(1430)];raw=ts.call('POST',f'/collections/{shared}/documents/import?action=create','\n'.join(map(json.dumps,docs)),raw=True);assert all(json.loads(s)['success'] for s in raw.splitlines())
    parent=ts.createkey([shared]);keys={('A1',t):base.scoped(parent,{'filter_by':f'tenant:={t} && active:=true','exclude_fields':'vec'}) for t in sorted(set(tenant))}
    client=v4.Client(ts,parent,shared,'unused_',keys,vectors,port);qs=[{'case_id':q['case_id'],'tenant':owner[q['account_id']],'source_index':support[q['account_id']]} for q in queries];scores=[vectors@vectors[q['source_index']] for q in qs]
    for rate,n,arm in CFG['conditions']:
     removed=set(random.Random(seed+9000).sample(range(1430),n));ledger.commit(removed);active=np.array([i not in removed for i in range(1430)]);arms=[arm,CFG['reference']]
     for x in arms:
      for q in qs[:3]:client.run(x,q,0)
     order=[(r_i,j,x) for r_i in range(CFG['repeats']) for j in range(160) for x in arms];random.Random(seed+rate+77).shuffle(order)
     for r_i,j,x in order:
      q=qs[j];r=client.run(x,q,0);mask=(tenant==q['tenant'])&active;strict,ties,k=base.reference(scores[j],mask,5);ret=set(r['returned']);recall=(len(ret&strict)+min(len(ret&ties),k-len(strict)))/k
      out.write(json.dumps({'representation':rep,'seed':seed,'rate_percent':rate,'revoked_count':n,'arm':x,'repeat':r_i,'case_id':q['case_id'],'tenant':q['tenant'],'exposure':bool(any(not mask[i] for i in ret)),'false_denial':not ret,'recall':recall,**r})+'\n')
     out.flush();print('Measured',rep,seed,rate,flush=True)
    ts.call('DELETE',f'/collections/{shared}');client.policy.close()
  out.close();(D/'requests.partial.jsonl').rename(D/'requests.jsonl')
 finally:out.close();server.shutdown();subprocess.run(['docker','stop',name],capture_output=True)
 print('Prospective run completed',flush=True)
if __name__=='__main__':main()
