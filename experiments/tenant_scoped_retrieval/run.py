"""Isolated Typesense A0/A1/A2 experiment. No production connection or paid API."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import base64,hashlib,hmac,http.client,json,platform,random,secrets,subprocess,time,sys
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import sklearn

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'data';OUT.mkdir(exist_ok=True)
SOURCE=ROOT/'data/benchmark.json'
CONFIG={'seeds':[11,29,47],'tenants':20,'k':5,'dimensions':128,'repeats':3,'ef':100,'flat_search_cutoff':0,'image':'typesense/typesense:30.2','port':8112,'cpu_limit':2,'memory_limit':'2g','phases':['current','stale_revocation','applied_revocation'],'scope':'tenant membership and document active state; vector-only retrieval; serial warm local HTTP'}
class APIError(RuntimeError):
 def __init__(self,status,message):super().__init__(message);self.status=status
class TS:
 def __init__(self,key):self.key=key;self.conn=http.client.HTTPConnection('127.0.0.1',CONFIG['port'],timeout=30)
 def call(self,method,path,body=None,key=None,raw=False):
  data=body.encode() if raw else (json.dumps(body,separators=(',',':')).encode() if body is not None else None)
  self.conn.request(method,path,body=data,headers={'X-TYPESENSE-API-KEY':key or self.key,'Content-Type':'text/plain' if raw else 'application/json'})
  res=self.conn.getresponse();buf=res.read()
  if res.status>=400:raise APIError(res.status,f'{method} {path}: HTTP {res.status}: {buf[:160]!r}')
  return buf.decode() if raw else json.loads(buf)
 def createkey(self,collections):return self.call('POST','/keys',{'description':'isolated study runtime','actions':['documents:search'],'collections':collections})['value']
def scoped(parent,params):
 msg=json.dumps(params,separators=(',',':'));digest=base64.b64encode(hmac.new(parent.encode(),msg.encode(),hashlib.sha256).digest()).decode()
 return base64.b64encode((digest+parent[:4]+msg).encode()).decode()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def text_record(r):
 # Deliberately omit account/case/tenant/document identifiers and oracle labels.
 allowed=['kind','title','statement','catalog_kind','skus','eligible_skus','eligible_plans','segment','lines','amount','currency','credit_amount','asserted_net','line_prices']
 return ' '.join(f'{k} {json.dumps(r[k],sort_keys=True)}' for k in allowed if k in r)
def reference(scores,mask,k):
 eligible=np.flatnonzero(mask);n=min(k,len(eligible))
 if not n:return set(),set(),0
 vals=scores[eligible];cut=np.sort(vals)[-n];strict=set(eligible[vals>cut+1e-5].tolist());ties=set(eligible[np.abs(vals-cut)<=1e-5].tolist())
 return strict,ties,n
def main():
 source=json.loads(SOURCE.read_text());records=source['records'];cases=source['cases'];accounts=sorted({r['account_id'] for r in records})
 texts=[text_record(r) for r in records];tf=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True);sparse=tf.fit_transform(texts)
 svd=TruncatedSVD(n_components=CONFIG['dimensions'],random_state=71);vectors=normalize(svd.fit_transform(sparse)).astype('float32');assert np.all(np.linalg.norm(vectors,axis=1)>.99)
 idx={r['id']:i for i,r in enumerate(records)}
 support={r['account_id']:i for i,r in enumerate(records) if r['kind']=='support'}
 test=[c for c in cases if c['split']=='test'];dev=[c for c in cases if c['split']=='development']
 queries=[{'case_id':c['case_id'],'account_id':c['account_id'],'source_index':support[c['account_id']]} for c in test]
 np.savez_compressed(OUT/'vectors.npz',documents=vectors,queries=vectors[[q['source_index'] for q in queries]])
 design={'config':CONFIG,'source_sha256':sha(SOURCE),'runner_sha256':sha(Path(__file__)),'source_counts':{'records':len(records),'accounts':len(accounts),'development':len(dev),'test_queries':len(test)},'vectorizer':'TF-IDF unigram/bigram + rank-128 truncated SVD + L2 normalization; not neural embeddings','vocabulary':len(tf.vocabulary_),'projection_seed':71,'explained_variance_ratio_sum':float(svd.explained_variance_ratio_.sum()),'frozen_before_heldout_execution':True,'ground_truth':'Separate in-memory tenant assignment + active-state ledger; exact NumPy cosine reference; tie aware tolerance 1e-5','revocation':'Per query choose its initially authorized highest cosine-scored document; revoke the union. Stale phase intentionally does not update indexes. Applied phase updates both shared and tenant copies and waits for acknowledgements. No concurrent queries.'}
 (OUT/'design.json').write_text(json.dumps(design,indent=2)+'\n')
 rootkey=secrets.token_hex(24);name='tenant-scope-study-'+secrets.token_hex(4);ts=TS(rootkey);runtime=[];probes=[];updates=[];rows=[]
 before=subprocess.check_output(['docker','ps','--format','{{.ID}} {{.Names}}']).decode().splitlines()
 command=['docker','run','-d','--rm','--name',name,'--cpus','2','--memory','2g','-p',f'127.0.0.1:{CONFIG["port"]}:8108','--tmpfs','/data:rw,size=512m','-e',f'TYPESENSE_API_KEY={rootkey}',CONFIG['image'],'--data-dir=/data','--api-port=8108','--enable-cors=false']
 subprocess.run(command,check=True,capture_output=True)
 try:
  for attempt in range(50):
   try:
    if ts.call('GET','/health')['ok']:break
   except (OSError,http.client.HTTPException,RuntimeError):ts.conn.close();ts=TS(rootkey);time.sleep(.2)
  else:raise RuntimeError('isolated Typesense failed to become healthy')
  version=ts.call('GET','/debug')
  for seed in CONFIG['seeds']:
   rng=random.Random(seed);shuffled=accounts.copy();rng.shuffle(shuffled);assignment={a:f't{j%CONFIG["tenants"]:02}' for j,a in enumerate(shuffled)};tenant=np.array([assignment[r['account_id']] for r in records]);active=np.ones(len(records),dtype=bool)
   prefix=f'study_s{seed}_';shared=prefix+'shared';cols=[shared]+[prefix+t for t in sorted(set(tenant))]
   docs=[{'id':str(i),'tenant':tenant[i],'active':True,'text':texts[i],'vec':vectors[i].tolist()} for i in range(len(records))]
   for col in cols:
    ts.call('POST','/collections',{'name':col,'fields':[{'name':'tenant','type':'string','facet':True},{'name':'active','type':'bool','facet':True},{'name':'text','type':'string'},{'name':'vec','type':'float[]','num_dim':CONFIG['dimensions'],'hnsw_params':{'M':16,'ef_construction':200}}]})
    selected=docs if col==shared else [d for d in docs if prefix+d['tenant']==col]
    raw=ts.call('POST',f'/collections/{col}/documents/import?action=create','\n'.join(json.dumps(d) for d in selected),raw=True)
    assert all(json.loads(line)['success'] for line in raw.splitlines())
   parent=ts.createkey([shared]);a0=ts.createkey([shared]);keys={}
   for t in sorted(set(tenant)):
    keys[('A1',t)]=scoped(parent,{'filter_by':f'tenant:={t} && active:=true','exclude_fields':'vec,text'})
    p=ts.createkey([prefix+t]);keys[('A2',t)]=scoped(p,{'filter_by':'active:=true','exclude_fields':'vec,text'})
   def search(arm,q,override=None,col_override=None):
    t=assignment[q['account_id']];col=prefix+t if arm=='A2' else shared;key=a0 if arm=='A0' else keys[(arm,t)];v=vectors[q['source_index']]
    params={'collection':col_override or col,'q':'*','vector_query':'vec:(['+','.join(format(float(x),'.8g') for x in v)+f'],k:5,ef:100,flat_search_cutoff:0)','per_page':5,'exclude_fields':'vec,text'}
    if override is not None:params['filter_by']=override
    start=time.perf_counter_ns()
    try:res=ts.call('POST','/multi_search',{'searches':[params]},key=key)
    except APIError as error:
     if col_override is not None and error.status in [401,403]:return {'code':error.status,'hits':[]},(time.perf_counter_ns()-start)/1e6
     raise
    elapsed=(time.perf_counter_ns()-start)/1e6
    result=res['results'][0]
    return result,elapsed
   # Pilot checks never contribute held-out results. Test accounts used only after frozen design.
   pilot={'account_id':dev[0]['account_id'],'source_index':support[dev[0]['account_id']]}
   for arm in ['A0','A1','A2']:
    result,_=search(arm,pilot);assert 'hits' in result,(arm,result)
   # Real scoped-key bypass probes (missing filter, opposing tenant, and cross-collection).
   for q in queries:
    t=assignment[q['account_id']];other=next(v for v in sorted(set(tenant)) if v!=t)
    for arm in ['A1','A2']:
     for probe,override,colover in [('omitted_filter',None,None),('opposing_filter',f'tenant:={other}',None),('other_collection',None,prefix+other)]:
      result,_=search(arm,q,override,colover);hits=result.get('hits',[]);leaks=[int(h['document']['id']) for h in hits if tenant[int(h['document']['id'])]!=t]
      probes.append({'seed':seed,'arm':arm,'probe':probe,'case_id':q['case_id'],'http_error':result.get('code'),'hits':len(hits),'unauthorized_hits':len(leaks)})
   scores=[vectors@vectors[q['source_index']] for q in queries]
   revoked=set()
   for q,s in zip(queries,scores):
    eligible=np.flatnonzero(tenant==assignment[q['account_id']]);revoked.add(int(eligible[np.argmax(s[eligible])]))
   # Matched phases separate index-state freshness from tenant membership.
   for phase in CONFIG['phases']:
    if phase=='stale_revocation':active[list(revoked)]=False
    if phase=='applied_revocation':
     begin=time.perf_counter_ns()
     for col in cols:
      update=[{'id':str(i),'active':False} for i in sorted(revoked) if col==shared or col==prefix+tenant[i]]
      if update:
       raw=ts.call('POST',f'/collections/{col}/documents/import?action=update','\n'.join(json.dumps(d) for d in update),raw=True)
       assert all(json.loads(line)['success'] for line in raw.splitlines())
     updates.append({'seed':seed,'revoked_documents':len(revoked),'acknowledged_update_batch_ms':(time.perf_counter_ns()-begin)/1e6})
    # Warm each configuration equally; warmups omitted from measurements.
    for arm in ['A0','A1','A2']:
     for q in queries[:5]:search(arm,q)
    order=[(repeat,j,arm) for repeat in range(CONFIG['repeats']) for j in range(len(queries)) for arm in ['A0','A1','A2']];rng.shuffle(order)
    for repeat,j,arm in order:
     q=queries[j];t=assignment[q['account_id']];mask=(tenant==t)&active;strict,ties,n=reference(scores[j],mask,CONFIG['k']);result,ms=search(arm,q)
     if 'hits' not in result:raise RuntimeError(f'search failed: {arm} {result}')
     returned=[int(h['document']['id']) for h in result['hits']];ids=set(returned);unauth=[i for i in returned if not mask[i]];cross=[i for i in returned if tenant[i]!=t];rev=[i for i in returned if tenant[i]==t and not active[i]]
     recall=(len(ids&strict)+min(len(ids&ties),n-len(strict)))/n if n else None
     rows.append({'seed':seed,'phase':phase,'arm':arm,'repeat':repeat,'case_id':q['case_id'],'tenant':t,'latency_ms':ms,'eligible':int(mask.sum()),'returned':returned,'cross_tenant_hits':len(cross),'revoked_hits':len(rev),'unauthorized_hits':len(unauth),'recall_at_5':recall,'false_denial':bool(n and not any(mask[i] for i in returned))})
    print(f'Completed seed {seed} phase {phase}',flush=True)
   for col in cols:ts.call('DELETE',f'/collections/{col}')
  (OUT/'requests.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));(OUT/'probes.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in probes));(OUT/'updates.json').write_text(json.dumps(updates,indent=2)+'\n')
  env={'platform':platform.platform(),'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,'typesense_debug':version,'cpu_info':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string']).decode().strip(),'config':CONFIG,'existing_container_names_before':before,'docker_image_inspect':json.loads(subprocess.check_output(['docker','image','inspect',CONFIG['image'],'--format','{{json .RepoDigests}}']))}
  (OUT/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
 finally:subprocess.run(['docker','stop',name],capture_output=True)
 print('Measured requests:',len(rows),'Scoped-key probes:',len(probes),flush=True)
if __name__=='__main__':main()
