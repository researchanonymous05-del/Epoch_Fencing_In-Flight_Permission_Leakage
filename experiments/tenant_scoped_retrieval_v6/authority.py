"""Policy authority as a separate process: serialized ledger, global and per-tenant epochs, split policy wait, phantom churn."""
import sys,json,time,random,socket,threading
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
def now():return time.clock_gettime_ns(time.CLOCK_UPTIME_RAW)
class State:
 def __init__(self):self.lock=threading.Lock();self.tenants=[];self.revoked=set();self.epoch=0;self.tepoch={};self.churn=None;self.commits=[]
 def bump(self,ids):
  self.epoch+=1
  for t in {self.owner(i) for i in ids}:self.tepoch[t]=self.tepoch.get(t,0)+1
 def owner(self,i):return self.tenants[i] if i<len(self.tenants) else f't{(i-1_000_000)//10000:02}'
S=State()
def churn(rate,stop):
 """Toggle phantom (non-indexed) IDs across tenants at `rate` commits/s: moves epochs without changing eligibility."""
 rng=random.Random(5);gap=1/rate;nxt=now()
 while not stop.is_set():
  nxt+=int(rng.expovariate(1/gap)*1e9);d=(nxt-now())/1e9
  if d>0:time.sleep(d)
  i=1_000_000+rng.randrange(20)*10000+rng.randrange(50)
  with S.lock:
   (S.revoked.discard if i in S.revoked else S.revoked.add)(i);S.bump([i])
class Handler(BaseHTTPRequestHandler):
 protocol_version='HTTP/1.1'
 def setup(self):super().setup();self.connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
 def log_message(self,*a):pass
 def do_POST(self):
  q=json.loads(self.rfile.read(int(self.headers['Content-Length'])));a=q['action'];w=q.get('wait_ms',0)/2000
  if a in ['list','check','fence'] and w:time.sleep(w)  # request half of the policy round trip
  with S.lock:
   snap=now();t=q.get('tenant')
   if a=='setup':S.tenants=q['tenants'];S.revoked=set(q['revoked']);S.epoch=0;S.tepoch={};S.commits=[];obj={}
   elif a=='list':obj={'revoked':[i for i in sorted(S.revoked) if S.owner(i)==t]}
   elif a=='check':obj={'allowed':[i for i in q['ids'] if S.tenants[i]==t and i not in S.revoked]}
   elif a=='fence':
    cur=S.tepoch.get(t,0) if q['scope']=='tenant' else S.epoch
    obj={'allowed':q['ids'],'fast':True} if cur==q['epoch'] else {'allowed':[i for i in q['ids'] if S.tenants[i]==t and i not in S.revoked],'fast':False}
   elif a=='commit':S.revoked|=set(q['ids']);S.bump(q['ids']);S.commits.append(snap);obj={'commit_ns':snap}
   elif a=='restore':S.revoked-=set(q['ids']);S.bump(q['ids']);obj={}
   elif a=='churn':
    if S.churn:S.churn.set();S.churn=None
    if q['rate']:S.churn=threading.Event();threading.Thread(target=churn,args=(q['rate'],S.churn),daemon=True).start()
    obj={}
   elif a=='ping':obj={}
   else:raise ValueError(a)
   obj.update({'epoch':S.epoch,'tenant_epoch':S.tepoch.get(t,0) if t else None,'snapshot_ns':snap})
  if a in ['list','check','fence'] and w:time.sleep(w)  # response half
  raw=json.dumps(obj).encode();self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
if __name__=='__main__':
 server=ThreadingHTTPServer(('127.0.0.1',int(sys.argv[1])),Handler);server.daemon_threads=True;print('ready',flush=True);server.serve_forever()
