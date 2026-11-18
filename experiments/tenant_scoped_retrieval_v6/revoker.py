"""Revoker as a separate process: spins to an absolute target time, then commits a revocation at the authority."""
import sys,json,time,socket,threading,http.client
from http.server import HTTPServer,BaseHTTPRequestHandler
sys.setswitchinterval(0.0002)
def now():return time.clock_gettime_ns(time.CLOCK_UPTIME_RAW)
AUTH=int(sys.argv[2]);job={};done=threading.Event();go=threading.Event()
def worker():
 conn=http.client.HTTPConnection('127.0.0.1',AUTH,timeout=10)
 while True:
  go.wait();go.clear();target=job['target_ns']
  while now()<target:
   if target-now()>2_000_000:time.sleep(.001)
  fired=now();conn.request('POST','/policy',body=json.dumps({'action':'commit','ids':job['ids']}),headers={'Content-Type':'application/json'});r=conn.getresponse();obj=json.loads(r.read())
  job.update({'fired_ns':fired,'commit_ns':obj['commit_ns']});done.set()
class Handler(BaseHTTPRequestHandler):
 protocol_version='HTTP/1.1'
 def setup(self):super().setup();self.connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
 def log_message(self,*a):pass
 def do_POST(self):
  q=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  if q['action']=='schedule':job.clear();job.update(q);done.clear();go.set();obj={'ok':True}
  else:done.wait(10);obj={k:job.get(k) for k in ['target_ns','fired_ns','commit_ns']}
  raw=json.dumps(obj).encode();self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
if __name__=='__main__':
 threading.Thread(target=worker,daemon=True).start();server=HTTPServer(('127.0.0.1',int(sys.argv[1])),Handler);print('ready',flush=True);server.serve_forever()
