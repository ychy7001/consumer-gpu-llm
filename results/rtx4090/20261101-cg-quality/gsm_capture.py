import argparse,json,re,time,urllib.request
from concurrent.futures import ThreadPoolExecutor
p=argparse.ArgumentParser();p.add_argument('url');p.add_argument('model');p.add_argument('data');p.add_argument('out');p.add_argument('--conc',type=int,default=16);a=p.parse_args()
def post(body):
 r=urllib.request.Request(a.url.rstrip('/')+'/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
 with urllib.request.urlopen(r,timeout=600) as x:return json.load(x)
def extract(s):
 m=re.search(r'Final answer:\s*\**\s*\$?(-?[\d,]*\.?\d+)',s)
 if m:return m.group(1).replace(',','')
 z=re.findall(r'-?\d[\d,]*\.?\d*',s.replace('$',''));return z[-1].replace(',','') if z else None
def one(item):
 i,row=item; gold=row['a'].split('####')[-1].strip().replace(',','')
 body={'model':a.model,'max_tokens':768,'temperature':0,'chat_template_kwargs':{'enable_thinking':False},'messages':[{'role':'user','content':row['q']+'\n\nSolve step by step, then give the final answer as \'Final answer: <number>\'.'}]}
 try:
  r=post(body); txt=r['choices'][0]['message'].get('content') or ''; pred=extract(txt)
  try: ok=abs(float(pred)-float(gold))<1e-6
  except: ok=False
  u=r.get('usage',{}); return {'i':i,'ok':ok,'gold':gold,'pred':pred,'text':txt,'completion_tokens':u.get('completion_tokens'),'prompt_tokens':u.get('prompt_tokens'),'error':None}
 except Exception as e:return {'i':i,'ok':False,'gold':gold,'pred':None,'text':'','completion_tokens':None,'prompt_tokens':None,'error':repr(e)}
rows=[json.loads(x) for x in open(a.data)];t=time.time()
with ThreadPoolExecutor(a.conc) as ex: out=list(ex.map(one,enumerate(rows)))
out.sort(key=lambda x:x['i']); result={'n':len(out),'correct':sum(x['ok'] for x in out),'acc':sum(x['ok'] for x in out)/len(out),'errors':sum(x['error'] is not None for x in out),'secs':round(time.time()-t),'items':out}
json.dump(result,open(a.out,'w'),ensure_ascii=False)
print(json.dumps({k:result[k] for k in ('n','correct','acc','errors','secs')}))
