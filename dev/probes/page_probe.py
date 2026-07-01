import json, time, urllib.request, websocket, sys
BASE='http://127.0.0.1:9222'
url=sys.argv[1]
req=urllib.request.Request(BASE+'/json/new', method='PUT')
target=json.load(urllib.request.urlopen(req))
ws=websocket.create_connection(target['webSocketDebuggerUrl'], timeout=10)
mid=0
def call(m,p=None):
 global mid; mid+=1; ws.send(json.dumps({'id':mid,'method':m,'params':p or {}}))
 while True:
  msg=json.loads(ws.recv())
  if msg.get('id')==mid: return msg
call('Page.enable'); call('Runtime.enable'); call('Page.navigate', {'url':url}); time.sleep(4)
expr=r'''(()=>({title:document.title,url:location.href,text:document.body.innerText.slice(0,500),ids:[...document.querySelectorAll('[data-listing-id], .vehicle-card')].filter(el=>el.innerText&&/(Used|New|Certified)\s+(19\d{2}|20\d{2})/.test(el.innerText)).map(el=>({id:el.getAttribute('data-listing-id'),txt:el.innerText.slice(0,120)})),links:[...document.querySelectorAll('a')].map(a=>({txt:a.innerText,aria:a.getAttribute('aria-label'),href:a.href})).filter(x=>/page|next|2|3/i.test((x.txt||'')+' '+(x.aria||''))).slice(0,20)}))()'''
res=call('Runtime.evaluate', {'expression':expr,'returnByValue':True})
print(json.dumps(res.get('result',{}).get('result',{}).get('value'),indent=2)[:12000])
ws.close()
