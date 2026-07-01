import json, time, urllib.request, websocket
BASE='http://127.0.0.1:9222'
url='https://www.cars.com/shopping/results/?zip=98103&maximum_distance=9999&sort=listed_at_desc&stock_type=new&list_price_min=25000&list_price_max=30000'
req=urllib.request.Request(BASE+'/json/new', method='PUT')
target=json.load(urllib.request.urlopen(req))
ws=websocket.create_connection(target['webSocketDebuggerUrl'], timeout=10)
mid=0
def call(m,p=None):
 global mid; mid+=1; ws.send(json.dumps({'id':mid,'method':m,'params':p or {}}))
 while True:
  msg=json.loads(ws.recv())
  if msg.get('id')==mid: return msg
call('Page.enable'); call('Runtime.enable'); call('Page.navigate', {'url':url}); time.sleep(5)
expr=r'''
(()=>[...document.querySelectorAll('[data-listing-id], .vehicle-card')]
 .filter(el=>el.innerText && /^(?:.|\n)*(Used|New|Certified)\s+(19\d{2}|20\d{2})/.test(el.innerText))
 .map((el,i)=>({i, id:el.getAttribute('data-listing-id'), cls:el.className, attrs:[...el.attributes].map(a=>[a.name,a.value]).filter(x=>/sponsor|ad|listing|data|seller|inventory|promot/i.test(x.join(' '))).slice(0,30), text:el.innerText.slice(0,500), parentCls:el.parentElement&&el.parentElement.className, parentAttrs:el.parentElement?[...el.parentElement.attributes].map(a=>[a.name,a.value]).filter(x=>/sponsor|ad|listing|data|promot/i.test(x.join(' '))).slice(0,20):[]}))
)()
'''
res=call('Runtime.evaluate', {'expression':expr, 'returnByValue':True})
print(json.dumps(res.get('result',{}).get('result',{}).get('value'), indent=2)[:20000])
ws.close()
