import json, time, urllib.request, websocket, sys
base='http://127.0.0.1:9222'; url=sys.argv[1] if len(sys.argv)>1 else 'https://www.cars.com/vehicledetail/74edaab7-1711-4e85-a78d-5c6022ba458a/'
req=urllib.request.Request(base+'/json/new', method='PUT'); target=json.load(urllib.request.urlopen(req)); ws=websocket.create_connection(target['webSocketDebuggerUrl'], timeout=10)
msgid=0
def call(m,p=None):
 global msgid; msgid+=1; ws.send(json.dumps({'id':msgid,'method':m,'params':p or {}}))
 while True:
  msg=json.loads(ws.recv())
  if msg.get('id')==msgid: return msg
call('Page.enable'); call('Runtime.enable'); call('Page.navigate', {'url':url}); time.sleep(5)
expr=r'''(()=>[...document.scripts].map(s=>s.type+' '+s.innerText.slice(0,2000)).filter(t=>/Vehicle|Acura|TSX|schema|application\/ld/.test(t)).slice(0,10))()'''
res=call('Runtime.evaluate', {'expression':expr,'returnByValue':True})
print(json.dumps(res.get('result',{}).get('result',{}).get('value'), indent=2)[:12000])
ws.close()
