import json, time, urllib.request, websocket, sys
base='http://127.0.0.1:9222'
url=sys.argv[1] if len(sys.argv)>1 else 'https://www.cars.com/vehicledetail/74edaab7-1711-4e85-a78d-5c6022ba458a/'
req=urllib.request.Request(base+'/json/new', method='PUT')
target=json.load(urllib.request.urlopen(req))
ws=websocket.create_connection(target['webSocketDebuggerUrl'], timeout=10)
msgid=0
def call(method, params=None):
    global msgid
    msgid+=1
    ws.send(json.dumps({'id':msgid,'method':method,'params':params or {}}))
    while True:
        msg=json.loads(ws.recv())
        if msg.get('id')==msgid: return msg
call('Page.enable'); call('Runtime.enable'); call('Page.navigate', {'url': url}); time.sleep(8)
expr=r"""
(() => {
 const pairs=[];
 document.querySelectorAll('dt').forEach(dt=>{let dd=dt.nextElementSibling; if(dd) pairs.push([dt.innerText.trim(),dd.innerText.trim()]);});
 const specs=[];
 document.querySelectorAll('[class*=basics] li, [class*=spec] li, [data-qa*=spec], [data-qa*=basics]').forEach(el=>{let t=el.innerText.trim(); if(t) specs.push(t)});
 return {title:document.title,url:location.href,text:document.body.innerText.slice(0,6000),pairs, specs:specs.slice(0,100)};
})()
"""
res=call('Runtime.evaluate', {'expression':expr,'returnByValue':True})
print(json.dumps(res.get('result',{}).get('result',{}).get('value'), indent=2)[:20000])
ws.close()
