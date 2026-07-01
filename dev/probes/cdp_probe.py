import json, time, urllib.request, websocket
base='http://127.0.0.1:9222'
url='https://www.cars.com/shopping/results/?zip=98103&maximum_distance=9999&sort=listed_at_desc&stock_type=used&list_price_min=0&list_price_max=5000'
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
        if msg.get('id')==msgid:
            return msg
call('Page.enable'); call('Runtime.enable')
call('Page.navigate', {'url': url})
time.sleep(10)
expr=r"""
(() => {
 const cardSel='[data-listing-id], .vehicle-card';
 const cards=[...document.querySelectorAll(cardSel)].filter(el=>/vehicledetail|Used|New|Certified|\$/.test(el.innerText));
 return {title:document.title,url:location.href,text:document.body.innerText.slice(0,1500),count:cards.length,cards:cards.slice(0,10).map(el=>({id:el.getAttribute('data-listing-id')||'',txt:el.innerText.slice(0,900),links:[...el.querySelectorAll('a[href*="/vehicledetail/"]')].map(a=>a.href)}))};
})()
"""
res=call('Runtime.evaluate', {'expression':expr,'returnByValue':True})
print(json.dumps(res.get('result',{}).get('result',{}).get('value'), indent=2)[:15000])
ws.close()
