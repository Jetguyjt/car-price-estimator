#!/usr/bin/env python3
"""Careful Cars.com collector using a real Chrome browser via DevTools.
Records only fields visible in loaded pages/card text; leaves unknowns blank.
"""
import csv, json, os, re, time, urllib.request
from datetime import datetime, timezone
from urllib.parse import quote
import websocket

import paths

CSV_PATH = os.environ.get('CSV_PATH', str(paths.RAW_CSV))
BASE = os.environ.get('CDP_BASE', 'http://127.0.0.1:9222')
COLUMNS = ['listing_key','zip','region','scraped_at','listing_url','title','year','make','model','trim','price','mileage','body_style','stock_type','stock_number','exterior_color','interior_color','fuel_type_raw','drivetrain_raw','transmission_raw','engine_raw','mpg_text','mpg_city','mpg_highway','mpg_combined','mpg_equivalent','title_status_raw','no_accidents_raw','owner_count','cpo_raw','enriched_at']
PRICE_BANDS = [(0,5000),(5000,10000),(10000,15000),(15000,20000),(20000,25000),(25000,30000),(30000,40000),(40000,50000),(50000,70000),(70000,100000),(100000,150000),(150000,None)]
if os.environ.get('NO_PRICE_BANDS','').lower() in ('1','true','yes'):
    PRICE_BANDS = [(None, None)]
STOCKS = [s.strip() for s in os.environ.get('STOCKS','used,new,cpo').split(',') if s.strip()]
MAX_NEW = int(os.environ.get('MAX_NEW','25'))
MAX_PAGES_PER_SEARCH = int(os.environ.get('MAX_PAGES_PER_SEARCH','2'))
PAGE_PAUSE = float(os.environ.get('PAGE_PAUSE','4'))
DETAIL_PAUSE = float(os.environ.get('DETAIL_PAUSE','3.5'))
MAX_PER_EXACT_TITLE = int(os.environ.get('MAX_PER_EXACT_TITLE','3'))
MAX_PER_SELLER_RUN = int(os.environ.get('MAX_PER_SELLER_RUN','12'))
MAX_EMPTY_PAGES = int(os.environ.get('MAX_EMPTY_PAGES','5'))
MAKE_FILTERS = [m.strip() for m in os.environ.get('MAKE_FILTERS','').split(',') if m.strip()]
YEAR_FILTERS = [y.strip() for y in os.environ.get('YEAR_FILTERS','').split(',') if y.strip()]
ZIP_CODE = os.environ.get('ZIP_CODE', '98103').strip() or '98103'
REGION_NAME = os.environ.get('REGION_NAME', '').strip()
MAX_DISTANCE = os.environ.get('MAX_DISTANCE', '9999').strip() or '9999'

MAKES = ['Aston Martin','Alfa Romeo','Land Rover','Mercedes-Benz','Rolls-Royce','Harley-Davidson','International','Acura','Audi','BMW','Buick','Cadillac','Chevrolet','Chrysler','Dodge','Ferrari','FIAT','Ford','Genesis','GMC','Honda','Hyundai','INFINITI','Jaguar','Jeep','Kia','Lamborghini','Lexus','Lincoln','Lucid','Maserati','Mazda','McLaren','Mercury','MINI','Mitsubishi','Nissan','Polestar','Pontiac','Porsche','Ram','Rivian','Saab','Saturn','Scion','Subaru','Tesla','Toyota','Volkswagen','Volvo']

class CDP:
    def __init__(self):
        # Reuse an existing Chrome tab instead of opening a new one. This keeps the
        # visible browser tidy and preserves the user's current Cars.com session.
        targets = json.load(urllib.request.urlopen(BASE + '/json', timeout=10))
        pages = [t for t in targets if t.get('type') == 'page' and t.get('webSocketDebuggerUrl')]
        if not pages:
            raise RuntimeError('No existing Chrome page target found; open one tab before starting the collector')
        cars_pages = [t for t in pages if 'cars.com' in (t.get('url') or '')]
        target = (cars_pages or pages)[0]
        self.ws = websocket.create_connection(target['webSocketDebuggerUrl'], timeout=60)
        self.msgid = 0
        self.call('Page.enable'); self.call('Runtime.enable')
    def call(self, method, params=None):
        self.msgid += 1
        self.ws.send(json.dumps({'id': self.msgid, 'method': method, 'params': params or {}}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get('id') == self.msgid:
                return msg
    def navigate(self, url, pause):
        self.call('Page.navigate', {'url': url})
        time.sleep(pause)
    def eval(self, expression):
        res = self.call('Runtime.evaluate', {'expression': expression, 'returnByValue': True, 'awaitPromise': True})
        if 'exceptionDetails' in res.get('result', {}):
            raise RuntimeError(res['result']['exceptionDetails'])
        return res.get('result', {}).get('result', {}).get('value')
    def close(self):
        try: self.ws.close()
        except Exception: pass

def iso_now(): return datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00','Z')
def key_from_url(u):
    m = re.search(r'/vehicledetail/([0-9a-fA-F-]+)/', u or '')
    return m.group(1) if m else ''
def clean_num(s):
    if not s: return ''
    m = re.search(r'\$?([0-9][0-9,]*)', s)
    return m.group(1).replace(',','') if m else ''

def parse_title(t):
    # Visible title examples: Used 2010 Acura TSX 3.5; Certified 2022 Toyota RAV4 XLE
    out = {'year':'','make':'','model':'','trim':'','stock_type':''}
    words = t.split()
    if words and words[0] in ('Used','New','Certified'):
        out['stock_type'] = 'Certified' if words[0]=='Certified' else words[0]
    m = re.search(r'\b(19\d{2}|20\d{2})\b', t)
    if not m: return out
    out['year'] = m.group(1)
    rest = t[m.end():].strip()
    for make in sorted(MAKES, key=len, reverse=True):
        if rest == make or rest.startswith(make + ' '):
            out['make'] = make
            rem = rest[len(make):].strip()
            # Conservative: first visible token after make as model; rest as trim.
            # If model boundary is ambiguous, preserve full string in title as source of truth.
            if rem:
                parts = rem.split()
                out['model'] = parts[0]
                out['trim'] = ' '.join(parts[1:])
            return out
    return out

def parse_card(card):
    txt = card.get('txt','')
    vehicle_details = {}
    if card.get('vehicleDetails'):
        try: vehicle_details = json.loads(card.get('vehicleDetails') or '{}')
        except Exception: vehicle_details = {}
    lines = [l.strip() for l in txt.split('\n') if l.strip()]
    title = ''
    for l in lines:
        if re.match(r'^(Used|New|Certified)\s+(19\d{2}|20\d{2})\b', l):
            title = l; break
    price = next((clean_num(l) for l in lines if l.startswith('$') and 'mo' not in l.lower()), '')
    mileage = ''
    for l in lines:
        if ' mi' in l.lower():
            mileage = clean_num(l); break
    link = (card.get('links') or [''])[0]
    d = parse_title(title)
    seller = vehicle_details.get('seller') or {}
    return {
        'listing_key': card.get('id') or key_from_url(link),
        'listing_url': link,
        'title': title,
        'price': price,
        'mileage': mileage,
        'body_style': vehicle_details.get('bodyStyle') or '',
        '_seller_id': seller.get('customerId') or seller.get('zip') or '',
        '_isa_context': vehicle_details.get('isaContext') or '',
        **d,
    }

def extract_search(cdp):
    expr = r"""
    (() => {
      const bodyText = document.body ? (document.body.innerText || '') : '';
      const byId=[...document.querySelectorAll('[data-listing-id], .vehicle-card')]
        .filter(el => el && (el.innerText || '') && /(Used|New|Certified)\s+(19\d{2}|20\d{2})/.test(el.innerText || ''));
      const byLink=[];
      for (const a of [...document.querySelectorAll('a[href*="/vehicledetail/"]')]) {
        const href=a.href || '';
        if (href.includes('attribution_type=carousel')) continue;
        if (!/(Used|New|Certified)\s+(19\d{2}|20\d{2})/.test(a.innerText || '')) continue;
        let el=a;
        for (let depth=0; depth<4 && el; depth++, el=el.parentElement) {
          const t=(el.innerText || '');
          if (/\$[0-9,]+/.test(t) && /\bmi\.?\b/i.test(t) && /(Used|New|Certified)\s+(19\d{2}|20\d{2})/.test(t)) { byLink.push(el); break; }
        }
      }
      const seen=new Set();
      const cards=[];
      for (const el of byId.concat(byLink)) {
        const link=[...el.querySelectorAll('a[href*="/vehicledetail/"]')].map(a=>a.href)[0] || '';
        const id=el.getAttribute('data-listing-id') || ((link.match(/vehicledetail\/([^/]+)\//)||[])[1] || '');
        const key=id || link;
        if (!key || seen.has(key)) continue;
        seen.add(key); cards.push(el);
      }
      return {title:document.title || '', url:location.href, body:bodyText.slice(0,1200),
        cards:cards.map(el=>({
          id:el.getAttribute('data-listing-id')||'',
          txt:(el.innerText || '').slice(0,1000),
          vehicleDetails:el.getAttribute('data-vehicle-details')||'',
          links:[...el.querySelectorAll('a[href*="/vehicledetail/"]')].map(a=>a.href)
        })),
        nextUrl: ''};
    })()
    """
    return cdp.eval(expr)

def parse_detail_text(text, title):
    vals = {k:'' for k in ['stock_number','exterior_color','interior_color','fuel_type_raw','drivetrain_raw','transmission_raw','engine_raw','mpg_text','mpg_city','mpg_highway','mpg_combined','mpg_equivalent','title_status_raw','no_accidents_raw','owner_count','cpo_raw']}
    # Only mark CPO true when the visible listing/detail title itself says Certified.
    # Generic page/footer text can mention "Certified Pre-Owned" even for non-CPO listings.
    vals['cpo_raw'] = 'true' if title.startswith('Certified ') else ('false' if title else '')
    m = re.search(r'VIN:\s*[^/\n]+/\s*Stock\s*#:\s*([^\n]*)', text)
    if m: vals['stock_number'] = m.group(1).strip()
    patterns = [
        ('exterior_color', r'([^\n]+) exterior color'),
        ('interior_color', r'([^\n]+) interior color'),
        ('fuel_type_raw', r'([^\n]+) fuel type'),
        ('drivetrain_raw', r'([^\n]+) drivetrain'),
        ('transmission_raw', r'([^\n]+) transmission'),
        ('engine_raw', r'([^\n]+) engine'),
    ]
    for key, pat in patterns:
        mm = re.search(pat, text, re.I)
        if mm: vals[key] = mm.group(1).strip()
    # MPG often appears as "22 City / 31 Hwy" or similar when visible.
    mm = re.search(r'((?:\d+\s*)?(?:City|city)[^\n]{0,40}(?:Hwy|Highway|highway)[^\n]{0,40})', text)
    if mm: vals['mpg_text'] = mm.group(1).strip()
    mm = re.search(r'(No accidents reported|\d+ accidents? reported)', text, re.I)
    if mm: vals['no_accidents_raw'] = mm.group(1).strip()
    mm = re.search(r'(Clean title|Salvage title|Rebuilt title|Lemon title|Title issue[^\n]*)', text, re.I)
    if mm: vals['title_status_raw'] = mm.group(1).strip()
    mm = re.search(r'(\d+)\s+owners?', text, re.I)
    if mm: vals['owner_count'] = mm.group(1)
    return vals

def extract_detail(cdp, url, title):
    cdp.navigate(url, DETAIL_PAUSE)
    expr = r"""
    (() => {
      const bodyText = document.body ? (document.body.innerText || '') : '';
      return {title:document.title || '', url:location.href, text:bodyText.slice(0,9000)};
    })()
    """
    data = cdp.eval(expr)
    if 'Sorry, you have been blocked' in data.get('text','') or 'Attention Required' in data.get('title',''):
        return None, 'blocked'
    vals = parse_detail_text(data.get('text',''), title)
    final_url = data.get('url') or url
    return (vals, final_url), None

def load_existing():
    keys=set(); title_counts={}
    paths_to_read=[]
    for p in (os.environ.get('EXISTING_CSVS','').split(':') if os.environ.get('EXISTING_CSVS') else []):
        if p: paths_to_read.append(p)
    paths_to_read.append(CSV_PATH)
    for path in paths_to_read:
        if os.path.exists(path):
            with open(path, newline='') as f:
                for r in csv.DictReader(f):
                    if r.get('listing_key'): keys.add(r['listing_key'])
                    if r.get('title'):
                        tk = r['title'].strip().lower()
                        title_counts[tk] = title_counts.get(tk, 0) + 1
    if not os.path.exists(CSV_PATH):
        os.makedirs(os.path.dirname(CSV_PATH) or '.', exist_ok=True)
        with open(CSV_PATH, 'w', newline='') as f: csv.DictWriter(f, fieldnames=COLUMNS).writeheader()
    return keys, title_counts

def append_rows(rows):
    with open(CSV_PATH, 'a', newline='') as f:
        w=csv.DictWriter(f, fieldnames=COLUMNS)
        for r in rows: w.writerow({c:r.get(c,'') for c in COLUMNS})

def search_url(stock, lo, hi, page=1, make='', year=''):
    u = f'https://www.cars.com/shopping/results/?zip={ZIP_CODE}&maximum_distance={MAX_DISTANCE}&sort=listed_at_desc&stock_type={stock}'
    if lo is not None: u += f'&list_price_min={lo}'
    if hi is not None: u += f'&list_price_max={hi}'
    if make: u += f'&makes[]={make}'
    if year: u += f'&year_min={year}&year_max={year}'
    if page and page > 1: u += f'&page={page}'
    return u


class CollectionBlocked(RuntimeError):
    """Stop the run when the marketplace returns a block page."""


def main():
    existing, title_counts = load_existing(); added=0; pending=[]; covered=[]; blocked=[]; seller_counts={}; skipped_quality=0
    cdp = CDP()
    try:
      for stock in STOCKS:
        for make in (MAKE_FILTERS or ['']):
          for year in (YEAR_FILTERS or ['']):
            for lo,hi in PRICE_BANDS:
              band = f'{stock}:{make+":" if make else ""}{year+":" if year else ""}{lo}-{hi if hi is not None else ""}'
              if added >= MAX_NEW: break
              page=1
              empty_pages = 0
              while page <= MAX_PAGES_PER_SEARCH and added < MAX_NEW:
                url = search_url(stock,lo,hi,page,make,year)
                print(f'SEARCH {band} page {page}: {url}', flush=True)
                try:
                    cdp.navigate(url, PAGE_PAUSE)
                    s = extract_search(cdp)
                    if not s.get('cards'):
                        time.sleep(max(2.5, PAGE_PAUSE))
                        s = extract_search(cdp)
                except Exception as e:
                    print(f'RECOVERABLE search error {band} p{page}: {type(e).__name__}: {e}', flush=True)
                    try: cdp.close()
                    except Exception: pass
                    time.sleep(2)
                    cdp = CDP()
                    page += 1
                    continue
                body = s.get('body','') or ''
                if 'Sorry, you have been blocked' in body or 'Attention Required' in s.get('title',''):
                    print(f'BLOCKED search {band} page {page}', flush=True)
                    blocked.append(f'{band} p{page}')
                    raise CollectionBlocked(f'search page {band} p{page}')
                raw_cards = []
                for c in s.get('cards',[]):
                    pc = parse_card(c)
                    if stock == 'used' and pc.get('title','').startswith('New '):
                        skipped_quality += 1
                        continue
                    if stock == 'new' and not pc.get('title','').startswith('New '):
                        skipped_quality += 1
                        continue
                    if stock == 'cpo' and not pc.get('title','').startswith('Certified '):
                        skipped_quality += 1
                        continue
                    if pc.get('listing_key') and pc.get('listing_url'):
                        raw_cards.append(pc)
                # Avoid repeatedly collecting near-identical cards from one dealer at
                # the top of a price band.
                # Keep at most one exact title and two listings from the same seller per
                # search page, and ignore explicitly non-standard ad contexts if visible.
                cards = []
                seen_titles = set(); page_seller_counts = {}
                for pc in raw_cards:
                    if pc.get('_isa_context') and pc.get('_isa_context') != 'STANDARD':
                        skipped_quality += 1
                        continue
                    title_key = pc.get('title','').lower()
                    seller_key = pc.get('_seller_id','')
                    if title_key in seen_titles:
                        skipped_quality += 1
                        continue
                    if seller_key and page_seller_counts.get(seller_key, 0) >= 2:
                        skipped_quality += 1
                        continue
                    if title_key and title_counts.get(title_key, 0) >= MAX_PER_EXACT_TITLE:
                        skipped_quality += 1
                        continue
                    if seller_key and seller_counts.get(seller_key, 0) >= MAX_PER_SELLER_RUN:
                        skipped_quality += 1
                        continue
                    seen_titles.add(title_key)
                    if seller_key:
                        page_seller_counts[seller_key] = page_seller_counts.get(seller_key, 0) + 1
                    cards.append(pc)
                print(f' visible_cards={len(raw_cards)} diverse_cards={len(cards)} title={s.get("title")!r}', flush=True)
                covered.append(f'{band} p{page} cards={len(cards)} raw={len(raw_cards)}')
                if not raw_cards:
                    empty_pages += 1
                    if empty_pages >= MAX_EMPTY_PAGES:
                        print(f'END band {band}: {empty_pages} consecutive empty pages at p{page}', flush=True)
                        break
                else:
                    empty_pages = 0
                for card in cards:
                    if added >= MAX_NEW: break
                    if card['listing_key'] in existing: continue
                    scraped_at = iso_now()
                    try:
                        detail, err = extract_detail(cdp, card['listing_url'], card.get('title',''))
                    except Exception as e:
                        print(f'RECOVERABLE detail error {card["listing_key"]}: {type(e).__name__}: {e}', flush=True)
                        try: cdp.close()
                        except Exception: pass
                        time.sleep(2)
                        cdp = CDP()
                        continue
                    if err == 'blocked':
                        print(f'BLOCKED detail {card["listing_key"]}', flush=True)
                        blocked.append(f'detail:{card["listing_key"]}')
                        raise CollectionBlocked(f'detail page {card["listing_key"]}')
                    vals, final_url = detail
                    row = {c:'' for c in COLUMNS}
                    row.update(card); row.update(vals)
                    row['listing_key'] = key_from_url(final_url) or card['listing_key']
                    row['listing_url'] = re.sub(r'\?.*$', '/', final_url) if '/vehicledetail/' in final_url else card['listing_url']
                    row['zip']=ZIP_CODE; row['region']=REGION_NAME; row['scraped_at']=scraped_at; row['enriched_at']=iso_now()
                    pending.append(row); existing.add(row['listing_key']); added += 1
                    if row.get('title'):
                        tk = row['title'].strip().lower()
                        title_counts[tk] = title_counts.get(tk, 0) + 1
                    seller_key = card.get('_seller_id','')
                    if seller_key:
                        seller_counts[seller_key] = seller_counts.get(seller_key, 0) + 1
                    print(f' added {added}: {row["listing_key"]} {row["title"]}', flush=True)
                    if len(pending) >= 20:
                        append_rows(pending); print(f'SAVED {len(pending)} rows', flush=True); pending.clear()
                page += 1
        if added >= MAX_NEW: break
      if pending:
        append_rows(pending); print(f'SAVED {len(pending)} rows', flush=True); pending.clear()
    except CollectionBlocked as e:
      if pending:
        append_rows(pending); print(f'SAVED {len(pending)} rows before stop', flush=True); pending.clear()
      print(f'STOPPED after block page: {e}', flush=True)
    except Exception as e:
      if pending:
        append_rows(pending); print(f'SAVED {len(pending)} rows after exception', flush=True); pending.clear()
      print(f'ERROR {type(e).__name__}: {e}', flush=True)
      raise
    finally:
      cdp.close()
    print('SUMMARY_JSON=' + json.dumps({'added':added,'covered':covered,'blocked':blocked,'skipped_quality':skipped_quality,'csv':CSV_PATH}, ensure_ascii=False))
if __name__ == '__main__': main()
