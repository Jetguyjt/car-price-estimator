import pandas as pd
import paths
p=str(paths.RAW_CSV)
df=pd.read_csv(p,dtype=str,keep_default_na=False)
for c in ['price','mileage','year']:
    df[c+'_num']=pd.to_numeric(df[c].str.replace(r'[^0-9.]','',regex=True).replace('',pd.NA),errors='coerce')
checks={
 'missing_price': df[df.price_num.isna() | (df.price_num<=0)],
 'new_high_mileage_gt1000': df[(df.stock_type=='New') & (df.mileage_num>1000)],
 'used_price_lt1000': df[(df.stock_type.isin(['Used','Certified'])) & (df.price_num<1000)],
 'price_gt250k': df[df.price_num>250000],
 'title_stock_mismatch': df[((df.title.str.startswith('Certified')) & (df.stock_type!='Certified')) | ((df.title.str.startswith('New')) & (df.stock_type!='New')) | ((df.title.str.startswith('Used')) & (~df.stock_type.isin(['Used','Certified'])))]
}
for name, sub in checks.items():
    print('\n###', name, len(sub))
    for _,r in sub.head(8).iterrows():
        print(r.get('listing_key',''), '|', r.get('stock_type',''), '|', r.get('title',''), '| price', r.get('price',''), '| mi', r.get('mileage',''), '| has_url', bool(r.get('listing_url','')))
