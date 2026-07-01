import csv, json, math, os, re, warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, median_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, KFold, train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

import paths
warnings.filterwarnings('ignore')
CSV_PATH = paths.RAW_CSV
OUT_PATH = paths.PROJECT_ROOT / 'docs' / 'cars_com_dataset_stress_report.md'

def to_num(s):
    return pd.to_numeric(pd.Series(s).astype(str).str.replace(r'[^0-9.]+','', regex=True).replace('', np.nan), errors='coerce')

def split_age(scraped, year):
    return pd.to_datetime(scraped, errors='coerce').dt.year.fillna(2026) - year

def model_eval(df, name, train_idx, test_idx):
    feature_cols_cat = ['make','model','stock_type','body_style','fuel_type_raw','drivetrain_raw','transmission_raw','exterior_color','interior_color']
    feature_cols_num = ['year_num','mileage_num','vehicle_age']
    text_col = 'title_text'
    X_train = df.iloc[train_idx].copy(); X_test = df.iloc[test_idx].copy()
    y_train = np.log1p(X_train['price_num'].values); y_test_raw = X_test['price_num'].values
    y_train_raw = X_train['price_num'].values
    # Preprocessors
    cat = Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='')), ('ohe', OneHotEncoder(handle_unknown='ignore', min_frequency=2))])
    num = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
    title_vec = Pipeline([('sel', FunctionTransformer(lambda X: X[text_col].fillna('').astype(str), validate=False)), ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=500))])
    tab = ColumnTransformer([('cat', cat, feature_cols_cat), ('num', num, feature_cols_num)], remainder='drop')
    pre = FeatureUnion([('tab', tab), ('title', title_vec)])
    models = {
        'dummy_median': DummyRegressor(strategy='median'),
        'ridge_log': Ridge(alpha=3.0, random_state=0),
        'rf_log': RandomForestRegressor(n_estimators=250, min_samples_leaf=3, random_state=0, n_jobs=-1),
    }
    rows=[]
    for mn, m in models.items():
        pipe = Pipeline([('pre', pre), ('model', m)])
        pipe.fit(X_train, y_train)
        pred = np.expm1(pipe.predict(X_test))
        pred = np.clip(pred, 500, 500000)
        rows.append({
            'split': name, 'model': mn, 'n_train': len(train_idx), 'n_test': len(test_idx),
            'mae': float(mean_absolute_error(y_test_raw, pred)),
            'medae': float(median_absolute_error(y_test_raw, pred)),
            'mape': float(mean_absolute_percentage_error(y_test_raw, pred)),
            'r2': float(r2_score(y_test_raw, pred)),
            'within_10pct': float(np.mean(np.abs(pred-y_test_raw)/y_test_raw <= 0.10)),
            'within_20pct': float(np.mean(np.abs(pred-y_test_raw)/y_test_raw <= 0.20)),
        })
    return rows

def main():
    df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
    total = len(df)
    df['price_num'] = to_num(df['price'])
    df['mileage_num'] = to_num(df['mileage'])
    df['year_num'] = pd.to_numeric(df['year'], errors='coerce')
    df['vehicle_age'] = split_age(df['scraped_at'], df['year_num'])
    df['title_text'] = df['title'].fillna('').astype(str)
    req_cols = ['listing_key','listing_url','title','year','make','model','trim','price','mileage','body_style','stock_type','stock_number','exterior_color','interior_color','fuel_type_raw','drivetrain_raw','transmission_raw','engine_raw','mpg_text','title_status_raw','no_accidents_raw','owner_count','cpo_raw','scraped_at','enriched_at']

    missing = {c: int((df[c].astype(str).str.strip()=='').sum()) for c in df.columns}
    missing_pct = {c: round(missing[c]/total*100,1) for c in df.columns}
    dupe_keys = int(total - df['listing_key'].nunique())
    dupe_urls = int(total - df.loc[df['listing_url']!='','listing_url'].nunique() - int((df['listing_url']=='').sum())) if (df['listing_url']!='').any() else 0

    usable = df[df['price_num'].notna() & (df['price_num']>0)].copy()
    # Sanity flags
    flags = {}
    flags['missing_price'] = int(df['price_num'].isna().sum() + (df['price_num'].fillna(0)<=0).sum())
    flags['missing_year'] = int(df['year_num'].isna().sum())
    flags['missing_mileage'] = int(df['mileage_num'].isna().sum())
    flags['implausible_year'] = int(((df['year_num']<1980) | (df['year_num']>2027)).fillna(False).sum())
    flags['used_or_cpo_low_mileage_lt100'] = int(((df['stock_type'].isin(['Used','Certified'])) & (df['mileage_num']<100)).fillna(False).sum())
    flags['new_high_mileage_gt1000'] = int(((df['stock_type']=='New') & (df['mileage_num']>1000)).fillna(False).sum())
    flags['used_price_lt1000'] = int(((df['stock_type'].isin(['Used','Certified'])) & (df['price_num']<1000)).fillna(False).sum())
    flags['price_gt250k'] = int((df['price_num']>250000).fillna(False).sum())
    flags['title_stock_mismatch'] = int((((df['title'].str.startswith('Certified')) & (df['stock_type']!='Certified')) | ((df['title'].str.startswith('New')) & (df['stock_type']!='New')) | ((df['title'].str.startswith('Used')) & (~df['stock_type'].isin(['Used','Certified'])))).fillna(False).sum())
    flags['cpo_mismatch'] = int((((df['title'].str.startswith('Certified')) & (df['cpo_raw']!='true')) | ((~df['title'].str.startswith('Certified')) & (df['cpo_raw']=='true'))).fillna(False).sum())

    # Price distribution
    dist = usable['price_num'].describe(percentiles=[.01,.05,.1,.25,.5,.75,.9,.95,.99]).to_dict()
    mileage_dist = df['mileage_num'].dropna().describe(percentiles=[.01,.05,.1,.25,.5,.75,.9,.95,.99]).to_dict()
    stock_counts = df['stock_type'].replace('', '(blank)').value_counts().to_dict()
    make_counts = df['make'].replace('', '(blank)').value_counts().head(25).to_dict()
    year_counts = df['year'].replace('', '(blank)').value_counts().head(20).to_dict()
    scraped_counts = pd.to_datetime(df['scraped_at'], errors='coerce').dt.date.astype(str).value_counts().to_dict()

    # Feature availability for model rows
    ml = usable[(usable['year_num'].notna()) & (usable['make']!='') & (usable['model']!='')].copy()
    # Avoid training on extreme impossible? Include stress on full as-is; also filtered realistic.
    realistic = ml[(ml['price_num'].between(1000,250000)) & (ml['year_num'].between(1980,2027))].copy()

    eval_rows=[]
    if len(realistic) >= 100:
        idx = np.arange(len(realistic))
        train_idx, test_idx = train_test_split(idx, test_size=0.25, random_state=42, stratify=realistic['stock_type'].where(realistic['stock_type'].map(realistic['stock_type'].value_counts())>=4, 'other'))
        eval_rows += model_eval(realistic.reset_index(drop=True), 'random_stratified_stock', train_idx, test_idx)
        # Older-model stress: train on >=2018, test older than 2018, if enough
        newer = realistic.index[realistic['year_num']>=2018].to_numpy(); older = realistic.index[realistic['year_num']<2018].to_numpy()
        if len(newer) > 100 and len(older) > 50:
            pos = {idx:i for i,idx in enumerate(realistic.index)}
            eval_rows += model_eval(realistic.reset_index(drop=True), 'train_2018plus_test_pre2018', [pos[i] for i in newer], [pos[i] for i in older])
        # Make holdout stress: hold out top common makes one-by-one
        vc = realistic['make'].value_counts()
        for hold in [m for m,c in vc.head(6).items() if c>=25]:
            train = realistic.index[realistic['make']!=hold].to_numpy(); test = realistic.index[realistic['make']==hold].to_numpy()
            if len(test)>=25 and len(train)>=100:
                pos={idx:i for i,idx in enumerate(realistic.index)}
                eval_rows += model_eval(realistic.reset_index(drop=True), f'holdout_make_{hold}', [pos[i] for i in train], [pos[i] for i in test])

    # Leakage / representativeness
    title_repeat = df['title'].value_counts().head(20).to_dict()
    seller_proxy = df.groupby(['stock_type','make','model','trim']).size().sort_values(ascending=False).head(20).to_dict()
    old_rows = int((df['listing_url']=='').sum())
    recent_rows = int((df['listing_url']!='').sum())

    lines=[]
    lines.append('# Cars.com raw dataset stress report for price prediction')
    lines.append('')
    lines.append(f'CSV: `{CSV_PATH}`')
    lines.append(f'Total rows: **{total}**; unique listing keys: **{df["listing_key"].nunique()}**; duplicate keys: **{dupe_keys}**')
    lines.append(f'Rows with detail URLs: **{recent_rows}**; rows without listing_url, likely older/card-only rows: **{old_rows}**')
    lines.append('')
    lines.append('## Target/coverage')
    lines.append(f'Rows with usable positive price: **{len(usable)} / {total} ({len(usable)/total:.1%})**')
    lines.append(f'Rows usable for basic ML (price + year + make + model): **{len(ml)} / {total} ({len(ml)/total:.1%})**')
    lines.append(f'Realistic filtered ML rows ($1k-$250k, years 1980-2027): **{len(realistic)} / {total} ({len(realistic)/total:.1%})**')
    lines.append('')
    lines.append('## Missingness worst fields')
    for c,pct in sorted(missing_pct.items(), key=lambda kv: kv[1], reverse=True)[:20]:
        lines.append(f'- `{c}`: {missing[c]} missing ({pct}%)')
    lines.append('')
    lines.append('## Sanity flags')
    for k,v in flags.items(): lines.append(f'- {k}: **{v}**')
    lines.append('')
    lines.append('## Stock type counts')
    lines.append('```json\n'+json.dumps(stock_counts, indent=2)+'\n```')
    lines.append('## Top makes')
    lines.append('```json\n'+json.dumps(make_counts, indent=2)+'\n```')
    lines.append('## Price distribution')
    lines.append('```json\n'+json.dumps({k: (None if pd.isna(v) else round(float(v),2)) for k,v in dist.items()}, indent=2)+'\n```')
    lines.append('## Mileage distribution')
    lines.append('```json\n'+json.dumps({k: (None if pd.isna(v) else round(float(v),2)) for k,v in mileage_dist.items()}, indent=2)+'\n```')
    lines.append('## Most repeated exact titles')
    lines.append('```json\n'+json.dumps(title_repeat, indent=2)+'\n```')
    lines.append('')
    lines.append('## Baseline model stress tests')
    if eval_rows:
        edf=pd.DataFrame(eval_rows).sort_values(['split','mae'])
        cols = ['split','model','n_train','n_test','mae','medae','mape','r2','within_10pct','within_20pct']
        lines.append('| ' + ' | '.join(cols) + ' |')
        lines.append('| ' + ' | '.join(['---']*len(cols)) + ' |')
        for _, rr in edf[cols].iterrows():
            vals=[]
            for c in cols:
                v=rr[c]
                if isinstance(v, float):
                    vals.append(f'{v:.3f}')
                else:
                    vals.append(str(v))
            lines.append('| ' + ' | '.join(vals) + ' |')
    else:
        lines.append('Not enough rows for model evaluation.')
    lines.append('')
    lines.append('## Notes')
    lines.append('- Metrics are intentionally rough stress tests, not final model claims.')
    lines.append('- Target was trained as log(price), then predictions were transformed back to dollars.')
    lines.append('- Random split likely overstates real performance because similar vehicles/dealers/price bands appear in train and test.')
    lines.append('- Holdout-make and old-vehicle tests are more realistic stress tests for generalization.')
    OUT_PATH.write_text('\n'.join(lines))

    summary = {
        'csv': str(CSV_PATH), 'report': str(OUT_PATH), 'rows': total, 'unique_keys': int(df['listing_key'].nunique()),
        'dupe_keys': dupe_keys, 'usable_price_rows': int(len(usable)), 'ml_rows': int(len(ml)), 'realistic_ml_rows': int(len(realistic)),
        'stock_counts': stock_counts, 'flags': flags, 'eval_rows': eval_rows[:50]
    }
    print(json.dumps(summary, indent=2))

if __name__ == '__main__': main()
