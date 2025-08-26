import argparse, json, pandas as pd, numpy as np, joblib

def parse_runtime(s):
    import re, pandas as pd, numpy as np
    if pd.isna(s): return np.nan
    m = re.search(r'(\d+)', str(s))
    return float(m.group(1)) if m else np.nan

def parse_money(s):
    import pandas as pd, numpy as np
    if pd.isna(s): return np.nan
    s = str(s).replace(',','').replace('$','').replace('£','').replace('€','')
    try: return float(s)
    except: return np.nan

def main(model_path, js, input_path):
    model = joblib.load(model_path)
    if js:
        data = json.loads(js)
    else:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    df = pd.DataFrame([data])

    df['runtime_min'] = df['Runtime'].apply(parse_runtime) if 'Runtime' in df else np.nan
    df['gross_num'] = df['Gross'].apply(parse_money) if 'Gross' in df else np.nan
    if 'No_of_Votes' in df: df['votes_num'] = pd.to_numeric(df['No_of_Votes'], errors='coerce')
    if 'Meta_score' in df: df['metascore_num'] = pd.to_numeric(df['Meta_score'], errors='coerce')
    if 'Released_Year' in df: df['year_num'] = pd.to_numeric(df['Released_Year'], errors='coerce')

    cat_cols = [c for c in ['Certificate','Genre','Director','Star1','Star2','Star3','Star4'] if c in df]
    num_cols = [c for c in ['runtime_min','votes_num','metascore_num','year_num','gross_num'] if c in df]

    X = df[cat_cols + num_cols]
    pred = model.predict(X)[0]
    print(json.dumps({'prediction_imdb_rating': float(pred)}, ensure_ascii=False))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--json', help='JSON inline com os campos do filme')
    p.add_argument('--input', help='Caminho para arquivo JSON')
    a = p.parse_args()
    main(a.model, a.json, a.input)
