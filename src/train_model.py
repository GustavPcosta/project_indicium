import argparse, pandas as pd, numpy as np, joblib

from pathlib import Path

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline



def smart_col(df, keys):

  for c in df.columns:

    for k in keys:

      if k.lower() in c.lower():

        return c

  return None



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



def main(data, target, out):

  df = pd.read_csv(data)

  col = {

    'cert': smart_col(df, ['Certificate','Rating','MPAA']),

    'genre': smart_col(df, ['Genre','Genres']),

    'director': smart_col(df, ['Director']),

    'star1': smart_col(df, ['Star1','Actor1','Lead1']),

    'star2': smart_col(df, ['Star2','Actor2','Lead2']),

    'star3': smart_col(df, ['Star3','Actor3','Lead3']),

    'star4': smart_col(df, ['Star4','Actor4','Lead4']),

    'runtime': smart_col(df, ['Runtime']),

    'votes': smart_col(df, ['No_of_Votes','Votes']),

    'metascore': smart_col(df, ['Meta_score','Metascore']),

    'gross': smart_col(df, ['Gross','BoxOffice']),

    'year': smart_col(df, ['Released_Year','Year']),

    'imdb': smart_col(df, [target]),

  }

  assert col['imdb'] is not None, f"Coluna alvo '{target}' não encontrada."



  df['runtime_min'] = df[col['runtime']].apply(parse_runtime) if col['runtime'] else np.nan

  df['votes_num'] = pd.to_numeric(df[col['votes']], errors='coerce') if col['votes'] else np.nan

  df['metascore_num'] = pd.to_numeric(df[col['metascore']], errors='coerce') if col['metascore'] else np.nan

  df['gross_num'] = df[col['gross']].apply(parse_money) if col['gross'] else np.nan

  df['year_num'] = pd.to_numeric(df[col['year']], errors='coerce') if col['year'] else np.nan

  y = pd.to_numeric(df[col['imdb']], errors='coerce')



  cat_cols = [c for c in [col['cert'], col['genre'], col['director'], col['star1'], col['star2'], col['star3'], col['star4']] if c]

  num_cols = [c for c in ['runtime_min','votes_num','metascore_num','year_num','gross_num'] if c in df]



  x = df[cat_cols + num_cols].copy()



  pre = ColumnTransformer([

    ('num', StandardScaler(with_mean=False), num_cols),

    ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=10), cat_cols),

  ], remainder='drop')



  model = Pipeline([('prep', pre), ('rf', RandomForestRegressor(n_estimators=400, random_state=42))])

  model.fit(X, y)

  Path(out).parent.mkdir(parents=True, exist_ok=True)

  joblib.dump(model, out)

  print({'saved': out})



if __name__ == '__main__':

  import argparse

  p = argparse.ArgumentParser()

  p.add_argument('--data', required=True)

  p.add_argument('--target', default='IMDB_Rating')

  p.add_argument('--out', default='models/model_imdb.pkl')

  a = p.parse_args()

  main(a.data, a.target, a.out)

