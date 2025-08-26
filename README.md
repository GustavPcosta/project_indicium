# PProductions — Análise de Banco de Dados Cinematográfico

Este repositório contém:
- **EDA completa** (notebook em `notebooks/EDA_and_Modeling.ipynb`).
- **Modelagem** de predição de nota do IMDb e fatores associados ao faturamento (receita bruta).
- Scripts para **treino** e **predição**.
- Pipeline reproduzível com boas práticas.

## Estrutura
```
pproductions-imdb-analysis/
├── data/                  # coloque aqui o dataset .csv (ex.: imdb.csv)
├── models/                # modelos treinados (.pkl) serão salvos aqui
├── notebooks/
│   └── EDA_and_Modeling.ipynb
├── reports/               # gráficos e relatórios exportados
├── src/
│   ├── train_model.py
│   └── predict_single.py
└── requirements.txt
```

## Requisitos
Veja `requirements.txt` para versões testadas.

## Instalação
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

## Dados
Coloque um CSV em `data/imdb.csv`. O notebook e os scripts são tolerantes a nomes de colunas comuns.
Exemplos esperados (alguns): `Series_Title`, `Released_Year`, `Certificate`, `Runtime`, `Genre`, `Overview`,
`Meta_score`, `Director`, `Star1`, `Star2`, `Star3`, `Star4`, `No_of_Votes`, `Gross`, `IMDB_Rating`.

> Dica: No Kaggle há conjuntos como "IMDb Top 1000".
> Limpe `Gross` (remova vírgulas/símbolos) e `Runtime` (extraia minutos).

## Uso — Notebook
Abra `notebooks/EDA_and_Modeling.ipynb`, ajuste o caminho do CSV se necessário e rode as células. O notebook:
- Executa EDA (distribuições, correlações robustas, texto com TF-IDF e tópicos simples).
- Responde perguntas de negócio.
- Treina modelos para **IMDB_Rating** (regressão).
- Treina modelos para **Gross** (regressão) e fornece importâncias de features.
- Salva o melhor modelo para nota IMDb em `models/model_imdb.pkl`.

## Uso — Linha de Comando
Treinar e salvar o modelo IMDb:
```bash
python src/train_model.py --data data/imdb.csv --target IMDB_Rating --out models/model_imdb.pkl
```

Fazer predição para um único filme (JSON inline ou arquivo):
```bash
python src/predict_single.py --model models/model_imdb.pkl --json '{
  "Series_Title": "The Shawshank Redemption",
  "Released_Year": "1994",
  "Certificate": "A",
  "Runtime": "142 min",
  "Genre": "Drama",
  "Overview": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
  "Meta_score": 80.0,
  "Director": "Frank Darabont",
  "Star1": "Tim Robbins",
  "Star2": "Morgan Freeman",
  "Star3": "Bob Gunton",
  "Star4": "William Sadler",
  "No_of_Votes": 2343110,
  "Gross": "28,341,469"
}'
```

## Licença
Somente para fins educacionais e de avaliação interna da PProductions.
