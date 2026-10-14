# UrbanPulse AI

## PortuguÃªs

### Problema
A demanda por bicicletas compartilhadas varia ao longo do dia e estÃ¡ associada a fatores temporais, sazonais e meteorolÃ³gicos. Compreender esses padrÃµes ajuda a contextualizar o planejamento operacional de um sistema de mobilidade urbana.

### Objetivo
Este projeto de portfÃ³lio tem como objetivo explorar a demanda horÃ¡ria por bicicletas compartilhadas e, em etapas futuras, avaliar abordagens de previsÃ£o com validaÃ§Ã£o adequada para sÃ©ries temporais.

### Fonte de dados
Utilizamos o **Bike Sharing Dataset** da UCI Machine Learning Repository. Consulte [docs/data_source.md](docs/data_source.md) para fonte oficial, licenÃ§a, citaÃ§Ã£o e limitaÃ§Ãµes.

### IngestÃ£o
Com Python 3.10 ou superior, instale o projeto em modo editÃ¡vel e execute o comando:

```bash
python -m pip install -e .
urbanpulse-ingest
```

O comando baixa o arquivo horÃ¡rio oficial e extrai `hour.csv` sem transformar seus bytes em `data/raw/hour.csv`. Se o arquivo jÃ¡ existir, a ingestÃ£o nÃ£o faz outra requisiÃ§Ã£o. Para solicitar explicitamente uma atualizaÃ§Ã£o:

```bash
urbanpulse-ingest --force
```

Um destino alternativo pode ser informado com `--output caminho/para/hour.csv`. O download requer conexÃ£o com a internet. A origem, o destino e o tamanho do CSV salvo sÃ£o registrados no log.

### Arquitetura e roadmap
- `data/raw/`: dados originais baixados localmente e ignorados pelo Git.
- `data/processed/`: dados derivados localmente e ignorados pelo Git.
- `src/urbanpulse/`: cÃ³digo reutilizÃ¡vel de ingestÃ£o e futuras anÃ¡lises.
- `notebooks/`: exploraÃ§Ã£o e experimentos.
- `tests/`: testes automatizados.
- `reports/`: relatÃ³rios e artefatos selecionados.
- `docs/`: documentaÃ§Ã£o.

PrÃ³ximas etapas planejadas: validar os dados, explorar padrÃµes temporais, definir avaliaÃ§Ã£o temporal e entÃ£o comparar modelos de referÃªncia e candidatos. Limpeza, anÃ¡lise e previsÃ£o nÃ£o fazem parte da ingestÃ£o implementada aqui.

## English

### Problem
Hourly bike-sharing demand varies throughout the day and is associated with temporal, seasonal, and weather factors. Understanding these patterns can inform operational planning for an urban mobility system.

### Goal
This portfolio project aims to explore hourly bike-sharing demand and, in future stages, evaluate forecasting approaches using time-series-appropriate validation.

### Data source
We use the UCI Machine Learning Repository **Bike Sharing Dataset**. See [docs/data_source.md](docs/data_source.md) for the official source, license, citation, and limitations.

### Ingestion
With Python 3.10 or later, install the project in editable mode and run:

```bash
python -m pip install -e .
urbanpulse-ingest
```

The command downloads the official hourly archive and extracts `hour.csv` without transforming its bytes into `data/raw/hour.csv`. If the file already exists, no new request is made. To explicitly refresh it, run:

```bash
urbanpulse-ingest --force
```

Use `--output path/to/hour.csv` to choose another destination. Internet access is required. The log records the source, destination, and size of the saved CSV.

### Architecture and roadmap
- `data/raw/`: locally downloaded original data, ignored by Git.
- `data/processed/`: locally derived data, ignored by Git.
- `src/urbanpulse/`: reusable ingestion and future analysis code.
- `notebooks/`: exploration and experiments.
- `tests/`: automated tests.
- `reports/`: selected reports and artifacts.
- `docs/`: documentation.

Planned next steps are data validation, temporal exploration, time-aware evaluation, and comparison of baseline and candidate models. Cleaning, analysis, and forecasting are outside this ingestion implementation.

### Preparação dos dados
Com o ambiente virtual ativo e as dependências instaladas, gere `data/processed/hour.parquet`:

```bash
urbanpulse-prepare
```

O comando valida o CSV antes de processá-lo, preserva as colunas originais, acrescenta `datetime` a partir de `dteday` e `hr`, ordena os registros cronologicamente e não remove linhas. Se o Parquet de destino já existir, a execução falha sem sobrescrevê-lo. Para substituí-lo explicitamente, use:

```bash
urbanpulse-prepare --force
```

Use `--input caminho/hour.csv` ou `--output caminho/hour.parquet` para indicar outros caminhos. O arquivo em `data/raw/` permanece inalterado.

### Data preparation
With the virtual environment active and project dependencies installed, create `data/processed/hour.parquet`:

```bash
urbanpulse-prepare
```

The command validates the CSV before processing, preserves all original columns, adds `datetime` from `dteday` and `hr`, sorts records chronologically, and removes no rows. If the destination Parquet already exists, the command fails without overwriting it. To explicitly replace it, run:

```bash
urbanpulse-prepare --force
```

Use `--input path/hour.csv` or `--output path/hour.parquet` to choose other paths. The file under `data/raw/` remains unchanged.

### EDA notebook
Install the optional analysis dependencies in the project virtual environment and launch JupyterLab from the repository root:

```bash
python -m pip install -e ".[analysis]"
python -m jupyter lab
```

Open `notebooks/01_exploratory_analysis.ipynb` and run the cells from top to bottom. The notebook reads `data/processed/hour.parquet`; if it is missing, first run `urbanpulse-prepare`. It does not write to either source dataset.

### Notebook de EDA
Instale as dependências opcionais de análise no ambiente virtual do projeto e inicie o JupyterLab na raiz do repositório:

```bash
python -m pip install -e ".[analysis]"
python -m jupyter lab
```

Abra `notebooks/01_exploratory_analysis.ipynb` e execute as células de cima para baixo. O notebook lê `data/processed/hour.parquet`; se o arquivo não existir, execute primeiro `urbanpulse-prepare`. O notebook não grava nos datasets de origem.

### ValidaÃ§Ã£o inicial dos dados
Com o ambiente do projeto ativo e as dependÃªncias instaladas (`python -m pip install -e .`), valide o CSV original sem modificÃ¡-lo:

```bash
urbanpulse-validate
```

Para validar outro arquivo, use `urbanpulse-validate --input caminho/para/hour.csv`. Erros de esquema, dados ausentes, tipos/valores invÃ¡lidos ou inconsistÃªncias entre variÃ¡veis fazem o comando terminar com cÃ³digo diferente de zero. Linhas integralmente duplicadas e colunas adicionais sÃ£o avisos; identificadores `instant` repetidos sÃ£o erro. O relatÃ³rio mostra dimensÃµes, tipos, ausÃªncias, duplicatas, avisos e erros. As regras e limites estÃ£o descritos no cÃ³digo do validador.
### Initial data validation
With the project environment active and dependencies installed (`python -m pip install -e .`), validate the original CSV without modifying it:

```bash
urbanpulse-validate
```

To validate another file, run `urbanpulse-validate --input path/to/hour.csv`. Schema, missing-data, type, domain, and cross-field errors return a nonzero exit code. Fully duplicated rows and extra columns are warnings; repeated `instant` identifiers are errors. See [docs/data_validation.md](docs/data_validation.md) for the full criteria.

### Dashboard local
Instale o extra opcional do dashboard no ambiente virtual e inicie o Streamlit na raiz do repositório:

```bash
python -m pip install -e ".[dashboard]"
python -m streamlit run src/urbanpulse/dashboard.py
```

Abra o endereço local indicado pelo Streamlit (por padrão, `http://localhost:8501`). Os filtros de ano e tipo de dia atualizam todos os indicadores e gráficos. O filtro de dia segue `workingday` da UCI: dias não úteis incluem fins de semana e feriados. Se o Parquet não existir, gere-o antes com `urbanpulse-prepare`.

### Local dashboard
Install the optional dashboard extra in the project virtual environment, then start Streamlit from the repository root:

```bash
python -m pip install -e ".[dashboard]"
python -m streamlit run src/urbanpulse/dashboard.py
```

Open the local URL printed by Streamlit (default: `http://localhost:8501`). Year and day-type filters update all indicators and charts. Day type follows UCI's `workingday` field: non-working days include weekends and holidays. If the Parquet file is missing, create it first with `urbanpulse-prepare`.

### Baseline de Machine Learning
Instale o extra opcional de ML no ambiente virtual do projeto e execute a avaliação a partir da raiz:

```bash
python -m pip install -e ".[ml]"
urbanpulse-baseline
```

O módulo também pode ser executado com `python -m urbanpulse.ml.baseline` ou apontado para outro Parquet com `--input caminho/hour.parquet`. Os registros são ordenados por `datetime` e divididos cronologicamente em 80% para treino e 20% para teste, sem embaralhamento. São comparadas a média global de `cnt` no treino e a média de `cnt` no treino para a mesma hora do dia. Esta segunda referência usa apenas `datetime` do registro de teste para escolher a hora; todas as médias vêm exclusivamente do treino. `cnt` é o alvo, e `casual` e `registered` não são usados nas previsões. MAE e RMSE medem o erro em aluguéis por hora (quanto menores, melhor); R² compara a previsão com a variação observada (quanto maior, melhor, podendo ser negativo). Esse resultado é uma referência inicial, não uma avaliação de modelo avançado.

### Machine learning baseline
Install the optional ML extra in the project virtual environment and run the evaluation from the repository root:

```bash
python -m pip install -e ".[ml]"
urbanpulse-baseline
```

You can also run `python -m urbanpulse.ml.baseline` or pass another Parquet file with `--input path/hour.parquet`. Rows are sorted by `datetime` and split chronologically into 80% training and 20% test, without shuffling. The evaluation compares the global training mean of `cnt` with the training mean of `cnt` for the same hour of day. The latter uses only a test row's `datetime` to select its hour; all means come exclusively from training. `cnt` is the target, while `casual` and `registered` are not used for predictions. MAE and RMSE measure error in rentals per hour (lower is better); R² compares predictions with observed variation (higher is better and can be negative). This is an initial reference, not an advanced model evaluation.

### Regressão Ridge
Instale o extra opcional de ML e execute o modelo na raiz do projeto:

```bash
python -m pip install -e ".[ml]"
urbanpulse-ridge
```

O Ridge usa o mesmo split cronológico 80/20 dos baselines e `alpha=1.0`, sem busca de hiperparâmetros. As features são variáveis de calendário e condições meteorológicas; `instant`, `casual`, `registered` e `cnt` são excluídas das entradas. Variáveis categóricas passam por `OneHotEncoder`; variáveis meteorológicas numéricas passam por `StandardScaler`. Ambos os transformadores são ajustados somente no treino. **O uso das condições meteorológicas pressupõe que previsões meteorológicas para a hora futura estejam disponíveis no momento da previsão.** Nesta primeira avaliação, os valores meteorológicos observados no dataset servem como representação dessas informações disponíveis; na aplicação real, usar observações futuras causaria vazamento e seria necessário fornecer previsões meteorológicas.

Resultados obtidos neste split (13.903 linhas de treino; 3.476 de teste):

| Abordagem | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Média global de treino | 174,9846 | 232,6084 | -0,1130 |
| Média de treino por hora | 113,7316 | 167,3239 | 0,4241 |
| Ridge (`alpha=1.0`) | 98,8691 | 133,9076 | 0,6311 |

Neste holdout, Ridge apresentou os menores erros e o maior R² entre as três abordagens. Essa comparação corresponde a um único período de teste e não substitui validação temporal adicional.

### Ridge regression
Install the optional ML extra and run the model from the repository root:

```bash
python -m pip install -e ".[ml]"
urbanpulse-ridge
```

Ridge uses the same chronological 80/20 split as the baselines and a fixed `alpha=1.0`, with no hyperparameter search. Predictors are calendar and weather variables; `instant`, `casual`, `registered`, and `cnt` are excluded from inputs. Categorical variables use `OneHotEncoder`; numeric weather variables use `StandardScaler`. Both transformers are fitted on training data only. **Using weather conditions assumes forecasts for the target future hour are available when the prediction is made.** In this initial evaluation, observed weather values in the dataset represent that available information; a real application must supply weather forecasts, since using future observations would leak information.

Results on this split (13,903 training rows; 3,476 test rows):

| Approach | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Global training mean | 174.9846 | 232.6084 | -0.1130 |
| Training mean by hour | 113.7316 | 167.3239 | 0.4241 |
| Ridge (`alpha=1.0`) | 98.8691 | 133.9076 | 0.6311 |

On this holdout, Ridge had the lowest errors and highest R² among the three approaches. This comparison covers one test period and does not replace further temporal validation.
