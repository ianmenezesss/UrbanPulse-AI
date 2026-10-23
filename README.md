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

O Ridge usa o mesmo split cronológico 80/20 dos baselines e `alpha=1.0`, sem busca de hiperparâmetros. As features são variáveis de calendário e condições meteorológicas; `instant`, `casual`, `registered` e `cnt` são excluídas das entradas. Variáveis categóricas passam por `OneHotEncoder`; variáveis meteorológicas numéricas passam por `StandardScaler`. Ambos os transformadores são ajustados somente no treino. A avaliação retrospectiva usa clima observado na própria hora prevista, que não estaria disponível em uma previsão operacional. Para medir desempenho operacional seriam necessárias previsões meteorológicas arquivadas, disponíveis no instante da previsão; esse cenário ainda não foi avaliado.

Resultados obtidos neste split (13.903 linhas de treino; 3.476 de teste):

| Abordagem | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Média global de treino | 174,9846 | 232,6084 | -0,1130 |
| Média de treino por hora | 113,7316 | 167,3239 | 0,4241 |
| Ridge (`alpha=1.0`) | 98,8691 | 133,9076 | 0,6311 |

Neste holdout já consultado, Ridge apresentou os menores erros e o maior R² entre as três abordagens. Essa comparação é retrospectiva e não representa uma avaliação independente do modelo escolhido.

### Ridge regression
Install the optional ML extra and run the model from the repository root:

```bash
python -m pip install -e ".[ml]"
urbanpulse-ridge
```

Ridge uses the same chronological 80/20 split as the baselines and a fixed `alpha=1.0`, with no hyperparameter search. Predictors are calendar and weather variables; `instant`, `casual`, `registered`, and `cnt` are excluded from inputs. Categorical variables use `OneHotEncoder`; numeric weather variables use `StandardScaler`. Both transformers are fitted on training data only. The retrospective evaluation uses weather observed at the target hour, which is not information available in a real forecast. Operational performance requires archived weather forecasts available at prediction time; this project has not evaluated that setting.

Results on this split (13,903 training rows; 3,476 test rows):

| Approach | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Global training mean | 174.9846 | 232.6084 | -0.1130 |
| Training mean by hour | 113.7316 | 167.3239 | 0.4241 |
| Ridge (`alpha=1.0`) | 98.8691 | 133.9076 | 0.6311 |

On this already consulted holdout, Ridge had the lowest errors and highest R² among the three approaches. This is a retrospective comparison, not an independent evaluation of the selected model.

### Random Forest Regressor
Execute o novo modelo na raiz do projeto, com o extra `ml` instalado:

```bash
urbanpulse-random-forest
```

O Random Forest usa as mesmas features de calendário e meteorologia do Ridge e o mesmo split cronológico 80/20, sem embaralhamento. `instant`, `casual`, `registered` e `cnt` não são features. As categorias são codificadas com `OneHotEncoder`, ajustado somente no treino; as variáveis numéricas são mantidas sem escala, pois árvores não dependem da escala. Os parâmetros fixos são `n_estimators=100`, `min_samples_leaf=2`, `random_state=42` e `n_jobs=-1`: 100 árvores dão uma média de conjunto estável, folhas com pelo menos duas observações limitam árvores excessivamente específicas, a semente permite reproduzir a amostragem e `n_jobs=-1` usa os processadores disponíveis. Não houve otimização de hiperparâmetros.

Assim como no Ridge, as condições meteorológicas pressupõem que previsões para a hora futura estejam disponíveis no momento da previsão. O dataset fornece observações meteorológicas históricas como proxy; na aplicação real, devem ser usadas previsões, nunca condições observadas após a hora prevista.

Resultados no mesmo holdout cronológico (13.903 linhas de treino; 3.476 de teste):

| Abordagem | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Média global de treino | 174,9846 | 232,6084 | -0,1130 |
| Média de treino por hora | 113,7316 | 167,3239 | 0,4241 |
| Ridge (`alpha=1.0`) | 98,8691 | 133,9076 | 0,6311 |
| Random Forest | 56,4021 | 84,6040 | 0,8528 |

Neste período de teste, o Random Forest teve os menores erros e o maior R² entre as abordagens avaliadas. Este holdout já foi consultado na comparação e seleção entre modelos; trate estes valores como avaliação retrospectiva, não como estimativa independente do modelo escolhido. A avaliação usa observações meteorológicas históricas da própria hora prevista, não previsões arquivadas.

### Random Forest Regressor
Run the new model from the repository root with the `ml` extra installed:

```bash
urbanpulse-random-forest
```

Random Forest uses the same calendar and weather features as Ridge and the same chronological 80/20 split, without shuffling. `instant`, `casual`, `registered`, and `cnt` are excluded from features. Categories are encoded with `OneHotEncoder`, fitted on training data only; numeric variables remain unscaled because trees do not depend on feature scale. Fixed parameters are `n_estimators=100`, `min_samples_leaf=2`, `random_state=42`, and `n_jobs=-1`: 100 trees provide a stable ensemble average, a minimum leaf size of two limits overly specific trees, the seed makes sampling reproducible, and `n_jobs=-1` uses available processors. No hyperparameter search was performed.

As with Ridge, the retrospective evaluation uses weather observed at the target hour, which is unavailable in a real forecast. Operational performance requires archived weather forecasts available at prediction time; this project has not evaluated that setting.

Results on the same chronological holdout (13,903 training rows; 3,476 test rows):

| Approach | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Global training mean | 174.9846 | 232.6084 | -0.1130 |
| Training mean by hour | 113.7316 | 167.3239 | 0.4241 |
| Ridge (`alpha=1.0`) | 98.8691 | 133.9076 | 0.6311 |
| Random Forest | 56.4021 | 84.6040 | 0.8528 |

On this test period, Random Forest had the lowest errors and highest R² among the evaluated approaches. This holdout has already been consulted to compare and select among models; treat these values as a retrospective evaluation, not an independent estimate for the selected model. The evaluation uses historical weather observations from the predicted hour, not archived weather forecasts.

### Validação temporal e análise de erros
Com o extra opcional de ML instalado, execute na raiz do projeto:

    urbanpulse-temporal-validation

O processo ordena os dados por datetime e reserva os últimos 20% sem usá-los na análise. Nos 80% iniciais, usa cinco janelas progressivas: metade desses dados inicia o treino, e o restante é dividido em cinco blocos contíguos de validação. O treino aumenta a cada janela; todos os ajustes dos pipelines Ridge e Random Forest usam somente as linhas anteriores à respectiva janela. Não há busca de hiperparâmetros. MAE, RMSE e R² são calculados para cada modelo em cada bloco, e o resumo apresenta média e desvio padrão amostral das cinco janelas.

Os CSVs são gravados em reports/temporal_validation/; os gráficos ficam em reports/figures/. O relatório inclui previsões fora da amostra, métricas por janela, resumo, erros por hora e erros por dia da semana. As opções --input, --report-dir e --figures-dir permitem alterar caminhos.

Resultados por janela (treino/validação em linhas):

| Janela | Linhas treino/validação | Ridge MAE | Ridge RMSE | Ridge R² | Random Forest MAE | Random Forest RMSE | Random Forest R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6.951 / 1.391 | 52,9489 | 74,3539 | 0,6340 | 45,4891 | 67,3302 | 0,6999 |
| 2 | 8.342 / 1.391 | 76,2816 | 97,2003 | 0,3259 | 60,5012 | 87,4975 | 0,4538 |
| 3 | 9.733 / 1.390 | 81,3675 | 118,3897 | 0,6036 | 76,6517 | 112,5962 | 0,6414 |
| 4 | 11.123 / 1.390 | 106,4210 | 134,5316 | 0,6177 | 54,4626 | 79,8866 | 0,8652 |
| 5 | 12.513 / 1.390 | 107,4127 | 134,8470 | 0,6111 | 91,1505 | 123,0484 | 0,6762 |

Média ± desvio padrão entre as janelas:

| Modelo | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Ridge | 84,8863 ± 22,7904 | 111,8645 ± 26,0128 | 0,5584 ± 0,1305 |
| Random Forest | 65,6510 ± 18,2307 | 94,0718 ± 23,1428 | 0,6673 ± 0,1471 |

O Random Forest teve MAE e RMSE menores e R² maior nas cinco janelas, então manteve vantagem geral sobre o Ridge nesta validação. A diferença não foi uniforme por horário: ambos erraram mais nos picos de deslocamento, especialmente às 8h e entre 17h e 18h, embora o Random Forest tenha reduzido bastante esses erros. No agregado por dia, o erro absoluto médio do Random Forest foi menor em todos os dias; sábado apresentou os maiores erros para ambos. Os erros médios ficaram negativos, indicando subestimação da demanda, com maior subestimação em horários de pico. Essas métricas resumem apenas as previsões fora da amostra dentro dos 80% iniciais.

As features meteorológicas pressupõem disponibilidade de previsões para cada hora futura. Os valores observados no dataset são apenas um proxy para essa informação; não use observações meteorológicas futuras em uma operação real.

### Temporal validation and error analysis
With the optional ML extra installed, run from the repository root:

    urbanpulse-temporal-validation

The process sorts records by datetime and reserves the final 20% without using them in this analysis. Within the first 80%, it uses five expanding windows: half of this development data forms the initial training set, and the remainder is split into five contiguous validation blocks. Training history grows at every window; Ridge and Random Forest pipelines are fitted only on rows before that window. No hyperparameter search is performed. MAE, RMSE, and R² are calculated for each model in each block; the summary reports the mean and sample standard deviation across the five windows.

CSV files are written to reports/temporal_validation/; plots go to reports/figures/. The reports include out-of-sample predictions, per-window metrics, summary metrics, and errors by hour and weekday. The --input, --report-dir, and --figures-dir options can override the paths.

Per-window results (training/validation rows):

| Window | Train/validation rows | Ridge MAE | Ridge RMSE | Ridge R² | Random Forest MAE | Random Forest RMSE | Random Forest R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6,951 / 1,391 | 52.9489 | 74.3539 | 0.6340 | 45.4891 | 67.3302 | 0.6999 |
| 2 | 8,342 / 1,391 | 76.2816 | 97.2003 | 0.3259 | 60.5012 | 87.4975 | 0.4538 |
| 3 | 9,733 / 1,390 | 81.3675 | 118.3897 | 0.6036 | 76.6517 | 112.5962 | 0.6414 |
| 4 | 11,123 / 1,390 | 106.4210 | 134.5316 | 0.6177 | 54.4626 | 79.8866 | 0.8652 |
| 5 | 12,513 / 1,390 | 107.4127 | 134.8470 | 0.6111 | 91.1505 | 123.0484 | 0.6762 |

Mean ± standard deviation across windows:

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Ridge | 84.8863 ± 22.7904 | 111.8645 ± 26.0128 | 0.5584 ± 0.1305 |
| Random Forest | 65.6510 ± 18.2307 | 94.0718 ± 23.1428 | 0.6673 ± 0.1471 |

Random Forest had lower MAE and RMSE and higher R² in all five windows, so it maintained an overall advantage over Ridge in this validation. The difference varied by hour: both models made larger errors during commute peaks, especially at 8 a.m. and between 5 p.m. and 6 p.m., although Random Forest reduced these errors substantially. By weekday, Random Forest had lower aggregate MAE on every day; Saturday had the largest errors for both models. Mean errors were negative, indicating demand underprediction, with stronger underprediction during peak hours. These metrics summarize only out-of-sample predictions within the initial 80%.

Weather features assume forecasts are available for each future hour. Observed dataset values are only a proxy for this information; future weather observations must not be used in a real deployment.

### Dashboard local
Instale o extra do dashboard e o extra de ML (necessário para gerar o relatório do teste final) no ambiente virtual:

    python -m pip install -e ".[dashboard,ml]"

Para atualizar os relatórios que alimentam o painel, execute uma vez na raiz:

    urbanpulse-model-comparison
    urbanpulse-temporal-validation

Inicie o Streamlit:

    python -m streamlit run src/urbanpulse/dashboard.py

Abra o endereço mostrado no terminal, normalmente http://localhost:8501. A navegação lateral tem cinco áreas:

- **Visão geral:** indicadores atuais e exploração da demanda; os filtros de ano e tipo de dia continuam disponíveis.
- **Modelos de Machine Learning:** MAE, RMSE e R² dos quatro modelos no teste final dos últimos 20%. Esse teste já foi consultado na comparação entre modelos; não o use para novas decisões nem o interprete como avaliação independente. As métricas são retrospectivas e usam clima histórico observado na hora prevista.
- **Validação temporal:** resultados fora da amostra de Ridge e Random Forest em cinco janelas dentro dos 80% iniciais. Também usam clima histórico observado; não representam desempenho com previsão meteorológica futura.
- **Simulador de demanda:** estima aluguéis por hora com o pipeline Random Forest salvo e condições informadas pelo usuário.
- **Interpretabilidade:** apresenta Permutation Importance calculada nas janelas de validação dos 80% iniciais; importância representa associação preditiva, não causalidade.

MAE e RMSE são erros em aluguéis por hora, então valores menores indicam previsões mais próximas; RMSE dá peso maior a erros grandes. R² compara a previsão com uma referência baseada na média e pode ser negativo. Na validação temporal, as curvas por janela mostram estabilidade entre períodos, pontos próximos à diagonal representam previsões próximas dos valores reais, e erro negativo indica subestimação. O painel apenas lê os relatórios: ele não treina modelos. Se os CSVs estiverem ausentes, as respectivas áreas mostram o comando para gerá-los.

O uso de variáveis meteorológicas nos modelos pressupõe previsões disponíveis para cada hora futura. As avaliações atuais usam condições observadas no dataset e, portanto, não medem o desempenho com previsões meteorológicas reais. Em operação, não use observações meteorológicas posteriores à hora prevista.

### Publicação no Streamlit Community Cloud

O deploy usa Python 3.12 e as dependências diretas fixadas em `requirements.txt`; a linha `-e .[cloud]` instala o pacote `urbanpulse` deste repositório junto com o runtime do dashboard. Os relatórios CSV exigidos pelas páginas de modelos, validação temporal e interpretabilidade já estão versionados em `reports/`; o dashboard valida os arquivos existentes e não os recalcula durante a inicialização.

Para preparar recursos sem uma etapa manual de shell no deploy, o dashboard chama `prepare_cloud_resources()` através de `st.cache_resource`: em cada processo frio ele baixa o arquivo oficial `hour.csv` se estiver ausente, cria `data/processed/hour.parquet` com a etapa de preparação existente e valida os relatórios presentes. A rotina é idempotente, não altera o CSV bruto e não treina modelos. Se o processo for reiniciado e o armazenamento local não estiver disponível, os recursos ausentes serão preparados novamente.

Para executar o mesmo bootstrap manualmente (por exemplo, para verificar um ambiente antes do deploy), instale o extra Cloud e rode o script na raiz do repositório:

```bash
python -m pip install -r requirements.txt
python scripts/prepare_cloud_resources.py
```

O artefato `random_forest.joblib` (~54,9 MB) e o manifesto `random_forest.joblib.metadata.json` não são colocados no Git. Anexe exatamente esses dois arquivos à GitHub Release. Gere os quatro valores dos Secrets a partir dos arquivos locais e da URL base da Release já criada:

```powershell
python scripts/generate_cloud_secrets.py `
  --release-base-url "https://github.com/OWNER/REPOSITORY/releases/download/TAG"
```

Substitua `OWNER/REPOSITORY/TAG` pelos valores reais, sem incluir nome de arquivo ao final. O comando imprime TOML para colar nos Secrets. `model_sha256` é o hash do joblib; `metadata_sha256` é calculado sobre os bytes do arquivo JSON completo. O `artifact_sha256` dentro do manifesto é uma terceira referência e deve corresponder ao hash do joblib; ele não substitui o hash externo do manifesto.

Em **App settings → Secrets** no Community Cloud, forneça URLs HTTPS públicas e os hashes SHA-256 obtidos de uma fonte confiável. Exemplo de configuração TOML (substitua os valores; não salve o arquivo de secrets no repositório):

```toml
[cloud_resources]
model_url = "https://github.com/OWNER/REPOSITORY/releases/download/TAG/random_forest.joblib"
model_sha256 = "<sha256-do-joblib>"
metadata_url = "https://github.com/OWNER/REPOSITORY/releases/download/TAG/random_forest.joblib.metadata.json"
metadata_sha256 = "<sha256-do-manifest>"
```

O bootstrap limita downloads, exige HTTPS e verifica separadamente os dois hashes antes de instalar os arquivos. Também confere a compatibilidade e o hash interno do manifesto. O pipeline só é desserializado quando o simulador é usado, através do carregador em cache existente. `joblib` pode executar código ao desserializar; publique e carregue somente artefatos de uma origem confiável. O arquivo local `.streamlit/secrets.toml` está ignorado pelo Git.

Para criar a Release manualmente com GitHub CLI (`gh`), depois de escolher uma tag:

```powershell
$tag = "urbanpulse-v0.1.0"
gh release create $tag `
  "models/random_forest.joblib" `
  "models/random_forest.joblib.metadata.json" `
  --title "UrbanPulse AI $tag" `
  --notes "Pipeline Random Forest e manifesto de integridade para o simulador."
```

Passos para publicar:

1. Envie o repositório ao GitHub com `requirements.txt`, `.streamlit/config.toml`, `src/`, os sete relatórios CSV exigidos e a documentação. Não adicione `data/` nem `models/` ao Git.
2. Crie uma GitHub Release e anexe o joblib e o manifesto correspondentes. Registre os hashes em um gerenciador confiável e configure os quatro campos de `cloud_resources` nos secrets do app.
3. Em [share.streamlit.io](https://share.streamlit.io), crie um app selecionando o repositório, a branch e `src/urbanpulse/dashboard.py` como caminho do arquivo principal.
4. Em **Advanced settings**, selecione Python 3.12 e cole a configuração `cloud_resources` no campo **Secrets**. Salve e inicie o deploy.
5. Abra a URL criada e confira Visão geral, Modelos de Machine Learning, Validação temporal, Simulador de demanda e Interpretabilidade. No primeiro início, aguarde a preparação dos dados e a transferência verificada do modelo.

O CSV UCI tem cerca de 1,16 MB e o Parquet local atual cerca de 0,40 MB; os relatórios são pequenos. O modelo requer uma transferência de aproximadamente 55 MB a cada inicialização fria quando o sistema de arquivos local não persistir. Isso é viável em armazenamento, mas a memória real do pipeline carregado não foi medida no Community Cloud. A [documentação de limites da plataforma](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app#resource-limits) informa valores aproximados de 690 MB–2,7 GB de memória e até 50 GB de armazenamento, observados em fevereiro de 2024, e ressalva que podem mudar; faça um deploy de smoke test e confira os recursos antes de divulgar. O manifesto aceita diferenças de patch do Python dentro da mesma série major/minor (por exemplo, 3.12.10 e 3.12.12). Scikit-learn, pandas e joblib continuam exigindo igualdade exata com as versões registradas; mantenha os pins do Cloud alinhados. Diferenças de versão minor ou major do Python são rejeitadas.

### Local dashboard
Install the dashboard and ML extras in the project virtual environment (ML is needed to generate the final-test report):

    python -m pip install -e ".[dashboard,ml]"

To refresh the reports used by the dashboard, run once from the repository root:

    urbanpulse-model-comparison
    urbanpulse-temporal-validation

Start Streamlit:

    python -m streamlit run src/urbanpulse/dashboard.py

Open the address printed in the terminal, usually http://localhost:8501. The sidebar has five areas:

- **Overview:** existing demand indicators and exploration charts; year and day-type filters remain available.
- **Machine learning models:** MAE, RMSE, and R² for all four models on the last 20%. This test has already been consulted to compare models; do not use it for new modeling decisions or interpret it as an independent evaluation. The retrospective metrics use observed weather from the predicted hour.
- **Temporal validation:** out-of-sample Ridge and Random Forest results over five windows within the first 80%. These results also use observed historical weather and do not represent performance with future weather forecasts.
- **Demand simulator:** estimates hourly rentals with the saved Random Forest pipeline and user supplied conditions. Its 24-hour curve keeps weather constant and represents scenarios, not an operational forecast.
- **Interpretability:** presents Permutation Importance generated on temporal validation windows within the initial 80%; importance describes predictive association, not causality.

MAE and RMSE are errors in rentals per hour, so lower values mean closer predictions; RMSE gives more weight to large errors. R² compares predictions with a mean-based reference and may be negative. In temporal validation, curves across windows show stability over time, points near the diagonal indicate predictions near actual values, and a negative error means underprediction. The dashboard only reads reports and never trains models. If a report CSV is missing, its area shows the command to generate it. To capture screenshots, open each area in the browser and use your operating system capture tool; if screenshots are added to the portfolio, save optimized images under `docs/images/` and avoid large raw files.

Weather predictors assume forecasts are available for each future hour. Current evaluations use observed dataset conditions, so they do not measure performance with real weather forecasts. In operation, do not use weather observations made after the target hour.

### Deploy on Streamlit Community Cloud

The deployment selects Python 3.12 and uses the direct dependency pins in `requirements.txt`; `-e .[cloud]` installs this repository's local `urbanpulse` package and the dashboard runtime. The CSV reports required by the model, temporal validation, and interpretability pages are already tracked under `reports/`; the dashboard validates existing files and does not recalculate them at startup.

To prepare resources without a manual shell step during deployment, the dashboard calls `prepare_cloud_resources()` through `st.cache_resource`: on each cold process it downloads the official `hour.csv` if missing, creates `data/processed/hour.parquet` with the existing preparation step, and validates available reports. This is idempotent, leaves the raw CSV unchanged, and does not train a model. If the process restarts and local storage is unavailable, missing resources are prepared again.

To run the same bootstrap manually (for example, to check an environment before deployment), install the Cloud extra and run the script from the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/prepare_cloud_resources.py
```

The `random_forest.joblib` artifact (~54.9 MB) and `random_forest.joblib.metadata.json` manifest are not committed to Git. Attach exactly these two files to a trusted GitHub Release. Generate all four Secret values from the local files and the base URL of the created Release:

```powershell
python scripts/generate_cloud_secrets.py `
  --release-base-url "https://github.com/OWNER/REPOSITORY/releases/download/TAG"
```

Replace `OWNER/REPOSITORY/TAG` with the real values, without appending a filename. The command prints TOML to paste into Secrets. `model_sha256` hashes the joblib; `metadata_sha256` hashes the complete JSON file bytes. The `artifact_sha256` inside the manifest is a separate reference that must match the joblib hash; it does not replace the external manifest hash.

In Community Cloud **App settings → Secrets**, provide public HTTPS URLs and SHA-256 pins obtained from a trusted source. Example TOML (replace the values; do not commit a secrets file):

```toml
[cloud_resources]
model_url = "https://github.com/OWNER/REPOSITORY/releases/download/TAG/random_forest.joblib"
model_sha256 = "<joblib-sha256>"
metadata_url = "https://github.com/OWNER/REPOSITORY/releases/download/TAG/random_forest.joblib.metadata.json"
metadata_sha256 = "<manifest-sha256>"
```

Bootstrap bounds download sizes, requires HTTPS, and verifies both hashes independently before installing the files. It also checks manifest compatibility and the artifact hash recorded inside the manifest. The pipeline is deserialized only when the simulator is used, through the existing cached loader. `joblib` can execute code during deserialization; publish and load artifacts only from a trusted source. Local `.streamlit/secrets.toml` is Git-ignored.

To create the Release manually with GitHub CLI (`gh`), after choosing a tag:

```powershell
$tag = "urbanpulse-v0.1.0"
gh release create $tag `
  "models/random_forest.joblib" `
  "models/random_forest.joblib.metadata.json" `
  --title "UrbanPulse AI $tag" `
  --notes "Random Forest pipeline and integrity manifest for the simulator."
```

Deployment steps:

1. Push the repository to GitHub with `requirements.txt`, `.streamlit/config.toml`, `src/`, the seven required CSV reports, and documentation. Do not add `data/` or `models/` to Git.
2. Create a GitHub Release and attach the matching joblib and manifest. Record their hashes in a trusted manager and configure all four `cloud_resources` fields in the app secrets.
3. At [share.streamlit.io](https://share.streamlit.io), create an app by selecting the repository, branch, and `src/urbanpulse/dashboard.py` as the entrypoint.
4. In **Advanced settings**, select Python 3.12 and paste the `cloud_resources` TOML into **Secrets**. Save and deploy.
5. Open the generated URL and inspect Overview, Machine Learning models, Temporal validation, Demand simulator, and Interpretability. Allow the first run to prepare the data and transfer the verified model.

The UCI CSV is about 1.16 MB and the current local Parquet about 0.40 MB; the reports are small. The model needs an approximately 55 MB transfer on each cold start if local files do not persist. Storage should be manageable, but memory use after loading the pipeline has not been measured on Community Cloud. The [platform limits documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app#resource-limits) gives approximate values of 690 MB–2.7 GB memory and up to 50 GB storage, observed in February 2024, and notes they can change; run a deployment smoke test and inspect resource use before sharing it. The manifest accepts Python patch updates within the same major/minor series (for example, 3.12.10 and 3.12.12). Scikit-learn, pandas, and joblib still require exact version matches; keep Cloud pins aligned. Python minor or major differences are rejected.

### Simulador de demanda

Gere o pipeline do Random Forest antes de abrir o dashboard. A rotina usa o mesmo split cronológico dos modelos e ajusta o pipeline somente nos 80% iniciais; os 20% finais permanecem reservados:

```bash
python -m pip install -e ".[ml,dashboard]"
urbanpulse-train-random-forest
python -m streamlit run src/urbanpulse/dashboard.py
```

Na área **Simulador de demanda**, selecione data, hora, feriado, categoria do tempo, temperatura, sensação térmica, umidade e vento. A data só pode estar entre 2011 e 2012, período representado no treinamento. As entradas meteorológicas são convertidas para a escala normalizada UCI: temperatura `(°C + 8) / 47`, sensação térmica `(°C + 16) / 66`, umidade `% / 100` e vento `escala UCI / 67`. A fonte não especifica unidade física do vento. O painel deriva as variáveis de calendário e carrega o artefato salvo; não treina ao interagir. A estimativa é uma simulação histórica condicionada às entradas, sem consulta de meteorologia em tempo real. O arquivo `models/random_forest.joblib` é local e ignorado pelo Git; gere-o novamente com o comando acima quando necessário.

### Demand simulator

Train the Random Forest pipeline before opening the dashboard. The routine uses the same chronological split as the models and fits only on the initial 80%; the final 20% stays reserved:

```bash
python -m pip install -e ".[ml,dashboard]"
urbanpulse-train-random-forest
python -m streamlit run src/urbanpulse/dashboard.py
```

In **Simulador de demanda**, configure a date, hour, holiday flag, weather category, temperature, apparent temperature, humidity, and wind. Dates are restricted to 2011–2012, the period represented during training. Weather inputs are converted to UCI's normalized scale: temperature `(°C + 8) / 47`, apparent temperature `(°C + 16) / 66`, humidity `% / 100`, and wind `UCI scale / 67`. UCI does not specify a physical unit for wind. The dashboard derives calendar features and loads the saved artifact; it does not train during interaction. The estimate is a historical simulation conditioned on the inputs, with no real-time weather lookup. `models/random_forest.joblib` is local and Git-ignored; regenerate it with the command above when needed.

O treinamento também grava `models/random_forest.joblib.metadata.json`, com hash SHA-256 do Parquet processado, lista ordenada de features, período e tamanho do treino, parâmetros e versões de Python/scikit-learn/pandas/joblib. O carregador exige correspondência dessas versões, do hash do artefato, do schema e dos parâmetros; gere novamente o modelo após atualizar o ambiente. Joblib desserializa objetos Python e pode executar código ao carregar: use somente artefatos produzidos localmente ou obtidos de uma fonte confiável.

O calendário do simulador foi conferido contra o `hour.csv`: `season` muda em 21/03, 21/06, 23/09 e 21/12; `weekday` segue domingo=0; `yr` e `mnth` são derivados do ano/mês; o modo automático de feriado usa as datas marcadas pela UCI em 2011–2012 e `workingday` exclui fins de semana e feriados. A regra só é garantida para os dois anos do dataset. A opção de forçar feriado serve apenas para cenários.

### Gráfico de demanda prevista nas 24 horas

Na área **Simulador de demanda**, gere uma estimativa para o cenário informado. A curva mostra uma estimativa para cada hora, de 00:00 a 23:00; o marcador destaca a hora escolhida no controle. Os indicadores de maior e menor demanda identificam os extremos dessa curva, e o total diário é a soma das 24 estimativas (arredondada apenas para exibição). Para tornar a comparação horária consistente, o gráfico mantém data, feriado e condições meteorológicas constantes ao longo do dia. É uma análise de cenários históricos, não uma previsão meteorológica em tempo real. O dashboard reutiliza o artefato salvo e não retreina o Random Forest.

Training also writes `models/random_forest.joblib.metadata.json` with the processed Parquet SHA-256, ordered feature list, training period and size, parameters, and Python/scikit-learn/pandas/joblib versions. The loader requires matching versions, artifact hash, schema, and parameters; regenerate the model after changing the environment. Joblib deserializes Python objects and can execute code while loading: use only locally produced artifacts or artifacts from a trusted source.

The simulator calendar was checked against `hour.csv`: `season` changes on Mar 21, Jun 21, Sep 23, and Dec 21; `weekday` uses Sunday=0; `yr` and `mnth` come from the date; automatic holidays use UCI-marked dates in 2011–2012; and `workingday` excludes weekends and holidays. These rules are verified only for the two dataset years. Forcing a holiday is a scenario override.

### 24-hour estimated demand chart

In **Simulador de demanda**, submit the selected scenario to see estimates from 00:00 through 23:00. The selected hour is highlighted. The highest and lowest indicators show the extrema of this curve; the daily total sums all 24 estimates (rounded only for display). Date, holiday, and weather conditions stay fixed across the day so the chart compares hourly scenarios consistently. It is a historical scenario analysis, not a real-time weather forecast. The dashboard reuses the saved artifact and does not retrain Random Forest.

### Interpretabilidade com Permutation Importance

Instale os extras de ML e dashboard e gere o relatório antes de abrir a área **Interpretabilidade**:

```bash
python -m pip install -e ".[ml,dashboard]"
urbanpulse-interpretability
```

O comando ajusta um pipeline Random Forest em cada uma das cinco janelas progressivas dos 80% iniciais e mede a importância no bloco seguinte, fora da amostra. Os 20% finais não participam. O CSV `reports/interpretability/permutation_importance.csv` contém a média e o desvio padrão por variável em cada janela e uma linha agregada por variável. A métrica é o aumento do MAE (aluguéis/hora) depois de embaralhar a variável; valores maiores indicam maior importância preditiva. Na visão agregada, o desvio padrão mede a variação entre janelas; em cada janela, mede a variação entre as cinco permutações. Variáveis correlacionadas podem substituir umas às outras e reduzir a importância individual observada. A importância mede associação preditiva, não causalidade. O dashboard apenas carrega o CSV, permite alternar entre agregado e janelas e não treina nem calcula importâncias durante a interação.

### Permutation importance interpretability

Install the ML and dashboard extras, then generate the report before opening **Interpretabilidade**:

```bash
python -m pip install -e ".[ml,dashboard]"
urbanpulse-interpretability
```

The command fits a Random Forest pipeline in each of five expanding windows within the initial 80%, then measures importance on the next, out-of-sample block. The final 20% is excluded. `reports/interpretability/permutation_importance.csv` contains per-variable means and standard deviations for each window plus one aggregate row per variable. Importance is the increase in MAE (rentals/hour) after shuffling a feature; larger values indicate greater predictive importance. In the aggregate view, standard deviation describes variation across windows; within a window, it describes variation across the five permutations. Correlated features can substitute for one another and reduce the observed individual importance. Importance measures predictive association, not causation. The dashboard only loads the CSV and lets users switch between aggregate and individual windows; it does not train or calculate importance during interaction.
