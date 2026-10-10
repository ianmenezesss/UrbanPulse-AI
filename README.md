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