# UrbanPulse AI

## Português

### Problema
A demanda por bicicletas compartilhadas varia ao longo do dia e está associada a fatores temporais, sazonais e meteorológicos. Compreender esses padrões ajuda a contextualizar o planejamento operacional de um sistema de mobilidade urbana.

### Objetivo
Este projeto de portfólio tem como objetivo explorar a demanda horária por bicicletas compartilhadas e, em etapas futuras, avaliar abordagens de previsão com validação adequada para séries temporais. Nenhuma análise ou modelo foi implementado nesta etapa.

### Fonte de dados
Será utilizado o **Bike Sharing Dataset** da UCI Machine Learning Repository, que contém registros horários e diários do sistema Capital Bikeshare, em Washington, D.C., nos anos de 2011 e 2012. Consulte [docs/data_source.md](docs/data_source.md) para a fonte oficial, licença, citação e limitações.

### Escopo inicial
Esta etapa prepara a documentação e a estrutura do projeto. O foco futuro é o arquivo horário (`hour.csv`) e a variável de contagem total (`cnt`). O dataset original contém também `day.csv`; os dados ainda não foram baixados nem processados neste projeto.

Arquitetura inicial planejada:

- `data/raw/`: cópias locais, sem versionamento, dos dados originais.
- `data/processed/`: dados derivados localmente, sem versionamento.
- `notebooks/`: exploração e experimentos reproduzíveis.
- `src/urbanpulse/`: código reutilizável de ingestão, validação e análise.
- `tests/`: testes automatizados para o código reutilizável.
- `reports/`: relatórios e artefatos de análise selecionados.
- `docs/`: documentação do projeto e das fontes de dados.

### Roadmap
1. Registrar a proveniência e obter os dados oficiais.
2. Inspecionar esquema, qualidade, cobertura temporal e padrões descritivos.
3. Definir uma divisão temporal e métricas de referência para previsão.
4. Implementar e comparar modelos de referência e modelos candidatos.
5. Documentar resultados, limitações e próximos passos.

O roadmap é planejado; nenhuma dessas funcionalidades está sendo declarada como concluída nesta etapa.

## English

### Problem
Hourly bike-sharing demand varies throughout the day and is associated with temporal, seasonal, and weather factors. Understanding these patterns can inform operational planning for an urban mobility system.

### Goal
This portfolio project aims to explore hourly bike-sharing demand and, in future stages, evaluate forecasting approaches using time-series-appropriate validation. No analysis or model has been implemented in this stage.

### Data source
The project will use the UCI Machine Learning Repository's **Bike Sharing Dataset**, which contains hourly and daily records from the Capital Bikeshare system in Washington, D.C., during 2011 and 2012. See [docs/data_source.md](docs/data_source.md) for the official source, license, citation, and limitations.

### Initial scope
This stage prepares the project documentation and structure. Future work will focus on the hourly file (`hour.csv`) and its total-count target (`cnt`). The original dataset also includes `day.csv`; the data has not yet been downloaded or processed in this project.

Planned initial architecture:

- `data/raw/`: local, untracked copies of the original data.
- `data/processed/`: locally derived, untracked data.
- `notebooks/`: exploration and reproducible experiments.
- `src/urbanpulse/`: reusable ingestion, validation, and analysis code.
- `tests/`: automated tests for reusable code.
- `reports/`: selected analysis reports and artifacts.
- `docs/`: project and data-source documentation.

### Roadmap
1. Record data provenance and obtain the official data.
2. Inspect schema, quality, temporal coverage, and descriptive patterns.
3. Define a temporal split and baseline forecasting metrics.
4. Implement and compare baseline and candidate models.
5. Document results, limitations, and next steps.

This roadmap describes planned work; none of these capabilities is claimed as complete in this stage.
