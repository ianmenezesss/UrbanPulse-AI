# Critérios de validação dos dados horários

O comando `urbanpulse-validate` lê o CSV com pandas e não grava no arquivo de origem. O alvo padrão é `data/raw/hour.csv`.

## Erros bloqueantes

- CSV inexistente, ilegível ou malformado: o comando termina com código 2. Um CSV com cabeçalho, mas sem registros, é erro de validação (código 1).
- Colunas obrigatórias ausentes: `instant`, `dteday`, `season`, `yr`, `mnth`, `hr`, `holiday`, `weekday`, `workingday`, `weathersit`, `temp`, `atemp`, `hum`, `windspeed`, `casual`, `registered`, `cnt`.
- Valores ausentes ou valores não numéricos em variáveis numéricas.
- Variáveis inteiras com valores fracionários.
- Limites inválidos: `instant` >= 1; `season` 1–4; `yr` 0–1; `mnth` 1–12; `hr` 0–23; `holiday` e `workingday` 0–1; `weekday` 0–6; `weathersit` 1–4; `temp`, `atemp`, `hum` e `windspeed` 0–1; contagens `casual`, `registered` e `cnt` >= 0.
- `instant` repetido, data `dteday` inválida ou inconsistência do mês/ano de `dteday` com `mnth`/`yr` (UCI codifica `yr=0` como 2011 e `yr=1` como 2012).
- `cnt` diferente de `casual + registered`.

Erros de validação fazem o comando terminar com código 1. Essas verificações refletem o esquema e as descrições publicadas pela UCI para o dataset horário.

## Avisos informativos

- Linhas integralmente duplicadas são contadas e reportadas como aviso; se repetirem `instant`, também haverá erro de identificador.
- Colunas adicionais são reportadas como aviso para revisão, mas não bloqueiam a leitura.

A validação não exige número fixo de linhas, não tenta corrigir dados e não verifica continuidade temporal. Essas decisões ficam para etapas futuras.

## Initial validation criteria (English)

`urbanpulse-validate` reads the CSV with pandas and does not write to the source file. The default input is `data/raw/hour.csv`.

Blocking errors include unreadable or malformed files (exit code 2); header-only datasets and missing required columns (exit code 1); missing values; non-numeric or fractional values in numeric/integer fields; domain violations; repeated `instant` identifiers; invalid dates; disagreement between date, month, and UCI year encoding; and `cnt != casual + registered`. A validation failure returns exit code 1; an unreadable or malformed input returns code 2.

Informational warnings report fully duplicated rows and unexpected extra columns. Duplicate IDs remain blocking errors. The validator does not enforce a fixed row count, repair values, or check temporal continuity.

