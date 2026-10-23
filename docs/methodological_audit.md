# Auditoria metodológica do UrbanPulse AI

**Data da auditoria:** 22/10/2026
**Escopo:** pipeline, validação, modelos, reports, simulador, dashboard, serialização e testes.
**Dados observados localmente:** 17.379 linhas, 18 colunas, de 01/01/2011 a 31/12/2012.

## Resumo executivo

As rotinas de baseline, Ridge, Random Forest, validação progressiva e Permutation Importance respeitam a ordem cronológica e não usam o teste final para ajustar os pipelines. A seleção explícita de features exclui `instant`, `casual`, `registered` e `cnt`; os dois estimadores usam pipelines do scikit-learn, ajustados em cada conjunto de treino. Os testes automatizados cobrem os principais caminhos de vazamento e os 70 testes passaram nesta auditoria.

Há, porém, pontos que precisam de decisão antes de apresentar os resultados como evidência de desempenho de produção. O holdout final já foi usado para comparar quatro abordagens e identificar o Random Forest como melhor, portanto deixou de ser uma avaliação final intocada para essa seleção. A avaliação também usa meteorologia observada no próprio horário previsto, informação que normalmente não está disponível em uma previsão operacional. Além disso, o simulador deriva `season` por trimestre, mas o CSV UCI tem transições de estação dentro de março, junho, setembro e dezembro. O artefato joblib não inclui verificação de versão nem manifesto de proveniência.

Não foi encontrado vazamento direto da variável-alvo para as features, nem vazamento entre treino e validação nas funções examinadas. A curva diária do simulador mantém as condições meteorológicas constantes e a interface comunica que é uma análise de cenários. Essas propriedades não resolvem as limitações de disponibilidade de clima e de calendário descritas abaixo.

## Pontos fortes

- `temporal_train_test_split()` ordena por `datetime`, usa divisão sequencial 80/20 e não embaralha os registros (`src/urbanpulse/ml/baseline.py`).
- Ridge e Random Forest selecionam uma lista fixa de 12 features de calendário e meteorologia. `datetime` é usado para ordenar e dividir, mas não entra nos estimadores (`src/urbanpulse/ml/ridge.py::select_features`, `src/urbanpulse/ml/random_forest.py::build_random_forest_pipeline`).
- `OneHotEncoder`, `StandardScaler` do Ridge e estimadores ficam em `Pipeline`; cada ajuste ocorre em `fit_predict_ridge()` ou `fit_predict_random_forest()` com o dataframe de treino. A validação temporal chama essas funções novamente para cada janela (`src/urbanpulse/ml/temporal_validation.py::evaluate_temporal_validation`).
- O baseline por hora calcula médias somente em `train`; para horas não observadas no treino, usa a média global do treino (`src/urbanpulse/ml/baseline.py::predict_training_baselines`).
- As janelas progressivas são blocos cronológicos não sobrepostos; o treino de cada janela termina antes da validação. Os testes verificam essa ordem e que mudar os alvos do último 20% não muda os resultados de validação e importância.
- A Permutation Importance é calculada em validações fora da amostra dentro dos primeiros 80%, com `neg_mean_absolute_error`, cinco repetições e semente fixa. O teste final não participa do cálculo (`src/urbanpulse/ml/interpretability.py::calculate_temporal_permutation_importance`).
- O simulador constrói exatamente as colunas de `FEATURE_COLUMNS`, converte temperatura, sensação térmica, umidade e vento para as escalas normalizadas usadas no treino, e limita estimativas negativas a zero (`src/urbanpulse/simulator.py`). A previsão de 24 horas usa uma chamada em lote, variando apenas `hr`.
- Os relatórios e a interface separam os resultados do teste final dos resultados das cinco janelas temporais. O dashboard verifica a estrutura e parte da consistência dos CSVs (`src/urbanpulse/dashboard_data.py`).
- O dashboard descreve a curva de 24 horas como cenário com condições meteorológicas constantes (`src/urbanpulse/dashboard.py::_show_simulator`).

## Problemas encontrados por gravidade

### Alta — O teste final já foi usado para comparar e selecionar modelos

`evaluate_final_test_models()` calcula as métricas dos quatro modelos sobre os mesmos últimos 20% (`src/urbanpulse/ml/model_comparison.py`). O relatório persistido `reports/model_comparison/final_test_metrics.csv` mostra Random Forest com MAE 56,4021, RMSE 84,6040 e R² 0,8528, melhor que Ridge e os dois baselines. O contexto e o dashboard identificam esse resultado como o melhor modelo atual. Assim, embora o código não ajuste parâmetros com esses dados, escolher o modelo vencedor depois de ver o holdout usa o teste para seleção. As métricas são válidas como comparação retrospectiva nesse conjunto, mas não devem ser divulgadas como estimativa imparcial de desempenho final do modelo escolhido.

**Recomendação:** congele a escolha do modelo e, antes de alegar desempenho final de publicação, avalie a configuração congelada em um período futuro ainda não consultado ou adote uma estratégia de validação externa. Se o novo teste não for viável, rotule explicitamente o número atual como holdout retrospectivo já consultado.

### Alta — Meteorologia observada dá ao modelo informação não disponível na previsão operacional

As 12 features incluem `weathersit`, `temp`, `atemp`, `hum` e `windspeed`. Na avaliação histórica, são valores observados para a mesma hora cujo `cnt` se prevê. `ridge.py`, `random_forest.py` e `temporal_validation.py` registram a suposição de que haveria previsão meteorológica disponível, mas a validação não usa previsões arquivadas: usa o clima observado no CSV. O erro publicado é, portanto, condicional a covariáveis meteorológicas conhecidas com antecedência perfeita; pode ser otimista para uso real.

**Recomendação:** antes de apresentar a métrica como desempenho operacional, testar com vintages de previsões meteorológicas que estariam disponíveis no instante da decisão. Até lá, rotular os resultados como avaliação retrospectiva condicionada ao clima observado e, idealmente, publicar também uma referência sem features meteorológicas.

### Média — `season` do simulador não reproduz as transições do dataset

`build_scenario_features()` define `season` por trimestre civil: `(month - 1) // 3 + 1` (`src/urbanpulse/simulator.py`). A inspeção do `data/raw/hour.csv` mostra que a UCI tem mais de uma estação em março, junho, setembro e dezembro. Exemplos observados: março contém `season` 1 e 2; junho 2 e 3; setembro 3 e 4; dezembro 1 e 4. Logo, parte das datas nesses quatro meses recebe no simulador um código diferente daquele usado no treino. O teste atual verifica 07/01/2012 e não cobre as fronteiras sazonais (`tests/test_simulator.py::ScenarioValidationTests`).

**Recomendação:** implementar a mesma regra de calendário usada na origem ou tornar `season` uma entrada explícita do cenário, documentando a escolha. Adicionar testes para datas próximas a cada transição e conferir os códigos contra os registros UCI. Corrigir esse mapeamento antes de apresentar o simulador como consistente com o modelo.

### Média — Validação e importância são divididas por quantidade de linhas, não por duração

As 17.379 datas horárias observadas são únicas e ordenadas, mas há 75 intervalos maiores que uma hora; a maior lacuna observada é de 1 dia e 13 horas. A divisão 80/20 e as janelas repartem linhas contíguas do conjunto disponível, não blocos de duração fixa (`baseline.py::temporal_train_test_split`, `temporal_validation.py::expanding_windows`). Isso é reproduzível e não embaralha o tempo, mas janelas com a mesma contagem podem cobrir durações ligeiramente diferentes e as lacunas não são avaliadas.

**Recomendação:** declarar que a partição é por número de observações e registrar os limites temporais (já presentes nos relatórios). Considerar uma validação de janelas por duração de calendário e um relatório de cobertura de horas ausentes, especialmente se os resultados forem generalizados para operação contínua.

### Média — Artefato joblib sem proveniência de versão e não confiável se não autenticado

`train_and_save_artifact()` persiste o pipeline inteiro por `joblib.dump()`; `load_simulator_model()` faz `joblib.load()` e verifica tipo, nomes de passos e schema, mas não valida versão de Python/scikit-learn/joblib, origem, hash ou metadados de treino (`src/urbanpulse/ml/train_artifact.py`, `src/urbanpulse/simulator.py`). O `pyproject.toml` declara limites mínimos (`scikit-learn>=1.4`, `joblib>=1.3`), sem lockfile. O ambiente usado nesta auditoria tem Python 3.12.10, pandas 3.0.6, scikit-learn 1.9.1 e joblib 1.6.0; isso não prova compatibilidade com outras versões. Joblib usa desserialização baseada em pickle: carregar arquivo não confiável pode executar código. Além disso, `/models/*` é ignorado pelo Git, portanto um clone limpo não recebe automaticamente o artefato do simulador.

**Recomendação:** fixar e registrar as versões usadas para treinamento e inferência, gerar um manifesto com hash do dataset/schema/parâmetros/versões e documentar o processo de obtenção do artefato. Carregar somente artefatos gerados ou verificados por fonte confiável. Para deploy, garantir que a etapa de build gere ou receba o modelo de forma reproduzível.

### Baixa — Os CSVs não carregam proveniência suficiente para provar que pertencem ao código e dados atuais

O dashboard confere modelo, split e contagens no relatório final. Para validação temporal, recalcula média e desvio padrão a partir das métricas por janela; para importância, confere as agregações contra as janelas. Porém, não recalcula os modelos nem comprova que os relatórios foram gerados a partir do Parquet, versão de código e dependências atuais. Também não valida integralmente se métricas por janela e análises de erro correspondem às previsões linha a linha. O relatório final contém datas de split, mas nenhum hash ou versão de dados/modelo.

Durante a auditoria, `reports/interpretability/permutation_importance.csv` já estava modificado no Git. A diferença observada é apenas de arredondamento em casas decimais e a tabela atual continua consistente com suas próprias agregações; o arquivo foi preservado, sem regeneração pela auditoria. Isso ilustra que a validação estrutural do dashboard não estabelece proveniência do relatório.

**Recomendação:** adicionar metadados de geração (hash do Parquet, versões, parâmetros, timestamp e revisão de código) e, quando possível, verificações que recalculam as métricas da tabela a partir das previsões salvas.

## Verificações específicas

### Vazamento e features

Não foi identificado vazamento direto de alvo nas features modeladas. `select_features()` usa uma allowlist com `season`, `yr`, `mnth`, `hr`, `holiday`, `weekday`, `workingday`, `weathersit`, `temp`, `atemp`, `hum` e `windspeed`; exclui implicitamente `instant`, `casual`, `registered`, `cnt` e `datetime`. `cnt` é alvo dos modelos supervisionados e entra nas médias de treino dos baselines. O teste `test_test_targets_and_user_counts_do_not_change_model_predictions` nos testes de Ridge e Random Forest confirma que alterar alvos e contagens de usuários do teste não muda previsões.

As métricas do conjunto final são calculadas depois da previsão, como esperado. O risco descrito acima é de seleção após observar o holdout, não contaminação das features ou ajuste do estimador nesse holdout.

### Calendário

- `yr`: a validação confere consistência com `dteday` (0 → 2011, 1 → 2012), e o simulador usa o mesmo mapeamento.
- `weekday`: o simulador converte o padrão Python (segunda=0) para UCI (domingo=0), alinhado ao campo bruto. A análise de erros usa nomes derivados de `datetime`, com segunda=0, e rotula-os explicitamente; não alimenta o modelo.
- `holiday` e `workingday`: o simulador garante que feriado marcado não seja dia útil e que fins de semana não sejam dias úteis. A checkbox de feriado é manual e não consulta um calendário oficial; o valor padrão é falso. O usuário precisa informar feriados corretamente.
- `season`: o trimestre civil não corresponde às fronteiras reais observadas em quatro meses, conforme problema de gravidade média acima.
- `mnth` e `hr`: derivados/validados a partir de data e hora e limitados aos domínios usados pelo dataset.

### Validação temporal e teste final

`evaluate_temporal_validation()` separa o último 20% antes de criar as cinco janelas nos primeiros 80%. O primeiro bloco de metade do período de desenvolvimento funciona como treino inicial; as janelas de validação cobrem a metade posterior dos 80% iniciais. Em todas as cinco janelas, o Random Forest tem MAE e R² melhores que Ridge nos CSVs persistidos; ainda assim, os valores variam entre janelas, então uma média não representa estabilidade uniforme.

O teste `test_final_twenty_percent_does_not_affect_validation_results` altera `cnt` e features do último 20% e verifica que métricas/previsões de validação não mudam. O teste de importância faz verificação análoga. `train_and_save_artifact()` ajusta o pipeline nos 80% iniciais. Essas garantias cobrem validação, importância e artefato; não desfazem o uso do holdout final para comparar/selecionar os quatro modelos.

### Permutation Importance

No relatório atual, as maiores médias são `hr` (aumento de MAE de aproximadamente 82,22 aluguéis/hora; desvio entre janelas 38,07), `workingday` (17,55; 11,61) e `hum` (11,80; 8,70). `atemp` tem média 4,34 e desvio 3,60; `temp`, 2,58 e 1,34. `mnth` tem média negativa (-0,55), o que significa que embaralhá-la reduziu ligeiramente o MAE médio observado, não que o mês tenha efeito causal negativo. O painel explica corretamente valores negativos como casos em que permutar reduziu o erro.

Temperatura e sensação térmica são correlacionadas; calendário também contém variáveis relacionadas (`mnth`/`season`, `weekday`/`workingday`/`holiday`). Com permutação individual, informação redundante pode permanecer nas variáveis correlacionadas, reduzindo a importância aparente de cada uma. O desvio reportado agrega variação entre cinco janelas (e, nas linhas de janela, cinco permutações); não é intervalo de confiança. As magnitudes diferem entre janelas, em especial para `hr`, `workingday` e variáveis meteorológicas, então a ordenação deve ser tratada como evidência preditiva condicionada a cada período, sem leitura causal.

### Métricas e gráficos no dashboard

O arquivo `reports/model_comparison/final_test_metrics.csv` contém os valores publicados de MAE, RMSE e R² para quatro modelos, todos com 13.903 linhas de treino e 3.476 linhas de teste (07/08/2012 12:00 a 31/12/2012 23:00). Os números do Random Forest correspondem ao contexto informado. O carregador confere que o relatório declara somente `final_test_20pct` e que os quatro modelos compartilham a mesma partição; não recalcula o CSV no momento da visualização.

Os relatórios temporais persistidos contêm cinco janelas por modelo e o carregador confere os resumos contra as linhas por janela. A rotina também produz as previsões fora da amostra e calcula os erros agrupados por hora e por dia da semana. O dashboard não treina modelos nem calcula Permutation Importance ao abrir. Os CSVs são artefatos gerados separadamente; sua correspondência ao código/Parquet atual depende de proveniência, recomendada acima.

No simulador, `predict_daily_demand()` prevê 24 registros em lote, limita valores negativos e o dashboard soma as 24 previsões estimadas. README e interface informam que os campos de clima ficam fixos ao longo do dia. Essa soma deve ser interpretada somente como total de um cenário sintético com clima constante, não como demanda diária observada ou previsão meteorológica horária.

## Lacunas de cobertura dos testes

Os 70 testes cobrem divisão cronológica, partições e ausência de influência do teste final em várias rotinas, cálculo de métricas, ajuste do encoder, consistência das agregações e schema/conversões do simulador. Lacunas que vale preencher antes da publicação:

- Datas próximas às transições sazonais UCI, especialmente março, junho, setembro e dezembro.
- Calendário oficial de feriados e equivalência de `holiday`/`workingday` para datas de feriado reais.
- Métricas da validação temporal recalculadas a partir das previsões salvas e consistência linha a linha de `error`, agregados por hora e weekday.
- Relatório final relacionado a hash/revisão do dataset e código, e verificação automatizada contra previsões congeladas.
- Compatibilidade de artefato em instalação limpa com versões travadas e comportamento explícito diante de incompatibilidade.
- Teste que verifique toda a curva diária, além do teste de consistência da hora selecionada, para provar que apenas `hr` varia entre 24 linhas.
- Impacto das lacunas temporais e escolha de janelas por duração, não apenas por contagem.
- Uso de covariáveis meteorológicas arquivadas como previsões disponíveis no momento da previsão.

## Recomendações priorizadas

1. **Antes do deploy:** não apresentar o holdout atual como teste final intocado; congelar a seleção e obter uma avaliação realmente não consultada, ou identificar o resultado como retrospectivo.
2. **Antes de alegar desempenho operacional:** reavaliar com previsões meteorológicas disponíveis no instante de decisão ou rotular claramente o resultado como condicionado ao clima observado.
3. **Antes de expor o simulador:** corrigir `season` para reproduzir a regra UCI e testar datas de fronteira; deixar explícito que feriados são informados manualmente.
4. **Antes de publicar artefato:** travar versões, fornecer manifesto/hash e definir como o artefato local ignorado pelo Git será obtido com segurança.
5. **Para aumentar a confiança nos relatórios:** incluir proveniência e verificações que relacionem previsões, métricas e agregações entre os CSVs.
6. **Para robustez da série:** documentar as 75 lacunas maiores que uma hora e comparar a divisão por linhas com janelas de duração fixa.
7. **Para interpretabilidade:** manter a descrição como importância preditiva, mostrar a variação entre janelas e evitar afirmações causais ou ranking estável sem evidência adicional.

## Checklist para publicação

- [ ] Identificar o holdout atual como já consultado e evitar chamá-lo de avaliação final imparcial.
- [ ] Definir uma avaliação posterior realmente reservada para a configuração escolhida.
- [ ] Documentar o cenário de disponibilidade de meteorologia e, para uso operacional, avaliar forecasts arquivados.
- [ ] Corrigir e testar a derivação de `season` e confirmar a codificação de feriado/dia útil no simulador.
- [ ] Manter a nota de que a curva diária assume o mesmo clima nas 24 horas e que o total é uma soma de cenário.
- [ ] Incluir versões de treino/inferência, hash do artefato e origem confiável no processo de distribuição do joblib.
- [ ] Garantir que o deploy gere/obtenha o modelo, já que `models/` está ignorado pelo Git.
- [ ] Versionar proveniência para os CSVs de teste, validação e importância e verificar sua consistência com o Parquet usado.
- [ ] Preservar as distinções visuais entre teste final e janelas de validação temporal.
- [ ] Executar a suíte automatizada e `git diff --check` na revisão candidata à publicação.

## Execução desta auditoria

- Suíte: `.venv\Scripts\python.exe -m unittest discover -s tests -v` — **70 testes passaram** em 30,7 segundos na última execução.
- Integridade do patch: `git diff --check` — sem erros de whitespace no diff verificado.
- Foram lidos os CSVs de relatório e o CSV bruto local para conferir métricas, variáveis de importância e mapeamento mensal de `season`. Nenhum CSV ou Parquet foi alterado pela auditoria. O CSV de importância já aparecia modificado no início; foi preservado.

## Atualização após correções metodológicas

**Atualizado em 22/10/2026.** As alterações abaixo corrigem parte dos achados; não constituem nova avaliação de desempenho e não recalculam métricas do modelo.

### Corrigido: calendário do simulador para os anos cobertos pela fonte

`src/urbanpulse/simulator.py::uci_season_for_date()` agora usa as transições de rótulo observadas diretamente em `data/raw/hour.csv`: 21/03, 21/06, 23/09 e 21/12, iguais em 2011 e 2012. Não interpreta estação como trimestre. `uci_holiday_for_date()` usa as 21 datas distintas marcadas com `holiday=1` no arquivo. O padrão do simulador é agora o calendário observado; uma opção explícita pode forçar ou retirar o feriado somente para compor um cenário. `weekday` continua no padrão UCI (domingo=0), `yr` e `mnth` vêm da data e `workingday` vale 1 apenas de segunda a sexta sem feriado.

Uma comparação somente de leitura entre os campos derivados e as 17.379 linhas do CSV, agrupadas nas 731 datas, encontrou **zero datas com divergência** em `season`, `yr`, `mnth`, `weekday`, `holiday` ou `workingday`. Os testes `test_season_codes_follow_observed_uci_transition_dates` e `test_calendar_fields_match_dataset_encoding_and_holiday_lookup` cobrem limites e valores de calendário. A regra fica verificada apenas para 2011–2012; não é um calendário geral para outros anos e o simulador continua restringindo a data a esse intervalo.

### Corrigido: manifesto e verificações do artefato

`src/urbanpulse/ml/train_artifact.py::train_and_save_artifact()` grava um JSON ao lado do joblib, com versões exatas de Python, scikit-learn, pandas e joblib; features e ordem; linhas totais e de treino; período de treino; parâmetros fixos; nome e SHA-256 do Parquet de entrada; e SHA-256 do artefato. O processo continua treinando apenas nos primeiros 80% e não consulta os alvos do holdout para ajuste.

`src/urbanpulse/simulator.py::load_simulator_model()` exige o manifesto, verifica versões exatas antes da desserialização, confere hash do joblib, schema e parâmetros do estimador. Alterar ambiente ou artefato requer regeneração. O hash junto ao JSON detecta corrupção acidental, mas não autentica a origem se um agente malicioso puder substituir os dois arquivos. Joblib continua podendo executar código durante a carga: carregar somente artefatos locais confiáveis. O artefato local foi regenerado com `urbanpulse-train-random-forest`, usando 13.903 linhas de treino até 07/08/2012 11:00; o teste final não foi usado no ajuste. Essa operação não reavaliou modelos nem alterou o CSV de métricas.

### Atualizado: comunicação no README e dashboard

O README e a área de comparação agora dizem que as métricas dos últimos 20% são retrospectivas e que o holdout já foi consultado durante a comparação de modelos. A área de validação identifica as cinco janelas nos primeiros 80% e esclarece que usa meteorologia observada. O simulador informa que produz cenários, não previsão operacional em tempo real, descreve calendário automático e overrides. O README documenta o JSON e alerta para uso de joblib somente de fonte confiável.

### Limitações ainda abertas

- **Meteorologia futura:** não há forecasts arquivados no dataset nem avaliação com vintages meteorológicos. Avisos, sozinhos, não resolvem esse limite; as métricas não demonstram desempenho operacional.
- **Holdout consultado:** os últimos 20% foram usados para comparar/selecionar entre modelos. As métricas e a seleção histórica foram preservadas conforme solicitado; não são uma avaliação independente do vencedor. Uma futura avaliação imparcial exige período ainda não consultado ou protocolo externo, sem reusar este holdout para novas decisões.
- **Compatibilidade de distribuição:** o manifesto permite rejeitar versões divergentes, mas o projeto ainda não possui lockfile/ambiente de deploy fixado. As versões mínimas do `pyproject.toml` não bastam para reproduzir o byte idêntico do artefato entre ambientes.
- **Confiança do joblib:** os hashes não substituem assinatura/proveniência externa. O artefato e o JSON devem vir de uma origem controlada.
- **Partição por linhas e lacunas temporais:** permanecem as 75 lacunas maiores que uma hora e a validação continua dividindo por contagem de linhas, como descrito acima.
- **Override de feriado:** o valor automático reproduz a flag UCI para datas históricas; um override intencional cria cenário contrafactual e não deve ser descrito como observação de calendário.

As métricas existentes dos relatórios não foram regeneradas. Nenhum CSV ou Parquet foi alterado nesta correção.

### Verificações desta atualização

- Comparação somente de leitura do calendário derivado com os campos UCI em 731 datas: zero divergências em `season`, `yr`, `mnth`, `weekday`, `holiday` e `workingday` (17.379 linhas de origem).
- `tests/test_simulator.py`: 14 testes passaram, incluindo paridade de calendário em todas as linhas do CSV local, fronteiras sazonais, feriado automático/override, manifesto, incompatibilidade de versão e hash de artefato adulterado.
- Suíte completa: **75 testes passaram** em 33,4 segundos.
- Streamlit `AppTest`: visão geral, comparação de modelos, validação temporal e simulador renderizaram sem exceções.
- Nenhum comando de avaliação ou interpretação foi executado; os resultados publicados não foram recalculados. O joblib local foi regenerado pelo processo de artefato, exclusivamente com os 80% iniciais, para gravar e verificar o novo manifesto.
