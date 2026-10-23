"""Local Streamlit dashboard for UrbanPulse AI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from plotly.graph_objects import Figure
from streamlit.errors import StreamlitSecretNotFoundError

from urbanpulse.dashboard_data import (
    DEFAULT_DATA_PATH,
    DEFAULT_FINAL_METRICS_PATH,
    DEFAULT_IMPORTANCE_REPORT_PATH,
    DEFAULT_TEMPORAL_REPORT_DIR,
    WEEKDAY_NAMES_PT,
    filter_dashboard_data,
    load_dashboard_data,
    load_final_model_metrics,
    load_permutation_importance_report,
    load_temporal_validation_reports,
)

st.set_page_config(
    page_title="UrbanPulse AI | Demanda de bicicletas",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#0F9D8A"
BLUE = "#4361A8"
MONTH_LABELS = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
MODEL_LABELS = {
    "training_mean": "Média global do treino",
    "training_hour_mean": "Média por hora do treino",
    "ridge": "Ridge Regression",
    "random_forest": "Random Forest",
}
TEMPORAL_MODEL_ORDER = ["Ridge", "Random Forest"]
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "random_forest.joblib"
WEATHER_SITUATIONS = {
    "1 · Céu limpo ou poucas nuvens": 1,
    "2 · Névoa ou céu parcialmente nublado": 2,
    "3 · Chuva ou neve leve": 3,
    "4 · Chuva forte, neve ou neblina intensa": 4,
}


@st.cache_data(show_spinner="Carregando os dados processados…", ttl="10m")
def load_data(path: str, modified_time_ns: int) -> pd.DataFrame:
    """Cache the prepared Parquet by path and modification time."""
    _ = modified_time_ns
    return load_dashboard_data(Path(path))


@st.cache_data(show_spinner="Carregando métricas do teste final…", ttl="10m")
def load_final_metrics(path: str, modified_time_ns: int) -> pd.DataFrame:
    """Cache the reproducible final-test metrics report."""
    _ = modified_time_ns
    return load_final_model_metrics(Path(path))


@st.cache_data(show_spinner="Carregando importância das variáveis…", ttl="10m")
def load_importance_report(path: str, modified_time_ns: int) -> pd.DataFrame:
    """Cache the precomputed CSV report; the dashboard does not calculate importance."""
    _ = modified_time_ns
    return load_permutation_importance_report(Path(path))


def _temporal_report_cache_key(path: Path) -> tuple[tuple[str, int], ...]:
    filenames = (
        "window_metrics.csv",
        "summary_metrics.csv",
        "out_of_sample_predictions.csv",
        "error_by_hour.csv",
        "error_by_weekday.csv",
    )
    return tuple(
        (filename, (path / filename).stat().st_mtime_ns)
        for filename in filenames
        if (path / filename).is_file()
    )


@st.cache_data(show_spinner="Carregando relatórios temporais…", ttl="10m")
def load_temporal_reports(
    path: str,
    modified_times: tuple[tuple[str, int], ...],
) -> dict[str, pd.DataFrame]:
    """Cache the existing temporal validation report tables."""
    _ = modified_times
    return load_temporal_validation_reports(Path(path))


@st.cache_resource(show_spinner="Carregando o modelo salvo…")
def load_cached_simulator_model(
    path: str,
    modified_time_ns: int,
    metadata_modified_time_ns: int,
):
    """Cache the fitted model artifact; this function never trains a model."""
    _ = modified_time_ns, metadata_modified_time_ns
    from urbanpulse.simulator import load_simulator_model

    return load_simulator_model(Path(path))


@st.cache_resource(show_spinner="Preparando recursos do dashboard…")
def prepare_deployment_resources(
    model_url: str | None,
    model_sha256: str | None,
    metadata_url: str | None,
    metadata_sha256: str | None,
):
    """Prepare missing deployment resources once per Streamlit process."""
    from urbanpulse.cloud_resources import prepare_cloud_resources

    return prepare_cloud_resources(
        model_url=model_url,
        model_sha256=model_sha256,
        metadata_url=metadata_url,
        metadata_sha256=metadata_sha256,
    )


def _style_figure(figure: Figure, *, height: int | None = None) -> Figure:
    figure.update_layout(
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 64, "b": 28},
        font={"family": "Arial, sans-serif", "color": "#203442", "size": 13},
        title={"x": 0.02, "xanchor": "left", "font": {"size": 18}},
        legend_title_text="",
        height=height,
        autosize=True,
    )
    figure.update_xaxes(automargin=True, gridcolor="#E8EEF1", zerolinecolor="#C8D5DB")
    figure.update_yaxes(automargin=True, gridcolor="#E8EEF1", zerolinecolor="#C8D5DB")
    return figure


def _format_integer_pt(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def _format_decimal_pt(value: float) -> str:
    return f"{value:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _show_header(subtitle: str) -> None:
    with st.container(border=True):
        brand, source = st.columns([4, 1], vertical_alignment="center")
        with brand:
            st.caption("MOBILIDADE URBANA · ANÁLISE DE DADOS")
            st.title("UrbanPulse AI")
            st.write(subtitle)
        with source:
            st.link_button(
                "Dataset oficial UCI",
                "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset",
                use_container_width=True,
            )
            st.caption("Bike Sharing Dataset · demanda horária")


def _show_metric_cards(items: list[tuple[str, str, str | None]]) -> None:
    """Render consistently sized native metric cards that can wrap on narrow screens."""
    columns = st.columns(len(items))
    for column, (label, value, help_text) in zip(columns, items):
        with column:
            with st.container(border=True):
                st.metric(label, value, help=help_text)


def _show_overview() -> None:
    data_path = DEFAULT_DATA_PATH
    st.sidebar.markdown("## Filtros")
    st.sidebar.caption("Os filtros são aplicados aos indicadores e a todos os gráficos desta área.")
    year_choice = st.sidebar.selectbox("Ano", ["Ambos", "2011", "2012"], index=0)
    day_choice = st.sidebar.selectbox(
        "Tipo de dia",
        ["Todos", "Dias úteis", "Dias não úteis (fim de semana/feriado)"],
        index=0,
    )
    st.sidebar.caption(
        "Classificação conforme workingday da UCI: 1 indica dia útil; 0 inclui fins de semana e feriados."
    )

    if not data_path.is_file():
        st.warning("O arquivo processado ainda não existe. Gere-o primeiro com o comando abaixo:")
        st.code("urbanpulse-prepare", language="bash")
        return
    try:
        data = load_data(str(data_path), data_path.stat().st_mtime_ns)
    except (OSError, ValueError) as exc:
        st.error(f"Não foi possível carregar o dataset processado: {exc}")
        return

    year_filter = None if year_choice == "Ambos" else int(year_choice)
    workingday_filter = {
        "Todos": None,
        "Dias úteis": True,
        "Dias não úteis (fim de semana/feriado)": False,
    }[day_choice]
    filtered = filter_dashboard_data(data, year=year_filter, workingday=workingday_filter)
    if filtered.empty:
        st.info("Não há registros para a combinação de filtros selecionada.")
        return

    st.caption(
        f"Exibindo {len(filtered):,} de {len(data):,} intervalos horários · "
        f"{filtered['datetime'].min():%d/%m/%Y} — {filtered['datetime'].max():%d/%m/%Y}"
    )
    _show_metric_cards([
        ("Total de aluguéis", _format_integer_pt(int(filtered["cnt"].sum())), "Soma dos aluguéis no período filtrado."),
        ("Média por hora", _format_decimal_pt(float(filtered["cnt"].mean())), "Média dos registros horários filtrados."),
        ("Mediana por hora", _format_integer_pt(int(filtered["cnt"].median())), "Metade dos intervalos tem valor menor ou igual a este."),
        ("Máximo em uma hora", _format_integer_pt(int(filtered["cnt"].max())), "Maior valor observado em um intervalo filtrado."),
    ])

    st.markdown("### Ritmos de uso")
    left, right = st.columns(2)
    with left:
        hourly = filtered.groupby("hr", as_index=False)["cnt"].mean()
        hourly_figure = px.line(
            hourly,
            x="hr",
            y="cnt",
            markers=True,
            labels={"hr": "Hora do dia", "cnt": "Aluguéis médios por hora"},
            title="Demanda média por hora do dia",
        )
        hourly_figure.update_xaxes(dtick=1)
        st.plotly_chart(_style_figure(hourly_figure), width="stretch")
    with right:
        weekday_data = filtered.assign(
            _weekday=filtered["datetime"].dt.dayofweek.map(WEEKDAY_NAMES_PT)
        )
        weekday_mean = (
            weekday_data.groupby("_weekday", as_index=False)["cnt"].mean()
            .rename(columns={"_weekday": "Dia da semana"})
        )
        weekday_figure = px.bar(
            weekday_mean,
            x="Dia da semana",
            y="cnt",
            category_orders={"Dia da semana": list(WEEKDAY_NAMES_PT.values())},
            labels={"cnt": "Aluguéis médios por hora"},
            title="Demanda média por dia da semana",
            color_discrete_sequence=[ACCENT],
        )
        st.plotly_chart(_style_figure(weekday_figure), width="stretch")

    st.markdown("### Sazonalidade e contexto")
    monthly_data = filtered.assign(
        Ano=filtered["datetime"].dt.year.astype(str),
        _month=filtered["datetime"].dt.month,
    )
    monthly_mean = (
        monthly_data.groupby(["_month", "Ano"], as_index=False)["cnt"].mean()
        .rename(columns={"_month": "Mês"})
    )
    monthly_figure = px.line(
        monthly_mean,
        x="Mês",
        y="cnt",
        color="Ano",
        markers=True,
        labels={"cnt": "Aluguéis médios por hora"},
        title="Demanda média mensal por ano",
        color_discrete_sequence=[ACCENT, BLUE],
    )
    monthly_figure.update_xaxes(tickmode="array", tickvals=list(range(1, 13)), ticktext=MONTH_LABELS)
    st.plotly_chart(_style_figure(monthly_figure), width="stretch")

    left, right = st.columns(2)
    with left:
        temperature_figure = px.scatter(
            filtered,
            x="temp",
            y="cnt",
            opacity=0.45,
            labels={
                "temp": "Temperatura normalizada (escala da UCI)",
                "cnt": "Aluguéis no intervalo de 1 hora",
            },
            hover_data={"datetime": True, "weathersit": True},
            title="Temperatura normalizada e demanda",
            color_discrete_sequence=[ACCENT],
        )
        st.plotly_chart(_style_figure(temperature_figure), width="stretch")
    with right:
        distribution_figure = px.histogram(
            filtered,
            x="cnt",
            nbins=40,
            labels={
                "cnt": "Aluguéis no intervalo de 1 hora",
                "count": "Número de intervalos horários",
            },
            title="Distribuição da demanda horária",
            color_discrete_sequence=[BLUE],
        )
        st.plotly_chart(_style_figure(distribution_figure), width="stretch")

    st.caption(
        "Fonte: UCI Bike Sharing Dataset. Visualizações descritivas; associação não implica causalidade. "
        "A temperatura está normalizada conforme a documentação da fonte."
    )


def _show_metric_explanations() -> None:
    with st.expander("Como interpretar as métricas e os baselines"):
        st.markdown(
            """
            - **MAE:** erro absoluto médio em aluguéis por hora; menor é melhor e a unidade é intuitiva.
            - **RMSE:** penaliza mais os erros grandes; também está em aluguéis por hora.
            - **R²:** compara o modelo com uma referência que prevê a média; quanto mais perto de 1, melhor.
              Pode ser negativo quando o modelo erra mais que essa referência.
            - **Baseline:** previsão simples usada como comparação. A média global prevê sempre a média do treino;
              a média por hora usa apenas a média histórica daquela hora, calculada no treino.
            """
        )


def _show_model_comparison() -> None:
    st.markdown("## Modelos de Machine Learning")
    st.info(
        "Avaliação retrospectiva nos últimos 20% da série cronológica. Este teste já foi "
        "consultado na comparação entre modelos e não é uma avaliação independente para novas "
        "decisões de modelagem. Os dados históricos também usam clima observado na própria hora "
        "prevista, não previsões meteorológicas arquivadas."
    )
    path = DEFAULT_FINAL_METRICS_PATH
    if not path.is_file():
        st.warning("O relatório de métricas do teste final ainda não existe.")
        st.code("urbanpulse-model-comparison", language="bash")
        return
    try:
        metrics = load_final_metrics(str(path), path.stat().st_mtime_ns)
    except (OSError, ValueError) as exc:
        st.error(f"Não foi possível carregar as métricas do teste final: {exc}")
        return

    first_row = metrics.iloc[0]
    st.caption(
        f"Treino: {int(first_row['train_rows']):,} linhas · Teste final: {int(first_row['test_rows']):,} linhas · "
        f"{pd.Timestamp(first_row['test_start']):%d/%m/%Y %H:%M} — "
        f"{pd.Timestamp(first_row['test_end']):%d/%m/%Y %H:%M}"
    )
    display = metrics.copy()
    display["Modelo"] = display["model"].map(MODEL_LABELS)
    display = display[["Modelo", "mae", "rmse", "r2"]].rename(
        columns={"mae": "MAE", "rmse": "RMSE", "r2": "R²"}
    )
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={
            "MAE": st.column_config.NumberColumn(format="%.2f"),
            "RMSE": st.column_config.NumberColumn(format="%.2f"),
            "R²": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    selected_metric = st.selectbox(
        "Métrica para comparar",
        ["mae", "rmse", "r2"],
        format_func={"mae": "MAE", "rmse": "RMSE", "r2": "R²"}.__getitem__,
        key="final_metric_selector",
    )
    chart_data = metrics.assign(Modelo=metrics["model"].map(MODEL_LABELS))
    metric_labels = {
        "mae": "Erro absoluto médio (aluguéis/hora)",
        "rmse": "Raiz do erro quadrático médio (aluguéis/hora)",
        "r2": "R²",
    }
    figure = px.bar(
        chart_data,
        x="Modelo",
        y=selected_metric,
        color="Modelo",
        text_auto=".3f",
        labels={selected_metric: metric_labels[selected_metric]},
        title=f"Comparação no teste final: {metric_labels[selected_metric]}",
        color_discrete_sequence=[ACCENT, BLUE, "#E5A93D", "#B45D72"],
    )
    st.plotly_chart(_style_figure(figure), width="stretch")
    _show_metric_explanations()
    st.caption(
        "Os modelos foram ajustados nos 80% iniciais e comparados neste holdout já consultado. "
        "As métricas são retrospectivas e condicionadas às observações meteorológicas históricas."
    )


def _show_temporal_validation() -> None:
    st.markdown("## Validação temporal")
    st.info(
        "Resultados fora da amostra dentro dos 80% iniciais. O teste final de 20% não participa "
        "destas janelas e seus resultados não são misturados aqui. Esta validação usa as condições "
        "meteorológicas observadas no dataset para cada hora; não representa desempenho com "
        "previsões meteorológicas disponíveis em tempo real."
    )
    st.markdown(
        "Em cada janela, o modelo aprende com o passado e é avaliado no próximo bloco de tempo. "
        "O treino cresce a cada rodada. O resumo mostra o desempenho médio e a variação entre os períodos."
    )
    report_dir = DEFAULT_TEMPORAL_REPORT_DIR
    try:
        reports = load_temporal_reports(
            str(report_dir),
            _temporal_report_cache_key(report_dir),
        )
    except FileNotFoundError as exc:
        st.warning("Os relatórios de validação temporal ainda não estão disponíveis.")
        st.code("urbanpulse-temporal-validation", language="bash")
        st.caption(str(exc))
        return
    except (OSError, ValueError) as exc:
        st.error(f"Não foi possível carregar os relatórios temporais: {exc}")
        return

    summary = reports["summary_metrics"].copy()
    ridge_mae = reports["window_metrics"].loc[
        reports["window_metrics"]["model"] == "Ridge", ["window", "mae"]
    ].set_index("window")["mae"]
    forest_mae = reports["window_metrics"].loc[
        reports["window_metrics"]["model"] == "Random Forest", ["window", "mae"]
    ].set_index("window")["mae"]
    shared_windows = ridge_mae.index.intersection(forest_mae.index)
    forest_wins = int((forest_mae.loc[shared_windows] < ridge_mae.loc[shared_windows]).sum())
    _show_metric_cards([
        ("Janelas com menor MAE · Random Forest", f"{forest_wins}/{len(shared_windows)}", "Comparação derivada das métricas já salvas por janela."),
        ("Períodos avaliados", str(len(shared_windows)), "Validação progressiva somente nos 80% iniciais."),
    ])
    summary["Modelo"] = summary["model"]
    summary = summary[["Modelo", "mae_mean", "mae_std", "rmse_mean", "rmse_std", "r2_mean", "r2_std"]]
    summary = summary.rename(
        columns={
            "mae_mean": "MAE médio",
            "mae_std": "Desvio MAE",
            "rmse_mean": "RMSE médio",
            "rmse_std": "Desvio RMSE",
            "r2_mean": "R² médio",
            "r2_std": "Desvio R²",
        }
    )
    st.markdown("### Resumo nas cinco janelas")
    st.dataframe(summary, hide_index=True, width="stretch")
    st.caption("O desvio padrão mostra quanto as métricas variam entre períodos; menor indica maior estabilidade.")

    window_metrics = reports["window_metrics"]
    selected_metric = st.selectbox(
        "Métrica por janela",
        ["mae", "rmse", "r2"],
        format_func={"mae": "MAE", "rmse": "RMSE", "r2": "R²"}.__getitem__,
        key="temporal_metric_selector",
    )
    metric_labels = {"mae": "MAE (aluguéis/hora)", "rmse": "RMSE (aluguéis/hora)", "r2": "R²"}
    figure = px.line(
        window_metrics,
        x="window",
        y=selected_metric,
        color="model",
        markers=True,
        category_orders={"model": TEMPORAL_MODEL_ORDER},
        labels={"window": "Janela progressiva", selected_metric: metric_labels[selected_metric]},
        title=f"{metric_labels[selected_metric]} por janela de validação",
        color_discrete_sequence=[ACCENT, BLUE],
    )
    figure.update_xaxes(dtick=1)
    st.plotly_chart(_style_figure(figure), width="stretch")
    st.dataframe(
        window_metrics[
            [
                "window", "model", "train_rows", "validation_rows",
                "validation_start", "validation_end", "mae", "rmse", "r2",
            ]
        ].rename(
            columns={
                "window": "Janela",
                "model": "Modelo",
                "train_rows": "Linhas treino",
                "validation_rows": "Linhas validação",
                "validation_start": "Início validação",
                "validation_end": "Fim validação",
                "mae": "MAE",
                "rmse": "RMSE",
                "r2": "R²",
            }
        ),
        hide_index=True,
        width="stretch",
    )

    predictions = reports["predictions"]
    st.markdown("### Valores reais e previstos")
    scatter = px.scatter(
        predictions,
        x="actual",
        y="predicted",
        color="model",
        opacity=0.38,
        category_orders={"model": TEMPORAL_MODEL_ORDER},
        labels={
            "actual": "Aluguéis reais por hora",
            "predicted": "Aluguéis previstos por hora",
            "model": "Modelo",
        },
        title="Previsões fora da amostra nas janelas temporais",
        color_discrete_sequence=[ACCENT, BLUE],
        hover_data=["datetime", "window"],
    )
    axis_min = min(predictions["actual"].min(), predictions["predicted"].min())
    axis_max = max(predictions["actual"].max(), predictions["predicted"].max())
    scatter.add_shape(
        type="line",
        x0=axis_min,
        y0=axis_min,
        x1=axis_max,
        y1=axis_max,
        line={"color": "#5B6573", "dash": "dash"},
    )
    st.plotly_chart(_style_figure(scatter), width="stretch")
    st.caption("Pontos próximos à diagonal indicam previsões próximas dos valores observados.")

    st.markdown("### Padrões dos erros")
    left, right = st.columns(2)
    with left:
        hourly = reports["errors_by_hour"]
        hour_figure = px.line(
            hourly,
            x="hour",
            y="mae",
            color="model",
            markers=True,
            category_orders={"model": TEMPORAL_MODEL_ORDER},
            labels={"hour": "Hora do dia", "mae": "MAE (aluguéis/hora)", "model": "Modelo"},
            title="Erro absoluto médio por hora",
            color_discrete_sequence=[ACCENT, BLUE],
        )
        hour_figure.update_xaxes(dtick=1)
        st.plotly_chart(_style_figure(hour_figure), width="stretch")
    with right:
        weekday = reports["errors_by_weekday"].copy()
        weekday["Dia da semana"] = weekday["weekday"].map(
            {index: name for index, name in enumerate(WEEKDAY_NAMES_PT.values())}
        )
        weekday_figure = px.bar(
            weekday,
            x="Dia da semana",
            y="mae",
            color="model",
            barmode="group",
            category_orders={
                "Dia da semana": list(WEEKDAY_NAMES_PT.values()),
                "model": TEMPORAL_MODEL_ORDER,
            },
            labels={"mae": "MAE (aluguéis/hora)", "model": "Modelo"},
            title="Erro absoluto médio por dia da semana",
            color_discrete_sequence=[ACCENT, BLUE],
        )
        st.plotly_chart(_style_figure(weekday_figure), width="stretch")

    errors_figure = px.histogram(
        predictions,
        x="error",
        color="model",
        barmode="overlay",
        opacity=0.55,
        nbins=45,
        category_orders={"model": TEMPORAL_MODEL_ORDER},
        labels={"error": "Erro (previsto − real), aluguéis/hora", "model": "Modelo"},
        title="Distribuição dos erros fora da amostra",
        color_discrete_sequence=[ACCENT, BLUE],
    )
    errors_figure.add_vline(x=0, line_dash="dash", line_color="#5B6573")
    st.plotly_chart(_style_figure(errors_figure), width="stretch")
    st.caption(
        "Erro negativo significa subestimação; erro positivo significa superestimação. "
        "As features meteorológicas pressupõem previsões disponíveis para cada hora futura."
    )
    _show_metric_explanations()


def _show_simulator() -> None:
    st.markdown("## Simulador de demanda")
    st.markdown(
        "Configure uma hora e condições informadas para estimar a quantidade de aluguéis "
        "com o Random Forest já treinado."
    )
    if not DEFAULT_MODEL_PATH.is_file():
        st.warning("O artefato do simulador não está disponível neste ambiente.")
        st.markdown(
            "Localmente, gere-o com as dependências de ML. No Community Cloud, configure as URLs "
            "e os hashes SHA-256 dos arquivos confiáveis conforme a seção de publicação do README."
        )
        st.code('python -m pip install -e ".[ml,dashboard]"\nurbanpulse-train-random-forest', language="bash")
        return
    metadata_path = DEFAULT_MODEL_PATH.with_name(f"{DEFAULT_MODEL_PATH.name}.metadata.json")
    if not metadata_path.is_file():
        st.warning("O artefato não tem o arquivo de metadados compatível exigido por esta versão.")
        st.code("urbanpulse-train-random-forest", language="bash")
        return

    st.info(
        "Esta é uma simulação baseada em padrões históricos e nas condições informadas. "
        "Não representa uma previsão operacional de demanda em tempo real e não consulta "
        "previsões meteorológicas. No gráfico diário, "
        "as condições meteorológicas informadas permanecem constantes nas 24 horas; trata-se "
        "de uma comparação de cenários, não de uma previsão do tempo."
    )
    with st.form("demand_simulator_form"):
        date_col, hour_col = st.columns(2)
        with date_col:
            scenario_date = st.date_input(
                "Data do cenário (modelo treinado com 2011–2012)",
                value=pd.Timestamp("2012-06-15").date(),
                min_value=pd.Timestamp("2011-01-01").date(),
                max_value=pd.Timestamp("2012-12-31").date(),
            )
            holiday_choice = st.selectbox(
                "Feriado",
                ["Automático (calendário observado UCI)", "Forçar feriado", "Forçar dia comum"],
            )
        with hour_col:
            hour = st.slider("Hora do dia", min_value=0, max_value=23, value=8, format="%d:00")
            weather_label = st.selectbox("Condição do tempo (categoria UCI)", list(WEATHER_SITUATIONS))

        st.markdown("### Condições meteorológicas")
        temp_col, atemp_col = st.columns(2)
        with temp_col:
            temperature = st.slider("Temperatura do ar (°C)", -8.0, 39.0, 20.0, 0.5)
            humidity = st.slider("Umidade relativa (%)", 0.0, 100.0, 60.0, 1.0)
        with atemp_col:
            apparent_temperature = st.slider("Sensação térmica (°C)", -16.0, 50.0, 22.0, 0.5)
            wind = st.slider("Velocidade do vento (escala original UCI)", 0.0, 67.0, 15.0, 1.0)
        submitted = st.form_submit_button("Simular demanda", type="primary")

    with st.expander("Como as unidades são representadas no modelo"):
        st.markdown(
            "A UCI normaliza a temperatura como `(°C + 8) / 47`, a sensação térmica como "
            "`(°C + 16) / 66`, a umidade como `% / 100` e o vento como `escala UCI / 67`. "
            "A documentação da UCI não especifica unidade física para o vento; por isso o controle "
            "usa a escala anterior à normalização. Estação usa as transições de data observadas no "
            "hour.csv (21/03, 21/06, 23/09 e 21/12); ano, mês e dia da semana são derivados da data. "
            "No modo automático, feriado usa as 21 datas marcadas pela UCI em 2011–2012; você pode "
            "sobrescrever essa flag apenas para criar um cenário. Dia útil é calculado a partir de "
            "fim de semana e feriado."
        )

    if not submitted:
        return
    try:
        from urbanpulse.simulator import build_daily_scenario_features, predict_daily_demand

        model = load_cached_simulator_model(
            str(DEFAULT_MODEL_PATH),
            DEFAULT_MODEL_PATH.stat().st_mtime_ns,
            metadata_path.stat().st_mtime_ns,
        )
        features = build_daily_scenario_features(
            scenario_date,
            temperature_c=temperature,
            apparent_temperature_c=apparent_temperature,
            humidity_percent=humidity,
            wind_speed_dataset_units=wind,
            weathersit=WEATHER_SITUATIONS[weather_label],
            holiday={
                "Automático (calendário observado UCI)": None,
                "Forçar feriado": True,
                "Forçar dia comum": False,
            }[holiday_choice],
        )
        hourly_predictions = predict_daily_demand(model, features)
    except (OSError, ValueError) as exc:
        st.error(f"Não foi possível gerar a estimativa: {exc}")
        return
    selected = hourly_predictions.loc[hourly_predictions["hr"] == hour].iloc[0]
    busiest = hourly_predictions.loc[hourly_predictions["demand_estimated"].idxmax()]
    quietest = hourly_predictions.loc[hourly_predictions["demand_estimated"].idxmin()]
    _show_metric_cards([
        ("Estimativa no horário selecionado", _format_decimal_pt(float(selected["demand_estimated"])), "Estimativa individual para a hora escolhida."),
        ("Pico estimado", f"{int(busiest['hr']):02d}:00 · {_format_decimal_pt(float(busiest['demand_estimated']))}", "Maior valor entre as 24 horas simuladas."),
        ("Menor demanda estimada", f"{int(quietest['hr']):02d}:00 · {_format_decimal_pt(float(quietest['demand_estimated']))}", "Menor valor entre as 24 horas simuladas."),
        ("Total estimado · 24 horas", _format_integer_pt(round(float(hourly_predictions["demand_estimated"].sum()))), "Soma das estimativas horárias; resultado arredondado apenas para exibição."),
    ])

    hourly_predictions = hourly_predictions.assign(
        hour_label=hourly_predictions["hr"].map(lambda value: f"{value:02d}:00")
    )
    figure = px.line(
        hourly_predictions,
        x="hour_label",
        y="demand_estimated",
        markers=True,
        labels={"hour_label": "Horário", "demand_estimated": "Aluguéis estimados na hora"},
        title="Demanda estimada ao longo das 24 horas",
    )
    figure.add_scatter(
        x=[f"{hour:02d}:00"],
        y=[float(selected["demand_estimated"])],
        mode="markers+text",
        name="Horário selecionado",
        text=["Selecionado"],
        textposition="top center",
        marker={"size": 14, "color": "#E05A47", "symbol": "star"},
    )
    figure.update_xaxes(categoryorder="array", categoryarray=hourly_predictions["hour_label"].tolist())
    st.plotly_chart(_style_figure(figure), width="stretch")
    st.caption("A previsão individual acima corresponde ao horário selecionado no gráfico. O total diário é a soma das 24 estimativas, arredondada para exibição.")


def _show_interpretability() -> None:
    st.markdown("## Interpretabilidade do Random Forest")
    st.info(
        "Resultados calculados nas cinco janelas de validação temporal dentro dos 80% iniciais. "
        "Os 20% finais permanecem reservados e não são usados nesta análise."
    )
    with st.expander("O que mostra a Permutation Importance?"):
        st.markdown(
            "Para cada variável, seu valor é embaralhado na validação e mede-se quanto o MAE aumenta "
            "em relação à previsão original. Valores maiores indicam maior contribuição preditiva nessa "
            "avaliação; valores negativos significam que a permutação reduziu o MAE. A média agregada "
            "é calculada sobre as cinco janelas, e seu desvio padrão mostra a variação entre elas. "
            "Em uma janela específica, o desvio padrão representa a variação entre as cinco permutações. "
            "Variáveis correlacionadas podem substituir umas às outras e levar a uma importância individual "
            "subestimada. Importância descreve associação preditiva, não causalidade."
        )

    path = DEFAULT_IMPORTANCE_REPORT_PATH
    if not path.is_file():
        st.warning("O relatório de Permutation Importance ainda não foi gerado.")
        st.code("urbanpulse-interpretability", language="bash")
        return
    try:
        report = load_importance_report(str(path), path.stat().st_mtime_ns)
    except (OSError, ValueError) as exc:
        st.error(f"Não foi possível carregar o relatório de interpretabilidade: {exc}")
        return

    scope_label = st.selectbox(
        "Resultados para exibir",
        ["Média agregada (cinco janelas)", "Janela 1", "Janela 2", "Janela 3", "Janela 4", "Janela 5"],
        key="importance_scope_selector",
    )
    if scope_label.startswith("Média"):
        selected = report.loc[report["scope"] == "aggregate"].copy()
        deviation_label = "Desvio padrão entre janelas"
    else:
        window = int(scope_label.split()[-1])
        selected = report.loc[
            (report["scope"] == "window") & (report["window"] == window)
        ].copy()
        deviation_label = "Desvio padrão entre repetições"

    feature_labels = {
        "season": "Estação",
        "yr": "Ano do dataset",
        "mnth": "Mês",
        "hr": "Hora do dia",
        "holiday": "Feriado",
        "weekday": "Dia da semana",
        "workingday": "Dia útil",
        "weathersit": "Condição do tempo",
        "temp": "Temperatura normalizada",
        "atemp": "Sensação térmica normalizada",
        "hum": "Umidade normalizada",
        "windspeed": "Vento normalizado",
    }
    selected["Variável"] = selected["feature"].map(feature_labels)
    selected = selected.sort_values("importance_mean", ascending=True)
    figure = px.bar(
        selected,
        x="importance_mean",
        y="Variável",
        orientation="h",
        error_x="importance_std",
        labels={
            "importance_mean": "Aumento médio do MAE (aluguéis/hora)",
            "Variável": "Variável",
            "importance_std": deviation_label,
        },
        title=f"Permutation Importance — {scope_label}",
        color_discrete_sequence=[ACCENT],
    )
    figure.update_yaxes(categoryorder="array", categoryarray=selected["Variável"].tolist())
    st.plotly_chart(_style_figure(figure, height=560), width="stretch")
    st.caption(
        "O aumento do MAE é calculado como MAE após embaralhar a variável menos MAE original. "
        "A análise usa somente previsões fora da amostra nas janelas iniciais."
    )
    feature_descriptions = {
        "season": "Categoria sazonal codificada pela UCI a partir das transições observadas no calendário.",
        "yr": "Ano do dataset: 0 representa 2011 e 1 representa 2012.",
        "mnth": "Mês do calendário, de 1 (janeiro) a 12 (dezembro).",
        "hr": "Hora do dia, de 0 a 23.",
        "holiday": "Indicador de feriado conforme o calendário publicado no dataset UCI.",
        "weekday": "Dia da semana na codificação UCI: domingo=0 até sábado=6.",
        "workingday": "Indica dia útil segundo a definição da UCI; fins de semana e feriados valem 0.",
        "weathersit": "Categoria UCI que resume a condição meteorológica observada.",
        "temp": "Temperatura do ar normalizada na escala UCI.",
        "atemp": "Sensação térmica normalizada na escala UCI.",
        "hum": "Umidade relativa normalizada na escala UCI.",
        "windspeed": "Velocidade do vento normalizada na escala UCI; a fonte não especifica unidade física.",
    }
    with st.expander("O que significa cada variável?"):
        descriptions = pd.DataFrame(
            [(feature, description) for feature, description in feature_descriptions.items()],
            columns=["Variável", "Descrição"],
        )
        st.dataframe(descriptions, hide_index=True, width="stretch")


def main() -> None:
    try:
        cloud_settings = st.secrets.get("cloud_resources", {})
    except StreamlitSecretNotFoundError:
        cloud_settings = {}
    try:
        resources = prepare_deployment_resources(
            cloud_settings.get("model_url"),
            cloud_settings.get("model_sha256"),
            cloud_settings.get("metadata_url"),
            cloud_settings.get("metadata_sha256"),
        )
        if resources.available_reports != ("final_metrics", "temporal_validation", "interpretability"):
            missing = sorted(
                {"final_metrics", "temporal_validation", "interpretability"}
                - set(resources.available_reports)
            )
            st.sidebar.warning(
                "Relatórios ausentes: " + ", ".join(missing) + ". Consulte os comandos no README."
            )
    except (OSError, ValueError, RuntimeError) as exc:
        st.error(f"Não foi possível preparar os recursos remotos: {exc}")

    st.sidebar.markdown("## UrbanPulse AI")
    st.sidebar.caption("Explore os dados, a avaliação e os cenários")
    page = st.sidebar.radio(
        "Navegação",
        ["Visão geral", "Modelos de Machine Learning", "Validação temporal", "Simulador de demanda", "Interpretabilidade"],
        label_visibility="collapsed",
        key="dashboard_page",
    )
    subtitles = {
        "Interpretabilidade": "Variáveis associadas à capacidade preditiva do Random Forest",
        "Simulador de demanda": "Explore cenários com o Random Forest treinado",
        "Visão geral": "Demanda horária por bicicletas compartilhadas",
        "Modelos de Machine Learning": "Comparação retrospectiva no teste final já consultado",
        "Validação temporal": "Estabilidade e padrões de erro em janelas progressivas",
    }
    _show_header(subtitles[page])
    if page == "Visão geral":
        _show_overview()
    elif page == "Modelos de Machine Learning":
        _show_model_comparison()
    elif page == "Validação temporal":
        _show_temporal_validation()
    elif page == "Simulador de demanda":
        _show_simulator()
    else:
        _show_interpretability()


if __name__ == "__main__":
    main()
