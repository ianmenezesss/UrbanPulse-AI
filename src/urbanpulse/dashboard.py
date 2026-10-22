"""Local Streamlit dashboard for UrbanPulse AI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from plotly.graph_objects import Figure

from urbanpulse.dashboard_data import (
    DEFAULT_DATA_PATH,
    DEFAULT_FINAL_METRICS_PATH,
    DEFAULT_TEMPORAL_REPORT_DIR,
    WEEKDAY_NAMES_PT,
    filter_dashboard_data,
    load_dashboard_data,
    load_final_model_metrics,
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


def _style_figure(figure: Figure, *, height: int | None = None) -> Figure:
    figure.update_layout(
        template="plotly_white",
        margin={"l": 16, "r": 16, "t": 56, "b": 16},
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#233142"},
        title={"x": 0.02, "xanchor": "left", "font": {"size": 18}},
        legend_title_text="",
        height=height,
    )
    return figure


def _format_integer_pt(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def _format_decimal_pt(value: float) -> str:
    return f"{value:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _show_header(subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-kicker">URBAN MOBILITY · DATA EXPLORATION</div>
            <h1>UrbanPulse AI</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    metrics = st.columns(4)
    metrics[0].metric("Total de aluguéis", _format_integer_pt(int(filtered["cnt"].sum())))
    metrics[1].metric("Média por hora", _format_decimal_pt(float(filtered["cnt"].mean())))
    metrics[2].metric("Mediana por hora", _format_integer_pt(int(filtered["cnt"].median())))
    metrics[3].metric("Máximo em uma hora", _format_integer_pt(int(filtered["cnt"].max())))

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
        "Resultados do teste final reservado (últimos 20% da série cronológica). "
        "Esta tabela é distinta da validação temporal mostrada na área seguinte."
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
        "Ridge e Random Forest foram ajustados apenas nos 80% iniciais. "
        "O teste final foi usado somente para calcular estas métricas."
    )


def _show_temporal_validation() -> None:
    st.markdown("## Validação temporal")
    st.info(
        "Resultados fora da amostra dentro dos 80% iniciais. O teste final de 20% não participa "
        "destas janelas e seus resultados não são misturados aqui."
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


def main() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1440px;}
        .hero {padding: 1.6rem 2rem; margin: 0 0 1.6rem 0; border-radius: 18px;
               background: linear-gradient(115deg, #11283f 0%, #17445b 58%, #087f78 100%);}
        .hero-kicker {color: #9ee5d6; font-size: .72rem; letter-spacing: .16em; font-weight: 700;}
        .hero h1 {color: white; margin: .45rem 0 .2rem 0; font-size: 2.25rem;}
        .hero p {color: #d7e7ed; margin: 0; font-size: 1.02rem;}
        [data-testid="stMetric"] {background: #fff; border: 1px solid #e4ebef;
                                   padding: 1rem 1.1rem; border-radius: 14px;}
        [data-testid="stMetricLabel"] {color: #536777;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("## UrbanPulse AI")
    page = st.sidebar.radio(
        "Navegação",
        ["Visão geral", "Modelos de Machine Learning", "Validação temporal"],
        label_visibility="collapsed",
    )
    subtitles = {
        "Visão geral": "Demanda horária por bicicletas compartilhadas",
        "Modelos de Machine Learning": "Comparação dos modelos no teste final reservado",
        "Validação temporal": "Estabilidade e padrões de erro em janelas progressivas",
    }
    _show_header(subtitles[page])
    if page == "Visão geral":
        _show_overview()
    elif page == "Modelos de Machine Learning":
        _show_model_comparison()
    else:
        _show_temporal_validation()


if __name__ == "__main__":
    main()
