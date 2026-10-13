"""Local Streamlit dashboard for UrbanPulse AI's hourly bike demand EDA."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure
import streamlit as st

from urbanpulse.dashboard_data import (
    DEFAULT_DATA_PATH,
    WEEKDAY_NAMES_PT,
    filter_dashboard_data,
    load_dashboard_data,
)

st.set_page_config(
    page_title="UrbanPulse AI | Demanda de bicicletas",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#0F9D8A"
MONTH_LABELS = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]


@st.cache_data(show_spinner="Carregando os dados processados…", ttl="10m")
def load_data(path: str, modified_time_ns: int) -> pd.DataFrame:
    """Cache the prepared Parquet by path and modification time."""
    # modified_time_ns participates in Streamlit's cache key so regenerated
    # Parquet files are picked up on the next rerun.
    _ = modified_time_ns
    return load_dashboard_data(Path(path))


def _style_figure(figure: Figure) -> Figure:
    figure.update_layout(
        template="plotly_white",
        margin={"l": 16, "r": 16, "t": 56, "b": 16},
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#233142"},
        title={"x": 0.02, "xanchor": "left", "font": {"size": 18}},
        legend_title_text="",
    )
    return figure


def _format_integer_pt(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def _format_decimal_pt(value: float) -> str:
    return f"{value:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _show_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">URBAN MOBILITY · DATA EXPLORATION</div>
            <h1>UrbanPulse AI</h1>
            <p>Demanda horária por bicicletas compartilhadas</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    _show_header()

    data_path = DEFAULT_DATA_PATH
    st.sidebar.markdown("## Filtros")
    st.sidebar.caption("Os filtros são aplicados aos indicadores e a todos os gráficos.")

    year_choice = st.sidebar.selectbox("Ano", ["Ambos", "2011", "2012"], index=0)
    day_choice = st.sidebar.selectbox(
        "Tipo de dia",
        ["Todos", "Dias úteis", "Dias não úteis (fim de semana/feriado)"],
        index=0,
    )
    st.sidebar.caption(
        "Classificação conforme `workingday` da UCI: 1 indica dia útil; 0 inclui fins de semana e feriados."
    )

    if not data_path.is_file():
        st.warning("O arquivo processado ainda não existe. Gere-o primeiro com o comando abaixo:")
        st.code("urbanpulse-prepare", language="bash")
        st.stop()

    try:
        data = load_data(str(data_path), data_path.stat().st_mtime_ns)
    except (OSError, ValueError) as exc:
        st.error(f"Não foi possível carregar o dataset processado: {exc}")
        st.stop()

    year_filter = None if year_choice == "Ambos" else int(year_choice)
    workingday_filter = {
        "Todos": None,
        "Dias úteis": True,
        "Dias não úteis (fim de semana/feriado)": False,
    }[day_choice]
    filtered = filter_dashboard_data(data, year=year_filter, workingday=workingday_filter)

    if filtered.empty:
        st.info("Não há registros para a combinação de filtros selecionada.")
        st.stop()

    st.caption(
        f"Exibindo {len(filtered):,} de {len(data):,} intervalos horários · "
        f"{filtered['datetime'].min():%d/%m/%Y} — {filtered['datetime'].max():%d/%m/%Y}"
    )

    total_rentals = int(filtered["cnt"].sum())
    average_per_hour = float(filtered["cnt"].mean())
    median_per_hour = float(filtered["cnt"].median())
    maximum_per_hour = int(filtered["cnt"].max())

    metrics = st.columns(4)
    metrics[0].metric("Total de aluguéis", _format_integer_pt(total_rentals))
    metrics[1].metric("Média por hora", _format_decimal_pt(average_per_hour))
    metrics[2].metric("Mediana por hora", _format_integer_pt(int(median_per_hour)))
    metrics[3].metric("Máximo em uma hora", _format_integer_pt(maximum_per_hour))

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
        weekday_order = list(WEEKDAY_NAMES_PT.values())
        weekday_mean = (
            weekday_data.groupby("_weekday", as_index=False)["cnt"].mean()
            .rename(columns={"_weekday": "Dia da semana"})
        )
        weekday_figure = px.bar(
            weekday_mean,
            x="Dia da semana",
            y="cnt",
            category_orders={"Dia da semana": weekday_order},
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
        color_discrete_sequence=[ACCENT, "#4361a8"],
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
            color_discrete_sequence=["#4361a8"],
        )
        st.plotly_chart(_style_figure(distribution_figure), width="stretch")

    st.caption(
        "Fonte: UCI Bike Sharing Dataset. Visualizações descritivas; associação não implica causalidade. "
        "A temperatura está normalizada conforme a documentação da fonte."
    )


if __name__ == "__main__":
    main()
