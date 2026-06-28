from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from qualar_core import (
    clean_qualar_dataframe,
    compute_tables,
    generate_figures,
    generate_text_report,
    identify_variables,
    read_input_file,
    run_pipeline,
)

st.set_page_config(page_title="QUALAR | Análise de Qualidade do Ar", layout="wide")

st.title("QUALAR — levantamento e análise de dados de qualidade do ar")
st.caption("ACH1026 | Consolidação, estatística descritiva, gráficos, meteorologia e relatório técnico automático")

with st.sidebar:
    st.header("Entrada")
    arquivo = st.file_uploader("Planilha consolidada QUALAR (.xlsx/.csv)", type=["xlsx", "xls", "csv"])
    estacao = st.text_input("Nome da estação", value="Estação analisada")
    col_anos = st.columns(2)
    ano_ini = col_anos[0].number_input("Ano inicial", min_value=2000, max_value=2100, value=2021)
    ano_fim = col_anos[1].number_input("Ano final", min_value=2000, max_value=2100, value=2025)
    gerar = st.button("Gerar análise", type="primary", width="stretch")

if arquivo is None:
    st.info("Envie a planilha consolidada. Exemplo esperado: colunas Data, Hora, NOx, O3, SO2, MP10, MP2.5, temperatura, umidade e vento.")
    st.stop()

if gerar:
    with st.spinner("Lendo, limpando, calculando tabelas, gerando PNGs e Excel..."):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            input_path = tmpdir / arquivo.name
            input_path.write_bytes(arquivo.getvalue())
            out_dir = tmpdir / "outputs_qualar"
            result = run_pipeline(input_path, out_dir, station_name=estacao, start_year=int(ano_ini), end_year=int(ano_fim))

            st.session_state["qualar_result"] = {
                "df": result["df"],
                "pollutants": result["pollutants"],
                "meteo": result["meteo"],
                "tables": result["tables"],
                "report": result["report"],
                "excel_bytes": Path(result["excel_path"]).read_bytes(),
                "text_bytes": Path(result["text_path"]).read_bytes(),
                "zip_bytes": Path(result["zip_path"]).read_bytes(),
                "chart_paths": result["chart_paths"],
                "chart_bytes": {Path(p).name: Path(p).read_bytes() for p in result["chart_paths"]},
            }

if "qualar_result" not in st.session_state:
    st.warning("Clique em **Gerar análise** para processar a planilha.")
    st.stop()

res = st.session_state["qualar_result"]
df = res["df"]
pollutants = res["pollutants"]
meteo = res["meteo"]
tables = res["tables"]

# KPIs
st.subheader("Resumo da base")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Registros horários", f"{len(df):,}".replace(",", "."))
col2.metric("Período", f"{int(df['Ano'].min())}–{int(df['Ano'].max())}")
col3.metric("Poluentes detectados", len(pollutants))
col4.metric("Variáveis meteorológicas", len(meteo))

if len(pollutants) < 3:
    st.error("Atenção: o enunciado pede pelo menos 3 poluentes. Esta base não atende ao mínimo.")
else:
    st.success("A base atende ao mínimo de 3 poluentes do trabalho.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Tabelas", "Gráficos Streamlit", "PNGs", "Texto técnico", "Exportação"])

with tab1:
    st.markdown("### Estatística descritiva")
    st.dataframe(tables["Estatistica_descritiva"], width="stretch", hide_index=True)

    st.markdown("### Completude dos dados")
    st.dataframe(tables["Completude"], width="stretch", hide_index=True)

    st.markdown("### Tendências anuais exploratórias")
    st.dataframe(tables["Tendencias"], width="stretch", hide_index=True)

    st.markdown("### Correlação poluente × meteorologia")
    if not tables["Correlacao_poluente_meteo"].empty:
        st.dataframe(tables["Correlacao_poluente_meteo"], width="stretch")
    else:
        st.info("Sem dados suficientes para matriz poluente × meteorologia.")

with tab2:
    st.markdown("### Séries temporais — médias diárias")
    daily = df.set_index("Datetime")[pollutants].resample("D").mean() if pollutants else pd.DataFrame()
    if not daily.empty:
        st.line_chart(daily, width="stretch")

    st.markdown("### Médias anuais")
    annual = tables["Medias_anuais"].set_index("Ano")
    if pollutants:
        st.bar_chart(annual[pollutants], width="stretch")

    st.markdown("### Perfil horário médio")
    hourly = tables["Perfil_horario"].set_index("Hora_Num")
    if pollutants:
        st.line_chart(hourly[pollutants], width="stretch")

    st.markdown("### Dispersão exploratória")
    if "O3 (µg/m³)" in df.columns and "Temperatura (°C)" in df.columns:
        scatter = df[["Temperatura (°C)", "O3 (µg/m³)"]].dropna()
        if len(scatter) > 5000:
            scatter = scatter.sample(5000, random_state=42)
        st.scatter_chart(scatter, x="Temperatura (°C)", y="O3 (µg/m³)", width="stretch")

with tab3:
    st.markdown("### Gráficos PNG gerados para relatório")
    for name, data in res["chart_bytes"].items():
        st.markdown(f"**{name}**")
        st.image(data, width="stretch")
        st.download_button(
            label=f"Baixar {name}",
            data=data,
            file_name=name,
            mime="image/png",
            width="stretch",
        )

with tab4:
    st.markdown(res["report"])

with tab5:
    st.markdown("### Baixar outputs")
    st.download_button(
        "Baixar pacote completo ZIP",
        data=res["zip_bytes"],
        file_name="QUALAR_outputs_tabelas_graficos_texto.zip",
        mime="application/zip",
        type="primary",
        width="stretch",
    )
    st.download_button(
        "Baixar tabelas Excel",
        data=res["excel_bytes"],
        file_name="QUALAR_tabelas_resultados.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )
    st.download_button(
        "Baixar relatório textual Markdown",
        data=res["text_bytes"],
        file_name="RELATORIO_TECNICO_QUALAR.md",
        mime="text/markdown",
        width="stretch",
    )
