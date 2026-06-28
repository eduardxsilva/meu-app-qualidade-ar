from __future__ import annotations

import io
import math
import os
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats


# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

RENAME_MAP = {
    "NOx(Óxidos de Nitrogênio) - ppb": "NOx (ppb)",
    "O3(Ozônio) - µg/m3": "O3 (µg/m³)",
    "SO2(Dióxido de Enxofre) - µg/m3": "SO2 (µg/m³)",
    "MP10(Partículas Inaláveis) - µg/m3": "MP10 (µg/m³)",
    "MP2.5(Partículas Inaláveis Finas) - µg/m3": "MP2.5 (µg/m³)",
    "TEMP(Temperatura do Ar) - °C": "Temperatura (°C)",
    "UR(Umidade Relativa do Ar) - %": "Umidade Relativa (%)",
    "VV(Velocidade do Vento) - m/s": "Velocidade do Vento (m/s)",
    "DV(Direção do Vento) - °": "Direção do Vento (°)",
    "DVG(Direção do Vento Global) - °": "Direção do Vento Global (°)",
}

POLLUTANT_ORDER = ["NOx (ppb)", "O3 (µg/m³)", "SO2 (µg/m³)", "MP10 (µg/m³)", "MP2.5 (µg/m³)"]
METEO_ORDER = [
    "Temperatura (°C)",
    "Umidade Relativa (%)",
    "Velocidade do Vento (m/s)",
    "Direção do Vento (°)",
    "Direção do Vento Global (°)",
]
MONTH_LABELS = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
SEASON_MAP = {
    12: "Verão", 1: "Verão", 2: "Verão",
    3: "Outono", 4: "Outono", 5: "Outono",
    6: "Inverno", 7: "Inverno", 8: "Inverno",
    9: "Primavera", 10: "Primavera", 11: "Primavera",
}
SEASON_ORDER = ["Verão", "Outono", "Inverno", "Primavera"]

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "figure.autolayout": False,
})


# ============================================================
# LEITURA E LIMPEZA
# ============================================================

def read_input_file(path_or_buffer) -> pd.DataFrame:
    """Lê XLSX/CSV já consolidado ou exportado do QUALAR/INMET.

    Aceita:
    - caminho local Path/str;
    - UploadedFile do Streamlit;
    - BytesIO.
    """
    name = getattr(path_or_buffer, "name", None) or str(path_or_buffer)
    lower = name.lower()

    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return pd.read_excel(path_or_buffer)

    # CSV: tenta separadores e codificações comuns em QUALAR/INMET.
    if hasattr(path_or_buffer, "seek"):
        path_or_buffer.seek(0)
    try:
        return pd.read_csv(path_or_buffer, sep=";", encoding="latin1", decimal=",", engine="python")
    except Exception:
        if hasattr(path_or_buffer, "seek"):
            path_or_buffer.seek(0)
        try:
            return pd.read_csv(path_or_buffer, sep=";", encoding="utf-8", decimal=",", engine="python")
        except Exception:
            if hasattr(path_or_buffer, "seek"):
                path_or_buffer.seek(0)
            return pd.read_csv(path_or_buffer, sep=",", encoding="utf-8", decimal=".", engine="python")


def _normalize_hour_string(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Retorna hora normalizada HH:MM e máscara de horas 24:00."""
    raw = series.astype(str).str.strip()
    raw = raw.str.replace("24:00:00", "24:00", regex=False)
    mask_24 = raw.str.match(r"^24:00", na=False)

    # INMET costuma trazer 0000 UTC, 1300 UTC etc.
    tmp = raw.str.replace(" UTC", "", regex=False).str.strip()
    is_hhmm_digits = tmp.str.match(r"^\d{4}$", na=False)
    tmp.loc[is_hhmm_digits] = tmp.loc[is_hhmm_digits].str.slice(0, 2) + ":" + tmp.loc[is_hhmm_digits].str.slice(2, 4)

    extracted = tmp.str.extract(r"(\d{1,2}:\d{2})", expand=False)
    hour_norm = extracted.fillna(tmp)
    hour_norm = hour_norm.str.replace("24:00", "00:00", regex=False)
    return hour_norm, mask_24


def parse_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Cria Datetime robusto e preserva Ano/Mês a partir da data-base original."""
    out = df.copy()
    if "Data" not in out.columns or "Hora" not in out.columns:
        # Se já existir Datetime, usa diretamente.
        if "Datetime" in out.columns:
            out["Datetime"] = pd.to_datetime(out["Datetime"], errors="coerce")
            out["Data_Base"] = out["Datetime"].dt.normalize()
            out["Ano"] = out["Datetime"].dt.year
            out["Mes"] = out["Datetime"].dt.month
            out["Hora_Num"] = out["Datetime"].dt.hour
            out["Estação"] = out["Mes"].map(SEASON_MAP)
            return out
        raise ValueError("A planilha precisa ter colunas 'Data' e 'Hora', ou uma coluna 'Datetime'.")

    # Detecta formatos dd/mm/yyyy, yyyy/mm/dd, yyyy-mm-dd.
    data_str = out["Data"].astype(str).str.strip()
    date_base = pd.to_datetime(data_str, dayfirst=True, errors="coerce")
    # Segunda tentativa para yyyy/mm/dd quando dayfirst falhar.
    mask_bad = date_base.isna()
    if mask_bad.any():
        date_base.loc[mask_bad] = pd.to_datetime(data_str.loc[mask_bad], yearfirst=True, errors="coerce")

    hour_norm, mask_24 = _normalize_hour_string(out["Hora"])
    dt = pd.to_datetime(date_base.dt.strftime("%d/%m/%Y") + " " + hour_norm, format="%d/%m/%Y %H:%M", errors="coerce")
    dt.loc[mask_24 & dt.notna()] = dt.loc[mask_24 & dt.notna()] + pd.Timedelta(days=1)

    out["Data_Base"] = date_base
    out["Datetime"] = dt
    out["Ano"] = date_base.dt.year
    out["Mes"] = date_base.dt.month
    out["Hora_Num"] = dt.dt.hour
    out["Estação"] = out["Mes"].map(SEASON_MAP)
    return out


def clean_qualar_dataframe(df_raw: pd.DataFrame, start_year: int = 2021, end_year: int = 2025) -> pd.DataFrame:
    """Padroniza nomes, datas, variáveis numéricas e filtra período-base."""
    df = df_raw.copy()

    # Remove colunas vazias/Unnamed.
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed", na=False)]
    df.rename(columns=RENAME_MAP, inplace=True)

    df = parse_datetime_columns(df)
    df = df[df["Ano"].between(start_year, end_year, inclusive="both")].copy()
    df = df.dropna(subset=["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)

    # Conversão numérica conservadora.
    non_numeric = {"Data", "Hora", "Data_Base", "Datetime", "Estação"}
    for col in df.columns:
        if col in non_numeric:
            continue
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def identify_variables(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    pollutants = [c for c in POLLUTANT_ORDER if c in df.columns]
    meteo = [c for c in METEO_ORDER if c in df.columns]
    numeric = [c for c in pollutants + meteo if c in df.columns]
    return pollutants, meteo, numeric


# ============================================================
# TABELAS ANALÍTICAS
# ============================================================

def max_consecutive_missing(series: pd.Series) -> int:
    mask = series.isna().astype(int)
    if len(mask) == 0:
        return 0
    groups = (mask != mask.shift()).cumsum()
    runs = mask.groupby(groups).sum()
    return int(runs.max()) if not runs.empty else 0


def wind_sector(degrees) -> Optional[str]:
    if pd.isna(degrees):
        return np.nan
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return dirs[int(((float(degrees) % 360) + 22.5) // 45) % 8]


def compute_tables(df: pd.DataFrame, pollutants: List[str], meteo: List[str]) -> Dict[str, pd.DataFrame]:
    variables = pollutants + meteo
    n_total = len(df)

    desc = df[variables].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]).T
    desc = desc.rename(columns={
        "count": "N válido",
        "mean": "Média",
        "std": "Desvio padrão",
        "min": "Mínimo",
        "25%": "P25",
        "50%": "Mediana",
        "75%": "P75",
        "90%": "P90",
        "95%": "P95",
        "max": "Máximo",
    })
    desc.insert(1, "N ausente", n_total - desc["N válido"])
    desc.insert(2, "Completude (%)", desc["N válido"] / n_total * 100)
    desc.insert(3, "Maior falha contínua (h)", [max_consecutive_missing(df[v]) for v in desc.index])
    desc = desc.reset_index().rename(columns={"index": "Variável"})

    completeness = desc[["Variável", "N válido", "N ausente", "Completude (%)", "Maior falha contínua (h)"]].copy()
    completeness["Classificação QA/QC"] = pd.cut(
        completeness["Completude (%)"],
        bins=[-0.1, 0.0, 50.0, 80.0, 95.0, 100.0],
        labels=["Não monitorado", "Crítico", "Limitado", "Bom", "Excelente"],
    ).astype(str)

    annual_mean = df.groupby("Ano")[variables].mean(numeric_only=True).reset_index()
    annual_count = df.groupby("Ano")[variables].count().reset_index()
    monthly_mean = df.groupby(["Ano", "Mes"])[variables].mean(numeric_only=True).reset_index()
    monthly_mean["Mês nome"] = monthly_mean["Mes"].map(lambda m: MONTH_LABELS[int(m)-1] if pd.notna(m) and 1 <= int(m) <= 12 else np.nan)
    monthly_climatology = df.groupby("Mes")[variables].mean(numeric_only=True).reset_index()
    monthly_climatology["Mês nome"] = monthly_climatology["Mes"].map(lambda m: MONTH_LABELS[int(m)-1])
    hourly_profile = df.groupby("Hora_Num")[variables].mean(numeric_only=True).reset_index()
    seasonal = df.groupby("Estação")[variables].mean(numeric_only=True).reindex(SEASON_ORDER).reset_index()

    trend_rows = []
    for var in pollutants:
        ser = annual_mean.set_index("Ano")[var].dropna()
        if len(ser) >= 2:
            res = stats.linregress(ser.index.astype(float), ser.values.astype(float))
            var_pct = ((ser.iloc[-1] - ser.iloc[0]) / ser.iloc[0] * 100) if ser.iloc[0] != 0 else np.nan
            direction = "alta" if res.slope > 0 else "queda"
            trend_rows.append({
                "Variável": var,
                "Anos usados": int(len(ser)),
                "Ano inicial": int(ser.index.min()),
                "Ano final": int(ser.index.max()),
                "Média inicial": float(ser.iloc[0]),
                "Média final": float(ser.iloc[-1]),
                "Inclinação anual": float(res.slope),
                "p-valor": float(res.pvalue),
                "Variação (%)": float(var_pct),
                "Direção exploratória": direction,
                "Força estatística": "fraca/não conclusiva" if res.pvalue >= 0.05 else "estatisticamente significativa",
            })
    trends = pd.DataFrame(trend_rows)

    corr = df[variables].corr(method="pearson", min_periods=50)
    corr_pm = corr.loc[pollutants, meteo] if pollutants and meteo else pd.DataFrame()

    events = []
    for var in pollutants:
        valid = df[["Datetime", "Ano", "Mes", "Hora_Num", var]].dropna()
        if valid.empty:
            continue
        threshold = valid[var].quantile(0.95)
        high = valid[valid[var] >= threshold].copy()
        events.append({
            "Variável": var,
            "Limite P95": threshold,
            "N eventos >= P95": int(len(high)),
            "Mês mais frequente nos eventos": int(high["Mes"].mode().iloc[0]) if not high["Mes"].mode().empty else np.nan,
            "Hora mais frequente nos eventos": int(high["Hora_Num"].mode().iloc[0]) if not high["Hora_Num"].mode().empty else np.nan,
        })
    high_events = pd.DataFrame(events)

    direction_col = "Direção do Vento (°)" if "Direção do Vento (°)" in df.columns else ("Direção do Vento Global (°)" if "Direção do Vento Global (°)" in df.columns else None)
    wind_sector_table = pd.DataFrame()
    if direction_col:
        tmp = df.copy()
        tmp["Setor do vento"] = tmp[direction_col].apply(wind_sector)
        wind_sector_table = tmp.groupby("Setor do vento")[pollutants + (["Velocidade do Vento (m/s)"] if "Velocidade do Vento (m/s)" in tmp.columns else [])].agg(["count", "mean"])
        wind_sector_table.columns = [f"{a} | {b}" for a, b in wind_sector_table.columns]
        wind_sector_table = wind_sector_table.reindex(["N", "NE", "E", "SE", "S", "SW", "W", "NW"]).reset_index()

    return {
        "Base_limpa": df,
        "Estatistica_descritiva": desc,
        "Completude": completeness,
        "Medias_anuais": annual_mean,
        "Contagem_anual": annual_count,
        "Media_mensal_ano": monthly_mean,
        "Sazonalidade_mensal": monthly_climatology,
        "Perfil_horario": hourly_profile,
        "Media_estacional": seasonal,
        "Tendencias": trends,
        "Correlacao": corr,
        "Correlacao_poluente_meteo": corr_pm,
        "Eventos_P95": high_events,
        "Setores_vento": wind_sector_table,
    }


# ============================================================
# GRÁFICOS PNG
# ============================================================

def _safe_filename(text: str) -> str:
    text = re.sub(r"[^0-9A-Za-zÀ-ÿ._ -]+", "", text)
    text = text.replace(" ", "_").replace("/", "-")
    return text[:80]


def generate_figures(df: pd.DataFrame, tables: Dict[str, pd.DataFrame], output_dir: Path, pollutants: List[str], meteo: List[str]) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []

    # 1) Séries diárias dos poluentes.
    if pollutants:
        daily = df.set_index("Datetime")[pollutants].resample("D").mean()
        n = len(pollutants)
        fig, axes = plt.subplots(n, 1, figsize=(13, max(3.0 * n, 6)), sharex=True)
        if n == 1:
            axes = [axes]
        for ax, var in zip(axes, pollutants):
            ax.plot(daily.index, daily[var], linewidth=0.9)
            ax.set_ylabel(var)
            ax.set_title(f"Média diária - {var}", loc="left", fontweight="bold")
            ax.grid(True, alpha=0.25)
        axes[-1].xaxis.set_major_locator(mdates.YearLocator())
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.suptitle("Séries temporais diárias dos poluentes (2021-2025)", fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        path = output_dir / "01_series_diarias_poluentes.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))

    # 2) Índice das médias anuais base 2021/primeiro ano disponível = 100.
    annual = tables["Medias_anuais"].set_index("Ano")
    annual_index = pd.DataFrame(index=annual.index)
    for var in pollutants:
        ser = annual[var].dropna()
        if len(ser) >= 2 and ser.iloc[0] != 0:
            annual_index[var] = annual[var] / ser.iloc[0] * 100
    if not annual_index.dropna(how="all").empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        annual_index.plot(marker="o", linewidth=2, ax=ax)
        ax.axhline(100, linestyle="--", linewidth=1)
        ax.set_title("Tendência comparativa das médias anuais — índice do primeiro ano disponível = 100", fontweight="bold")
        ax.set_ylabel("Índice")
        ax.set_xlabel("Ano")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        path = output_dir / "02_indice_medias_anuais_poluentes.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))

    # 3) Boxplot mensal por poluente.
    if pollutants:
        n = len(pollutants)
        fig, axes = plt.subplots(n, 1, figsize=(13, max(3.0 * n, 6)), sharex=True)
        if n == 1:
            axes = [axes]
        plot_df = df.copy()
        plot_df["Mês"] = pd.Categorical(plot_df["Mes"].map(lambda m: MONTH_LABELS[int(m)-1] if pd.notna(m) else np.nan), categories=MONTH_LABELS, ordered=True)
        for ax, var in zip(axes, pollutants):
            sns.boxplot(data=plot_df, x="Mês", y=var, ax=ax, fliersize=1.5, linewidth=0.8)
            ax.set_title(f"Sazonalidade mensal - {var}", loc="left", fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel(var)
        fig.suptitle("Distribuição mensal dos poluentes", fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        path = output_dir / "03_boxplot_mensal_poluentes.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))

    # 4) Perfil horário médio.
    hourly = tables["Perfil_horario"]
    if pollutants and not hourly.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        for var in pollutants:
            if var in hourly.columns:
                ax.plot(hourly["Hora_Num"], hourly[var], marker="o", linewidth=1.8, label=var)
        ax.set_xticks(range(0, 24, 2))
        ax.set_title("Perfil horário médio dos poluentes", fontweight="bold")
        ax.set_xlabel("Hora do dia")
        ax.set_ylabel("Concentração média")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        path = output_dir / "04_perfil_horario_poluentes.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))

    # 5) Matriz de correlação.
    corr = tables["Correlacao"]
    if not corr.empty:
        fig, ax = plt.subplots(figsize=(11, 9))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=True, linewidths=0.5, ax=ax)
        ax.set_title("Matriz de correlação de Pearson", fontweight="bold")
        path = output_dir / "05_matriz_correlacao.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))

    # 6) Dispersão polar dos ventos.
    wind_speed_col = "Velocidade do Vento (m/s)"
    wind_dir_col = "Direção do Vento (°)" if "Direção do Vento (°)" in df.columns else ("Direção do Vento Global (°)" if "Direção do Vento Global (°)" in df.columns else None)
    if wind_speed_col in df.columns and wind_dir_col in df.columns:
        wind_df = df[[wind_speed_col, wind_dir_col]].dropna()
        if not wind_df.empty:
            fig = plt.figure(figsize=(9, 8))
            ax = fig.add_subplot(111, polar=True)
            theta = np.radians(wind_df[wind_dir_col].to_numpy() % 360)
            speed = wind_df[wind_speed_col].to_numpy()
            scatter = ax.scatter(theta, speed, c=speed, cmap="viridis", s=5, alpha=0.55, edgecolors="none")
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
            ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
            ax.set_title("Distribuição polar: direção e velocidade do vento", fontweight="bold", pad=20)
            cbar = fig.colorbar(scatter, ax=ax, pad=0.1, fraction=0.046)
            cbar.set_label("Velocidade do vento (m/s)", rotation=270, labelpad=18)
            path = output_dir / "06_vento_polar.png"
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            paths.append(str(path))

    # 7) Médias por setor de vento.
    sectors = tables.get("Setores_vento", pd.DataFrame())
    if pollutants and not sectors.empty:
        mean_cols = [f"{p} | mean" for p in pollutants if f"{p} | mean" in sectors.columns]
        if mean_cols:
            fig, axes = plt.subplots(len(mean_cols), 1, figsize=(12, max(3 * len(mean_cols), 6)), sharex=True)
            if len(mean_cols) == 1:
                axes = [axes]
            order = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
            sectors_plot = sectors.set_index("Setor do vento").reindex(order)
            for ax, col in zip(axes, mean_cols):
                sectors_plot[col].plot(kind="bar", ax=ax)
                ax.set_title(col.replace(" | mean", ""), loc="left", fontweight="bold")
                ax.set_ylabel("Média")
                ax.grid(True, axis="y", alpha=0.25)
            axes[-1].set_xlabel("Setor de origem do vento")
            fig.suptitle("Concentrações médias por setor de vento", fontsize=15, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.98])
            path = output_dir / "07_poluentes_por_setor_vento.png"
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            paths.append(str(path))

    # 8) Relações meteorológicas selecionadas.
    relations = []
    if "O3 (µg/m³)" in df.columns and "Temperatura (°C)" in df.columns:
        relations.append(("Temperatura (°C)", "O3 (µg/m³)", "O3 x temperatura"))
    if "O3 (µg/m³)" in df.columns and "NOx (ppb)" in df.columns:
        relations.append(("NOx (ppb)", "O3 (µg/m³)", "O3 x NOx"))
    if "MP10 (µg/m³)" in df.columns and "Velocidade do Vento (m/s)" in df.columns:
        relations.append(("Velocidade do Vento (m/s)", "MP10 (µg/m³)", "MP10 x vento"))
    if "MP2.5 (µg/m³)" in df.columns and "Velocidade do Vento (m/s)" in df.columns:
        relations.append(("Velocidade do Vento (m/s)", "MP2.5 (µg/m³)", "MP2.5 x vento"))
    if relations:
        fig, axes = plt.subplots(len(relations), 1, figsize=(10, max(4 * len(relations), 6)))
        if len(relations) == 1:
            axes = [axes]
        for ax, (x, y, title) in zip(axes, relations):
            sample = df[[x, y]].dropna()
            if len(sample) > 6000:
                sample = sample.sample(6000, random_state=42)
            sns.regplot(data=sample, x=x, y=y, ax=ax, scatter_kws={"s": 8, "alpha": 0.25}, line_kws={"linewidth": 1.4}, lowess=True)
            ax.set_title(title, loc="left", fontweight="bold")
            ax.grid(True, alpha=0.25)
        fig.suptitle("Relações entre poluentes e variáveis meteorológicas", fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        path = output_dir / "08_relacoes_poluente_meteorologia.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))

    # 9) Heatmap ano x mês para cada poluente.
    if pollutants:
        n = len(pollutants)
        fig, axes = plt.subplots(n, 1, figsize=(12, max(2.6 * n, 6)))
        if n == 1:
            axes = [axes]
        for ax, var in zip(axes, pollutants):
            pivot = df.pivot_table(index="Ano", columns="Mes", values=var, aggfunc="mean")
            pivot = pivot.reindex(columns=range(1, 13))
            sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.5, ax=ax, cbar=True)
            ax.set_title(var, loc="left", fontweight="bold")
            ax.set_xlabel("Mês")
            ax.set_ylabel("Ano")
            ax.set_xticklabels(MONTH_LABELS, rotation=0)
        fig.suptitle("Médias mensais por ano", fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        path = output_dir / "09_heatmap_mensal_anual_poluentes.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))

    # 10) Completude.
    comp = tables["Completude"].copy()
    if not comp.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=comp, y="Variável", x="Completude (%)", ax=ax, color="#4C78A8")
        ax.axvline(80, linestyle="--", linewidth=1)
        ax.axvline(95, linestyle="--", linewidth=1)
        ax.set_xlim(0, 100)
        ax.set_title("Completude dos dados por variável", fontweight="bold")
        ax.set_xlabel("Registros válidos (%)")
        ax.set_ylabel("")
        path = output_dir / "10_completude_variaveis.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))

    return paths


# ============================================================
# RELATÓRIO TEXTUAL
# ============================================================

def classify_pollutant_source(var: str) -> str:
    upper = var.upper()
    if "O3" in upper:
        return "poluente secundário; tende a responder à radiação solar, temperatura e reações fotoquímicas envolvendo NOx e compostos orgânicos voláteis."
    if "NOX" in upper or "NO2" in upper or "NO " in upper:
        return "marcador de combustão, com forte associação esperada a tráfego veicular e emissões de diesel/gasolina."
    if "MP10" in upper:
        return "material particulado inalável; pode refletir ressuspensão de poeira, tráfego, obras, solo exposto e fontes industriais próximas."
    if "MP2.5" in upper or "MP2" in upper:
        return "particulado fino; mais associado a combustão, aerossóis secundários e transporte regional, com maior relevância sanitária."
    if "SO2" in upper:
        return "gás associado à queima de combustíveis sulfurados e processos industriais; em áreas urbanas atuais costuma apresentar concentrações menores."
    return "parâmetro com fonte dominante dependente do contexto local da estação."


def _fmt(x, digits=2) -> str:
    if pd.isna(x):
        return "n/d"
    return f"{x:.{digits}f}"


def generate_text_report(df: pd.DataFrame, tables: Dict[str, pd.DataFrame], pollutants: List[str], meteo: List[str], station_name: str = "Estação analisada") -> str:
    dt_min = df["Datetime"].min()
    dt_max = df["Datetime"].max()
    data_base_min = df["Data_Base"].min() if "Data_Base" in df.columns else dt_min
    data_base_max = df["Data_Base"].max() if "Data_Base" in df.columns else dt_max
    n = len(df)
    comp = tables["Completude"].set_index("Variável")
    desc = tables["Estatistica_descritiva"].set_index("Variável")
    trends = tables["Tendencias"]
    corr_pm = tables["Correlacao_poluente_meteo"]

    pollutants_available = [p for p in pollutants if comp.loc[p, "N válido"] > 0] if not comp.empty else []
    mandatory_msg = "Atende ao mínimo de três poluentes monitorados." if len(pollutants_available) >= 3 else "Não atende ao mínimo de três poluentes monitorados; complementar a base antes de fechar o relatório."

    lines = []
    lines.append(f"# Relatório técnico automático — qualidade do ar ({station_name})")
    lines.append("")
    lines.append("## 1. Base de dados e escopo")
    lines.append(f"A base consolidada contém **{n:,} registros horários** com data-base entre **{data_base_min:%d/%m/%Y}** e **{data_base_max:%d/%m/%Y}**. O período-base do trabalho é 2021–2025, com análise exploratória dos poluentes e das variáveis meteorológicas disponíveis.")
    if dt_max.year > data_base_max.year:
        lines.append("Observação técnica: registros com hora `24:00` foram convertidos para `00:00` do dia seguinte no campo `Datetime`, mas as agregações por ano usam a data-base original da CETESB.")
    lines.append(f"Poluentes detectados na planilha: **{', '.join(pollutants_available) if pollutants_available else 'nenhum'}**. {mandatory_msg}")
    lines.append("")
    lines.append("## 2. Qualidade dos dados")
    for var in pollutants + meteo:
        if var in comp.index:
            lines.append(f"- **{var}:** completude de {_fmt(comp.loc[var, 'Completude (%)'])}% e maior falha contínua de {int(comp.loc[var, 'Maior falha contínua (h)'])} horas.")
    lines.append("")
    lines.append("Leitura crítica: variáveis com baixa completude devem ser usadas com cautela. Em especial, correlações com meteorologia podem ficar enviesadas quando há poucas observações válidas ou quando as lacunas se concentram em meses específicos.")
    lines.append("")

    lines.append("## 3. Estatística descritiva dos poluentes")
    for var in pollutants:
        if var in desc.index and desc.loc[var, "N válido"] > 0:
            lines.append(f"- **{var}:** média = {_fmt(desc.loc[var, 'Média'])}; mediana = {_fmt(desc.loc[var, 'Mediana'])}; desvio padrão = {_fmt(desc.loc[var, 'Desvio padrão'])}; P95 = {_fmt(desc.loc[var, 'P95'])}; máximo = {_fmt(desc.loc[var, 'Máximo'])}.")
    lines.append("")

    lines.append("## 4. Tendências temporais")
    if trends.empty:
        lines.append("Não houve séries anuais suficientes para estimar tendência linear exploratória.")
    else:
        for _, row in trends.iterrows():
            cautela = " Resultado estatisticamente fraco; use como indício, não como conclusão causal." if row["p-valor"] >= 0.05 else " Resultado com significância estatística no teste linear simples."
            lines.append(f"- **{row['Variável']}:** tendência de **{row['Direção exploratória']}**, inclinação de {_fmt(row['Inclinação anual'])} unidade/ano e variação de {_fmt(row['Variação (%)'])}% entre {int(row['Ano inicial'])} e {int(row['Ano final'])}. p-valor = {_fmt(row['p-valor'], 3)}.{cautela}")
    lines.append("")

    lines.append("## 5. Meteorologia e interpretação ambiental")
    lines.append("A literatura meteorológica recomenda interpretar poluentes junto com temperatura, umidade, precipitação, estabilidade atmosférica e vento. Temperatura e radiação favorecem reações fotoquímicas, enquanto vento e turbulência alteram dispersão e diluição. Em atmosfera estável, sobretudo em noites frias e situações de inversão térmica, a mistura vertical é reduzida e contaminantes primários podem se acumular próximo à superfície.")
    if not corr_pm.empty:
        for pol in pollutants:
            if pol in corr_pm.index:
                candidates = corr_pm.loc[pol].dropna().sort_values(key=lambda s: s.abs(), ascending=False)
                if not candidates.empty:
                    top = candidates.index[0]
                    val = candidates.iloc[0]
                    lines.append(f"- **{pol}:** maior correlação meteorológica observada foi com **{top}** (r = {_fmt(val, 2)}).")
    lines.append("")

    lines.append("## 6. Hipóteses de fontes e entorno")
    for var in pollutants:
        if var in pollutants_available:
            lines.append(f"- **{var}:** {classify_pollutant_source(var)}")
    lines.append("Essas hipóteses precisam ser cruzadas com o entorno real da estação: avenidas, corredores de ônibus/caminhões, terminais, indústrias, obras, vegetação, topografia e classificação de representatividade espacial da CETESB.")
    lines.append("")

    lines.append("## 7. Conclusão preliminar")
    best_complete = comp.loc[pollutants, "Completude (%)"].sort_values(ascending=False) if pollutants else pd.Series(dtype=float)
    if not best_complete.empty:
        lines.append(f"O poluente com base mais robusta foi **{best_complete.index[0]}** ({_fmt(best_complete.iloc[0])}% de completude). A interpretação final deve priorizar variáveis com maior continuidade temporal e evitar conclusões fortes sobre variáveis com lacunas extensas.")
    if "O3 (µg/m³)" in pollutants_available and "Temperatura (°C)" in meteo and not corr_pm.empty:
        r = corr_pm.loc["O3 (µg/m³)", "Temperatura (°C)"] if "O3 (µg/m³)" in corr_pm.index and "Temperatura (°C)" in corr_pm.columns else np.nan
        if pd.notna(r):
            lines.append(f"O resultado mais claro de interação meteorológica foi a associação entre **O3 e temperatura** (r = {_fmt(r, 2)}), compatível com a natureza fotoquímica do ozônio troposférico.")
    lines.append("Não feche o trabalho apenas com estatística: a parte de fontes no entorno e características da estação precisa ser descrita com mapa, relatório CETESB e observação local.")
    lines.append("")

    lines.append("## 8. Referências usadas no texto automático")
    lines.append("- AHRENS, C. Donald. *Meteorology Today: An Introduction to Weather, Climate, and the Environment*. 9. ed. Brooks/Cole, 2009. Capítulos úteis: temperatura sazonal/diária, umidade, estabilidade, precipitação, vento e poluição do ar.")
    lines.append("- CETESB. Relatórios de Qualidade do Ar do Estado de São Paulo. Página oficial de publicações e relatórios: https://www.cetesb.sp.gov.br/cetesb/qualidade_ambiental/ar/publicacoes_e_relatorios/relatorios_de_qualidade_do_ar_de_sao_paulo")
    lines.append("- SANTOS, F. S. et al. Avaliação da influência das condições meteorológicas na qualidade do ar. *Engenharia Sanitária e Ambiental*, 2019. https://www.scielo.br/j/esa/a/4kSVDKgVcYnwNFt5R6Yb5pp/")
    lines.append("- Link adicional informado pelo usuário: https://www.sciencedirect.com/science/article/abs/pii/S1309104219304568")
    lines.append("")

    return "\n".join(lines)


# ============================================================
# EXPORTAÇÃO
# ============================================================

def export_excel(tables: Dict[str, pd.DataFrame], output_path: Path, text_report: Optional[str] = None) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="xlsxwriter", datetime_format="yyyy-mm-dd hh:mm", date_format="yyyy-mm-dd") as writer:
        workbook = writer.book
        fmt_header = workbook.add_format({"bold": True, "bg_color": "#0F766E", "font_color": "#FFFFFF", "border": 1, "align": "center", "valign": "vcenter"})
        fmt_num = workbook.add_format({"num_format": "0.00"})
        fmt_int = workbook.add_format({"num_format": "0"})
        fmt_pct = workbook.add_format({"num_format": "0.00"})
        fmt_text = workbook.add_format({"text_wrap": True, "valign": "top"})

        # Ordem das abas no arquivo.
        sheet_order = [
            "Estatistica_descritiva", "Completude", "Tendencias", "Medias_anuais", "Contagem_anual",
            "Sazonalidade_mensal", "Media_mensal_ano", "Media_estacional", "Perfil_horario",
            "Correlacao_poluente_meteo", "Correlacao", "Eventos_P95", "Setores_vento", "Base_limpa"
        ]
        if text_report:
            report_lines = pd.DataFrame({"Relatório técnico automático": text_report.splitlines()})
            report_lines.to_excel(writer, sheet_name="Relatorio_texto", index=False)

        for name in sheet_order:
            if name not in tables:
                continue
            table = tables[name]
            if table is None or table.empty:
                continue
            safe_name = name[:31]
            out = table.copy()
            # Excel não aceita timezone.
            for col in out.columns:
                if pd.api.types.is_datetime64_any_dtype(out[col]):
                    out[col] = out[col].dt.tz_localize(None) if getattr(out[col].dt, 'tz', None) is not None else out[col]
            out.to_excel(writer, sheet_name=safe_name, index=True if name == "Correlacao" else False)

        # Referências bibliográficas.
        refs = pd.DataFrame({
            "Fonte": ["CETESB", "Ahrens", "Santos et al. 2019", "ScienceDirect informado"],
            "Uso no trabalho": [
                "Relatórios anuais e caracterização/representatividade de estações.",
                "Base conceitual de estabilidade atmosférica, vento, precipitação, temperatura e poluição do ar.",
                "Discussão sobre influência das condições meteorológicas na qualidade do ar.",
                "Referência adicional enviada pelo usuário; conferir metadados no acesso institucional."
            ],
            "URL/Referência": [
                "https://www.cetesb.sp.gov.br/cetesb/qualidade_ambiental/ar/publicacoes_e_relatorios/relatorios_de_qualidade_do_ar_de_sao_paulo",
                "Ahrens, C. D. Meteorology Today. 9th ed. Brooks/Cole, 2009.",
                "https://www.scielo.br/j/esa/a/4kSVDKgVcYnwNFt5R6Yb5pp/",
                "https://www.sciencedirect.com/science/article/abs/pii/S1309104219304568"
            ]
        })
        refs.to_excel(writer, sheet_name="Referencias", index=False)

        # Formatação simples de todas as abas.
        for sheet_name, worksheet in writer.sheets.items():
            worksheet.freeze_panes(1, 0)
            worksheet.set_row(0, 22, fmt_header)
            # Ajuste de largura por conteúdo, com limite.
            if sheet_name == "Relatorio_texto":
                worksheet.set_column(0, 0, 120, fmt_text)
                continue
            # Descobre df associado.
            ws_df = None
            if sheet_name == "Referencias":
                ws_df = refs
            else:
                for k, v in tables.items():
                    if k[:31] == sheet_name:
                        ws_df = v.reset_index() if k == "Correlacao" else v
                        break
            if ws_df is not None:
                for idx, col in enumerate(ws_df.columns):
                    # Pandas/Streamlit Cloud pode usar dtype Arrow.
                    # Nessa condição, Series.map(len) quebra quando há <NA>,
                    # Timestamp, numpy scalars ou valores não-string.
                    # A conversão abaixo é deliberadamente defensiva.
                    sample_values = ws_df[col].head(1000).tolist()
                    sample_lengths = []
                    for value in sample_values:
                        if pd.isna(value):
                            sample_lengths.append(0)
                        else:
                            sample_lengths.append(len(str(value)))
                    max_len = max([len(str(col))] + sample_lengths)
                    width = min(max(max_len + 2, 10), 38)
                    worksheet.set_column(idx, idx, width)
                    if pd.api.types.is_numeric_dtype(ws_df[col]):
                        worksheet.set_column(idx, idx, min(max(width, 12), 18), fmt_num)
    return str(output_path)


def write_text_report(report: str, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return str(path)


def make_zip(files: Iterable[str], output_zip: Path, base_dir: Optional[Path] = None) -> str:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in files:
            p = Path(file)
            if not p.exists() or p.is_dir():
                continue
            arcname = str(p.relative_to(base_dir)) if base_dir and p.is_relative_to(base_dir) else p.name
            zf.write(p, arcname)
    return str(output_zip)


def run_pipeline(input_path, output_dir: Path, station_name: str = "Estação analisada", start_year: int = 2021, end_year: int = 2025) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "graficos_png"

    raw = read_input_file(input_path)
    df = clean_qualar_dataframe(raw, start_year=start_year, end_year=end_year)
    pollutants, meteo, variables = identify_variables(df)
    if df.empty:
        raise ValueError("A base ficou vazia após limpeza. Verifique Data, Hora e o filtro de anos.")
    if not variables:
        raise ValueError("Nenhum poluente ou parâmetro meteorológico conhecido foi identificado após padronização.")

    tables = compute_tables(df, pollutants, meteo)
    report = generate_text_report(df, tables, pollutants, meteo, station_name=station_name)
    chart_paths = generate_figures(df, tables, figures_dir, pollutants, meteo)
    excel_path = output_dir / "QUALAR_tabelas_resultados.xlsx"
    text_path = output_dir / "RELATORIO_TECNICO_QUALAR.md"
    export_excel(tables, excel_path, text_report=report)
    write_text_report(report, text_path)

    all_files = [str(excel_path), str(text_path)] + chart_paths
    zip_path = output_dir / "QUALAR_outputs_tabelas_graficos_texto.zip"
    make_zip(all_files, zip_path, base_dir=output_dir)

    return {
        "df": df,
        "pollutants": pollutants,
        "meteo": meteo,
        "tables": tables,
        "report": report,
        "excel_path": str(excel_path),
        "text_path": str(text_path),
        "chart_paths": chart_paths,
        "zip_path": str(zip_path),
    }
