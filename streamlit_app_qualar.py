# streamlit_app_qualar.py
# ACH1026 - Levantamento e análise de dados de qualidade do ar - QUALAR/CETESB
# Autor: versão consolidada para análise exploratória 2021-2025

from __future__ import annotations

import glob
import io
import os
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats


# ============================================================
# CONFIGURAÇÃO GERAL
# ============================================================

st.set_page_config(
    page_title="QUALAR | Análise de Qualidade do Ar",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ANOS_PADRAO = list(range(2021, 2026))
MESES_PT = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
ESTACOES = {
    12: "Verão", 1: "Verão", 2: "Verão",
    3: "Outono", 4: "Outono", 5: "Outono",
    6: "Inverno", 7: "Inverno", 8: "Inverno",
    9: "Primavera", 10: "Primavera", 11: "Primavera",
}

POLUENTES_OBRIGATORIOS = [
    "MP10 (µg/m³)",
    "MP2.5 (µg/m³)",
    "NOx (ppb)",
    "SO2 (µg/m³)",
    "O3 (µg/m³)",
]

METEO_PADRAO = [
    "Temperatura (°C)",
    "Umidade relativa (%)",
    "Precipitação (mm)",
    "Velocidade do vento (m/s)",
    "Direção do vento (°)",
]

# Aparência dos gráficos exportados
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "figure.autolayout": True,
})
sns.set_theme(style="whitegrid", palette="muted")


# ============================================================
# UTILITÁRIOS DE TEXTO / NORMALIZAÇÃO
# ============================================================

def remover_acentos(texto: str) -> str:
    texto = str(texto)
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return texto


def normalizar_texto(texto: str) -> str:
    texto = remover_acentos(str(texto)).lower().strip()
    texto = texto.replace("µ", "u").replace("³", "3").replace("²", "2")
    texto = re.sub(r"\s+", " ", texto)
    return texto


def nome_seguro_arquivo(nome: str) -> str:
    nome = remover_acentos(nome)
    nome = re.sub(r"[^A-Za-z0-9_.-]+", "_", nome).strip("_")
    return nome or "grafico"


def classificar_coluna_qualar(coluna: str) -> str:
    """Padroniza nomes comuns do QUALAR sem apagar nomes desconhecidos."""
    bruto = str(coluna).strip()
    n = normalizar_texto(bruto)

    # Evita confundir Data/Hora com parâmetros
    if n in {"data", "date"}:
        return "Data"
    if n in {"hora", "hour", "time"}:
        return "Hora"

    # Poluentes principais do trabalho
    if "mp2.5" in n or "mp2,5" in n or "mp 2.5" in n or "mp 2,5" in n or "particulas inalaveis finas" in n:
        return "MP2.5 (µg/m³)"
    if "mp10" in n or "mp 10" in n or "particulas inalaveis" in n:
        return "MP10 (µg/m³)"
    if re.search(r"\bnox\b", n) or "oxidos de nitrogenio" in n or "oxidos de nitrogenio" in n:
        return "NOx (ppb)"
    if re.search(r"\bno2\b", n) or "dioxido de nitrogenio" in n:
        return "NO2 (ppb)"
    if re.search(r"\bno\b", n) and "nox" not in n:
        return "NO (ppb)"
    if re.search(r"\bso2\b", n) or "dioxido de enxofre" in n:
        return "SO2 (µg/m³)"
    if re.search(r"\bo3\b", n) or "ozonio" in n:
        return "O3 (µg/m³)"
    if re.search(r"\bco\b", n) or "monoxido de carbono" in n:
        return "CO (ppm)"

    # Meteorologia
    if n.startswith("temp") or "temperatura" in n:
        return "Temperatura (°C)"
    if n in {"ur", "ur - %"} or "umidade relativa" in n:
        return "Umidade relativa (%)"
    if n.startswith("vv") or "velocidade do vento" in n:
        return "Velocidade do vento (m/s)"
    if n.startswith("dv") or "direcao do vento" in n or "direção do vento" in n:
        return "Direção do vento (°)"
    if "precipitacao" in n or "precipitação" in n or n.startswith("prec") or "chuva" in n:
        return "Precipitação (mm)"
    if "pressao" in n or "pressão" in n:
        return "Pressão atmosférica (hPa)"
    if "radiacao" in n or "radiação" in n:
        return "Radiação solar"

    return bruto


def deduplicar_nomes_colunas(colunas: Iterable[str]) -> list[str]:
    vistos: dict[str, int] = {}
    saida: list[str] = []
    for col in colunas:
        col = str(col).strip() or "Sem_nome"
        if col not in vistos:
            vistos[col] = 0
            saida.append(col)
        else:
            vistos[col] += 1
            saida.append(f"{col}__dup{vistos[col]}")
    return saida


# ============================================================
# LEITURA ROBUSTA DOS CSVs QUALAR
# ============================================================

@dataclass
class ArquivoLido:
    nome: str
    df: pd.DataFrame
    estacao: str | None
    observacao: str


def ler_bytes_arquivo(origem: str | Path | BinaryIO) -> tuple[bytes, str]:
    """Aceita caminho local ou arquivo enviado pelo st.file_uploader."""
    if isinstance(origem, (str, Path)):
        caminho = Path(origem)
        return caminho.read_bytes(), caminho.name

    nome = getattr(origem, "name", "arquivo.csv")
    try:
        origem.seek(0)
    except Exception:
        pass
    return origem.read(), nome


def decodificar_csv(raw: bytes) -> str:
    for enc in ("latin1", "cp1252", "utf-8-sig", "utf-8"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin1", errors="replace")


def linha_parece_data(linha: str) -> bool:
    partes = linha.strip().split(";")
    if not partes:
        return False
    primeiro = partes[0].strip()
    return bool(re.match(r"^\d{1,2}/\d{1,2}/\d{2,4}$", primeiro))


def extrair_metadados_qualar(linhas: list[str]) -> dict[str, str]:
    meta: dict[str, str] = {}
    for linha in linhas[:25]:
        partes = [p.strip() for p in linha.split(";")]
        if len(partes) < 2:
            continue
        chave = normalizar_texto(partes[0]).replace(":", "")
        valor = partes[1].strip()
        if not valor:
            continue
        if "nome da estacao" in chave:
            meta["estacao"] = valor
        elif "codigo da estacao" in chave or "cod estacao" in chave:
            meta["codigo_estacao"] = valor
        elif "tipo de monitoramento" in chave:
            meta["tipo_monitoramento"] = valor
    return meta


def identificar_linha_dados(linhas: list[str]) -> tuple[int, int]:
    """Retorna (indice_linha_parametros, indice_primeira_linha_dados)."""
    for i, linha in enumerate(linhas):
        if linha_parece_data(linha):
            return max(i - 1, 0), i
    # fallback do QUALAR usado no script original
    if len(linhas) > 8:
        return 7, 8
    raise ValueError("Não foi possível localizar a primeira linha de dados com Data/Hora.")


def montar_colunas_qualar(linha_parametros: str, n_colunas_df: int) -> list[str]:
    partes = [p.strip() for p in linha_parametros.strip().split(";")]
    partes = [p for p in partes if p != ""]

    # Alguns CSVs já trazem Data/Hora na linha de cabeçalho; outros trazem só parâmetros.
    partes_norm = [normalizar_texto(p) for p in partes]
    contem_data = any(p == "data" for p in partes_norm)
    contem_hora = any(p == "hora" for p in partes_norm)

    if contem_data and contem_hora:
        colunas = [classificar_coluna_qualar(p) for p in partes]
    else:
        colunas = ["Data", "Hora"] + [classificar_coluna_qualar(p) for p in partes]

    if len(colunas) < n_colunas_df:
        colunas += [f"Extra_{i}" for i in range(1, n_colunas_df - len(colunas) + 1)]
    elif len(colunas) > n_colunas_df:
        colunas = colunas[:n_colunas_df]

    return deduplicar_nomes_colunas(colunas)


def limpar_hora(valor) -> str:
    if pd.isna(valor):
        return ""
    s = str(valor).strip()
    s = s.replace("24:00:00", "24:00").replace("00:00:00", "00:00")

    # Caso venha como fração de dia do Excel
    try:
        if re.fullmatch(r"\d*\.\d+", s):
            frac = float(s)
            total_min = int(round(frac * 24 * 60))
            h = (total_min // 60) % 24
            m = total_min % 60
            return f"{h:02d}:{m:02d}"
    except Exception:
        pass

    # 2400, 0000, 930 etc.
    if re.fullmatch(r"\d{3,4}", s):
        s = s.zfill(4)
        return f"{s[:2]}:{s[2:]}"

    # 0:00 -> 00:00
    m = re.match(r"^(\d{1,2}):(\d{2})", s)
    if m:
        return f"{int(m.group(1)):02d}:{int(m.group(2)):02d}"

    return s[:5]


def criar_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if "Data" not in df.columns or "Hora" not in df.columns:
        raise ValueError("O arquivo não contém as colunas Data e Hora após a leitura.")

    data = df["Data"]
    if pd.api.types.is_datetime64_any_dtype(data):
        data_str = data.dt.strftime("%d/%m/%Y")
    else:
        data_str = data.astype(str).str.strip()

    hora_str = df["Hora"].apply(limpar_hora)
    mask_24 = hora_str.str.startswith("24:", na=False)
    hora_parse = hora_str.mask(mask_24, hora_str.str.replace(r"^24:", "00:", regex=True))

    dt = pd.to_datetime(data_str + " " + hora_parse, format="%d/%m/%Y %H:%M", errors="coerce")
    dt.loc[mask_24 & dt.notna()] = dt.loc[mask_24 & dt.notna()] + pd.Timedelta(days=1)

    df = df.copy()
    df["Datetime"] = dt
    df["Data"] = df["Datetime"].dt.strftime("%d/%m/%Y")
    df["Hora"] = df["Datetime"].dt.strftime("%H:%M")
    return df.dropna(subset=["Datetime"])


def converter_numericos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ignorar = {"Data", "Hora", "Datetime"}
    for col in df.columns:
        if col in ignorar:
            continue
        if df[col].dtype == object:
            serie = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(".", "", regex=False)       # remove milhar, se existir
                .str.replace(",", ".", regex=False)      # decimal brasileiro
                .replace({"": np.nan, "nan": np.nan, "None": np.nan, "-": np.nan})
            )
            df[col] = pd.to_numeric(serie, errors="coerce")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def ler_arquivo_qualar(origem: str | Path | BinaryIO) -> ArquivoLido:
    raw, nome = ler_bytes_arquivo(origem)
    texto = decodificar_csv(raw)
    linhas = texto.splitlines()
    meta = extrair_metadados_qualar(linhas)

    idx_param, idx_dados = identificar_linha_dados(linhas)
    csv_dados = "\n".join(linhas[idx_dados:])

    df = pd.read_csv(
        io.StringIO(csv_dados),
        sep=";",
        header=None,
        decimal=",",
        engine="python",
        na_values=["", " ", "-", "--", "NaN", "nan"],
    )

    # Remove colunas completamente vazias oriundas de ;;;;; no fim da linha.
    df = df.dropna(axis=1, how="all")
    df.columns = montar_colunas_qualar(linhas[idx_param], len(df.columns))
    df = criar_datetime(df)
    df = converter_numericos(df)

    # Remove extras sem utilidade.
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Extra_")], errors="ignore")
    df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed", case=False, regex=True)]

    obs = f"{len(df):,} linhas válidas".replace(",", ".")
    return ArquivoLido(nome=nome, df=df, estacao=meta.get("estacao"), observacao=obs)


# ============================================================
# CONSOLIDAÇÃO
# ============================================================

def colunas_numericas(df: pd.DataFrame) -> list[str]:
    ignorar = {"Data", "Hora", "Datetime", "Ano", "Mês", "Mes", "Estação"}
    return [c for c in df.columns if c not in ignorar and pd.api.types.is_numeric_dtype(df[c])]


def consolidar_dataframes(lista_lidos: list[ArquivoLido]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Junta arquivos horizontalmente por Datetime e empilha anos naturalmente pelo índice temporal."""
    consolidado: Optional[pd.DataFrame] = None
    logs = []

    for item in lista_lidos:
        df = item.df.copy()
        if df.empty:
            logs.append({"arquivo": item.nome, "status": "vazio", "linhas": 0, "colunas": 0, "conflitos": 0})
            continue

        cols_num = colunas_numericas(df)
        df = df[["Datetime"] + cols_num].copy()
        df = df.groupby("Datetime", as_index=True).mean(numeric_only=True).sort_index()

        conflitos = 0
        if consolidado is None:
            consolidado = df
        else:
            cols_comuns = [c for c in df.columns if c in consolidado.columns]
            for col in cols_comuns:
                ambos = consolidado[col].notna() & df[col].notna()
                if ambos.any():
                    conflitos += int((np.abs(consolidado.loc[ambos, col] - df.loc[ambos, col]) > 1e-9).sum())
            consolidado = consolidado.combine_first(df)

        logs.append({
            "arquivo": item.nome,
            "status": "lido",
            "linhas": len(item.df),
            "colunas": len(cols_num),
            "conflitos": conflitos,
        })

    if consolidado is None:
        return pd.DataFrame(), pd.DataFrame(logs)

    consolidado = consolidado.sort_index()
    consolidado = consolidado.reset_index()
    consolidado["Data"] = consolidado["Datetime"].dt.strftime("%d/%m/%Y")
    consolidado["Hora"] = consolidado["Datetime"].dt.strftime("%H:%M")
    consolidado["Ano"] = consolidado["Datetime"].dt.year
    consolidado["Mês"] = consolidado["Datetime"].dt.month
    consolidado["Estação"] = consolidado["Mês"].map(ESTACOES)

    cols = ["Datetime", "Data", "Hora", "Ano", "Mês", "Estação"] + [c for c in consolidado.columns if c not in {"Datetime", "Data", "Hora", "Ano", "Mês", "Estação"}]
    consolidado = consolidado[cols]
    return consolidado, pd.DataFrame(logs)


@st.cache_data(show_spinner=False)
def carregar_por_upload(arquivos_upload) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    lidos: list[ArquivoLido] = []
    erros: list[str] = []
    estacoes: list[str] = []

    for arq in arquivos_upload:
        try:
            item = ler_arquivo_qualar(arq)
            lidos.append(item)
            if item.estacao:
                estacoes.append(item.estacao)
        except Exception as e:
            erros.append(f"{getattr(arq, 'name', 'arquivo')}: {e}")

    df, log = consolidar_dataframes(lidos)
    if erros:
        log_erros = pd.DataFrame([{"arquivo": e.split(":", 1)[0], "status": "erro", "erro": e} for e in erros])
        log = pd.concat([log, log_erros], ignore_index=True)
    return df, log, sorted(set(estacoes))


@st.cache_data(show_spinner=False)
def carregar_por_pasta(pasta_entrada: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    caminhos = sorted(glob.glob(os.path.join(pasta_entrada, "*.csv")))
    lidos: list[ArquivoLido] = []
    erros: list[str] = []
    estacoes: list[str] = []

    for caminho in caminhos:
        try:
            item = ler_arquivo_qualar(caminho)
            lidos.append(item)
            if item.estacao:
                estacoes.append(item.estacao)
        except Exception as e:
            erros.append(f"{Path(caminho).name}: {e}")

    df, log = consolidar_dataframes(lidos)
    if erros:
        log_erros = pd.DataFrame([{"arquivo": e.split(":", 1)[0], "status": "erro", "erro": e} for e in erros])
        log = pd.concat([log, log_erros], ignore_index=True)
    return df, log, sorted(set(estacoes))


# ============================================================
# ESTATÍSTICAS E DIAGNÓSTICO
# ============================================================

def tabela_descritiva(df: pd.DataFrame, variaveis: list[str]) -> pd.DataFrame:
    linhas = []
    total = len(df)
    for col in variaveis:
        serie = df[col].dropna()
        linhas.append({
            "Parâmetro": col,
            "N válido": int(serie.count()),
            "Falhas": int(total - serie.count()),
            "% válido": (serie.count() / total * 100) if total else np.nan,
            "Média": serie.mean(),
            "Desvio padrão": serie.std(),
            "Mínimo": serie.min(),
            "P25": serie.quantile(0.25),
            "Mediana": serie.median(),
            "P75": serie.quantile(0.75),
            "Máximo": serie.max(),
        })
    return pd.DataFrame(linhas)


def tabela_media_anual(df: pd.DataFrame, variaveis: list[str]) -> pd.DataFrame:
    if not variaveis:
        return pd.DataFrame()
    return df.groupby("Ano")[variaveis].mean(numeric_only=True).reset_index()


def tabela_tendencias(df: pd.DataFrame, variaveis: list[str]) -> pd.DataFrame:
    annual = tabela_media_anual(df, variaveis)
    linhas = []
    if annual.empty:
        return pd.DataFrame()

    for col in variaveis:
        tmp = annual[["Ano", col]].dropna()
        if len(tmp) >= 3:
            reg = stats.linregress(tmp["Ano"], tmp[col])
            direcao = "aumento" if reg.slope > 0 else "queda" if reg.slope < 0 else "estável"
            linhas.append({
                "Parâmetro": col,
                "inclinação por ano": reg.slope,
                "R²": reg.rvalue ** 2,
                "p-valor": reg.pvalue,
                "leitura exploratória": direcao,
            })
        else:
            linhas.append({
                "Parâmetro": col,
                "inclinação por ano": np.nan,
                "R²": np.nan,
                "p-valor": np.nan,
                "leitura exploratória": "dados insuficientes",
            })
    return pd.DataFrame(linhas)


def diagnostico_completude(df: pd.DataFrame, variaveis: list[str]) -> pd.DataFrame:
    total = len(df)
    linhas = []
    for col in variaveis:
        validos = int(df[col].notna().sum())
        perc = validos / total * 100 if total else 0
        if perc >= 90:
            status = "bom"
        elif perc >= 70:
            status = "aceitável"
        elif perc > 0:
            status = "fraco"
        else:
            status = "não monitorado"
        linhas.append({"Parâmetro": col, "N válido": validos, "% válido": perc, "status": status})
    return pd.DataFrame(linhas)


def periodo_texto(df: pd.DataFrame) -> str:
    if df.empty:
        return "sem dados"
    return f"{df['Datetime'].min():%d/%m/%Y} a {df['Datetime'].max():%d/%m/%Y}"


def interpretar_meteorologia(var_meteo: str, coef: float) -> str:
    sinal = "positiva" if coef > 0 else "negativa" if coef < 0 else "nula"
    intensidade = abs(coef)
    if intensidade >= 0.7:
        grau = "forte"
    elif intensidade >= 0.4:
        grau = "moderada"
    elif intensidade >= 0.2:
        grau = "fraca"
    else:
        grau = "muito fraca"
    return f"Correlação {sinal} {grau} com {var_meteo} (r={coef:.2f}). Não é prova de causalidade; serve como evidência exploratória."


# ============================================================
# EXPORTAÇÃO
# ============================================================

def excel_bytes(df: pd.DataFrame, variaveis: list[str], log: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="dados_consolidados", index=False)
        if variaveis:
            tabela_descritiva(df, variaveis).to_excel(writer, sheet_name="estatistica_descritiva", index=False)
            diagnostico_completude(df, variaveis).to_excel(writer, sheet_name="completude", index=False)
            tabela_media_anual(df, variaveis).to_excel(writer, sheet_name="medias_anuais", index=False)
            tabela_tendencias(df, variaveis).to_excel(writer, sheet_name="tendencias", index=False)
        log.to_excel(writer, sheet_name="log_leitura", index=False)
    return buffer.getvalue()


def salvar_excel_em_pasta(df: pd.DataFrame, variaveis: list[str], log: pd.DataFrame, pasta_saida: str) -> str:
    os.makedirs(pasta_saida, exist_ok=True)
    caminho = os.path.join(pasta_saida, "Planilha_Geral_Unica_Tudo.xlsx")
    with open(caminho, "wb") as f:
        f.write(excel_bytes(df, variaveis, log))
    return caminho


# ============================================================
# GRÁFICOS MATPLOTLIB / SEABORN
# ============================================================

def fig_serie_diaria(df: pd.DataFrame, param: str):
    d = df.set_index("Datetime")[param].resample("D").mean()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(d.index, d.values, linewidth=1.2)
    ax.set_title(f"Série temporal da média diária — {param}", loc="left", fontweight="bold")
    ax.set_ylabel(param)
    ax.set_xlabel("Ano")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.35)
    return fig


def fig_boxplot_mensal(df: pd.DataFrame, param: str):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    sns.boxplot(data=df, x="Mês", y=param, ax=ax, fliersize=2, linewidth=1.1)
    ax.set_title(f"Sazonalidade mensal — {param}", loc="left", fontweight="bold")
    ax.set_xlabel("Mês")
    ax.set_ylabel(param)
    ax.set_xticks(range(12))
    ax.set_xticklabels(MESES_PT)
    return fig


def fig_heatmap_correlacao(df: pd.DataFrame, variaveis: list[str]):
    corr = df[variaveis].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax, linewidths=0.5)
    ax.set_title("Matriz de correlação entre parâmetros", loc="left", fontweight="bold")
    return fig


def fig_polar_vento(df: pd.DataFrame, col_vel: str, col_dir: str):
    vento = df[[col_vel, col_dir]].dropna().copy()
    vento = vento[(vento[col_dir] >= 0) & (vento[col_dir] <= 360) & (vento[col_vel] >= 0)]
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, polar=True)
    if vento.empty:
        ax.set_title("Sem dados válidos de vento")
        return fig
    theta = np.radians(vento[col_dir] % 360)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    sc = ax.scatter(theta, vento[col_vel], c=vento[col_vel], cmap="YlOrRd", s=8, alpha=0.65, edgecolors="none")
    ax.set_title("Distribuição polar — direção e velocidade do vento", fontweight="bold", pad=18)
    ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    cbar = fig.colorbar(sc, ax=ax, pad=0.10, fraction=0.046)
    cbar.set_label("Velocidade (m/s)", rotation=270, labelpad=16)
    return fig


def fig_meteo_poluente(df: pd.DataFrame, poluente: str, meteo: str):
    tmp = df[[poluente, meteo]].dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    if tmp.empty:
        ax.set_title("Sem dados pareados")
        return fig
    sns.regplot(data=tmp.sample(min(len(tmp), 5000), random_state=42), x=meteo, y=poluente, ax=ax, scatter_kws={"s": 8, "alpha": 0.35}, line_kws={"linewidth": 1.8})
    ax.set_title(f"Relação exploratória: {poluente} x {meteo}", loc="left", fontweight="bold")
    return fig


def png_bytes(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def zip_graficos(df: pd.DataFrame, variaveis: list[str], incluir_vento: bool = True) -> bytes:
    buffer_zip = io.BytesIO()
    with zipfile.ZipFile(buffer_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for param in variaveis:
            if df[param].notna().sum() == 0:
                continue
            nome = nome_seguro_arquivo(param)
            zf.writestr(f"Serie_Diaria_{nome}.png", png_bytes(fig_serie_diaria(df, param)))
            zf.writestr(f"Boxplot_Mensal_{nome}.png", png_bytes(fig_boxplot_mensal(df, param)))

        if len(variaveis) >= 2:
            zf.writestr("Heatmap_Correlacao.png", png_bytes(fig_heatmap_correlacao(df, variaveis)))

        if incluir_vento and "Velocidade do vento (m/s)" in df.columns and "Direção do vento (°)" in df.columns:
            zf.writestr("Distribuicao_Polar_Vento.png", png_bytes(fig_polar_vento(df, "Velocidade do vento (m/s)", "Direção do vento (°)")))
    buffer_zip.seek(0)
    return buffer_zip.getvalue()


# ============================================================
# INTERFACE STREAMLIT
# ============================================================

st.title("🌫️ QUALAR — Consolidação e análise de qualidade do ar")
st.caption("ACH1026 | 2021–2025 | Estatística descritiva, sazonalidade, tendência e meteorologia")

with st.sidebar:
    st.header("1) Entrada dos dados")
    modo = st.radio(
        "Como carregar os CSVs?",
        ["Upload no Streamlit", "Pasta local do computador"],
        help="Use pasta local se estiver rodando o app no seu próprio Windows. Em nuvem, use upload.",
    )

    arquivos_upload = []
    pasta_entrada = ""
    pasta_saida = r"C:\Users\eduar\Downloads\consolidados"

    if modo == "Upload no Streamlit":
        arquivos_upload = st.file_uploader(
            "Envie os CSVs baixados do QUALAR",
            type=["csv"],
            accept_multiple_files=True,
        )
    else:
        pasta_entrada = st.text_input("Pasta de entrada", value=r"C:\Users\eduar\Downloads\qualar")
        pasta_saida = st.text_input("Pasta de saída", value=r"C:\Users\eduar\Downloads\consolidados")

    st.header("2) Filtro do trabalho")
    anos = st.multiselect("Anos analisados", options=list(range(2010, 2031)), default=ANOS_PADRAO)
    st.info("O enunciado pede 2021–2025. Mude apenas se precisar conferir dados fora do recorte.")

    st.header("3) Estação")
    estacao_manual = st.text_input("Nome da estação", value="")
    lat = st.number_input("Latitude da estação", value=0.0, format="%.6f")
    lon = st.number_input("Longitude da estação", value=0.0, format="%.6f")

    st.header("4) Fontes no entorno")
    fontes_moveis = st.text_area(
        "Fontes móveis prováveis",
        value="Ex.: avenidas, corredores de ônibus, rodovias, tráfego de veículos leves e pesados.",
        height=80,
    )
    fontes_fixas = st.text_area(
        "Fontes fixas prováveis",
        value="Ex.: indústrias, caldeiras, terminais, obras, atividades comerciais ou ausência de fontes fixas relevantes.",
        height=80,
    )


# Carregamento efetivo
carregou = False
log = pd.DataFrame()
estacoes_detectadas: list[str] = []

if modo == "Upload no Streamlit" and arquivos_upload:
    with st.spinner("Lendo e consolidando arquivos enviados..."):
        df, log, estacoes_detectadas = carregar_por_upload(arquivos_upload)
        carregou = not df.empty
elif modo == "Pasta local do computador" and pasta_entrada:
    if st.sidebar.button("Ler CSVs da pasta local", type="primary"):
        with st.spinner("Lendo e consolidando arquivos da pasta local..."):
            df, log, estacoes_detectadas = carregar_por_pasta(pasta_entrada)
            carregou = not df.empty
    else:
        df = pd.DataFrame()
else:
    df = pd.DataFrame()

if not carregou:
    st.info("Carregue os CSVs do QUALAR para iniciar a análise.")
    st.stop()

# Filtro temporal
if anos:
    df = df[df["Ano"].isin(anos)].copy()

if df.empty:
    st.error("Após o filtro de anos, não sobrou dado. Confira se os arquivos são mesmo de 2021 a 2025.")
    st.stop()

# Identificação de variáveis disponíveis
variaveis_num = colunas_numericas(df)
poluentes_disp = [c for c in POLUENTES_OBRIGATORIOS if c in variaveis_num]
meteo_disp = [c for c in METEO_PADRAO if c in variaveis_num]
outros_disp = [c for c in variaveis_num if c not in poluentes_disp + meteo_disp]

estacao_detectada = ", ".join(estacoes_detectadas) if estacoes_detectadas else ""
estacao_nome = estacao_manual.strip() or estacao_detectada or "Estação não identificada"

# Seleção de parâmetros
st.sidebar.header("5) Parâmetros")
selecionados = st.sidebar.multiselect(
    "Parâmetros para análise",
    options=variaveis_num,
    default=poluentes_disp[:5] if poluentes_disp else variaveis_num[:min(5, len(variaveis_num))],
)

if not selecionados:
    st.warning("Selecione ao menos um parâmetro numérico.")
    st.stop()

# KPIs
st.subheader(f"Estação analisada: {estacao_nome}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Período", periodo_texto(df))
col2.metric("Registros horários", f"{len(df):,}".replace(",", "."))
col3.metric("Parâmetros numéricos", len(variaveis_num))
col4.metric("Poluentes do enunciado encontrados", f"{len(poluentes_disp)}/5")

if len(poluentes_disp) < 3:
    st.warning(
        "O enunciado exige pelo menos 3 poluentes. Esta base, do jeito que foi lida, tem menos de 3 entre MP10, MP2.5, NOx, SO2 e O3. "
        "Confira se a estação monitora esses parâmetros ou se faltam CSVs."
    )

if lat != 0.0 and lon != 0.0:
    with st.expander("Mapa simples da estação"):
        st.map(pd.DataFrame({"lat": [lat], "lon": [lon], "estação": [estacao_nome]}), latitude="lat", longitude="lon", zoom=12)

# Abas principais
aba1, aba2, aba3, aba4, aba5, aba6 = st.tabs([
    "Base consolidada",
    "Estatística descritiva",
    "Séries e sazonalidade",
    "Meteorologia",
    "Conclusões guiadas",
    "Exportação",
])

with aba1:
    st.markdown("### Log de leitura")
    st.dataframe(log, use_container_width=True)

    st.markdown("### Prévia da base consolidada")
    st.dataframe(df.head(500), use_container_width=True)

    st.markdown("### Completude por parâmetro")
    comp = diagnostico_completude(df, variaveis_num)
    st.dataframe(comp.style.format({"% válido": "{:.1f}"}), use_container_width=True)

    st.markdown("### Visual nativo do Streamlit: registros válidos por parâmetro")
    comp_chart = comp.set_index("Parâmetro")[["% válido"]]
    st.bar_chart(comp_chart)

with aba2:
    st.markdown("### Tabela obrigatória — estatística descritiva")
    desc = tabela_descritiva(df, selecionados)
    st.dataframe(
        desc.style.format({
            "% válido": "{:.1f}",
            "Média": "{:.2f}",
            "Desvio padrão": "{:.2f}",
            "Mínimo": "{:.2f}",
            "P25": "{:.2f}",
            "Mediana": "{:.2f}",
            "P75": "{:.2f}",
            "Máximo": "{:.2f}",
        }),
        use_container_width=True,
    )

    st.markdown("### Médias anuais")
    anual = tabela_media_anual(df, selecionados)
    st.dataframe(anual.style.format({c: "{:.2f}" for c in selecionados}), use_container_width=True)
    if not anual.empty:
        st.line_chart(anual.set_index("Ano")[selecionados])

    st.markdown("### Tendência linear exploratória")
    tend = tabela_tendencias(df, selecionados)
    st.dataframe(
        tend.style.format({"inclinação por ano": "{:.4f}", "R²": "{:.3f}", "p-valor": "{:.3g}"}),
        use_container_width=True,
    )
    st.caption("Use a tendência como apoio descritivo, não como prova causal. Cinco anos é uma janela curta.")

with aba3:
    param = st.selectbox("Parâmetro para gráficos", selecionados)

    st.markdown("### Série diária — gráfico nativo do Streamlit")
    diario = df.set_index("Datetime")[[param]].resample("D").mean()
    st.line_chart(diario)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Série diária — gráfico acadêmico")
        st.pyplot(fig_serie_diaria(df, param), clear_figure=True)
    with c2:
        st.markdown("### Boxplot mensal")
        st.pyplot(fig_boxplot_mensal(df, param), clear_figure=True)

    st.markdown("### Médias mensais por ano — visual nativo")
    mensal_ano = df.groupby(["Ano", "Mês"])[param].mean().reset_index()
    if not mensal_ano.empty:
        pivot = mensal_ano.pivot(index="Mês", columns="Ano", values=param).reindex(range(1, 13))
        pivot.index = MESES_PT
        st.line_chart(pivot)

    if len(selecionados) >= 2:
        st.markdown("### Correlação entre parâmetros")
        st.pyplot(fig_heatmap_correlacao(df, selecionados), clear_figure=True)

with aba4:
    st.markdown("### Parâmetros meteorológicos encontrados")
    if meteo_disp:
        st.write(", ".join(meteo_disp))
    else:
        st.warning("Não foram encontrados parâmetros meteorológicos padronizados. Se a estação não mede meteorologia, use INMET/CGE e junte por Data/Hora.")

    if "Velocidade do vento (m/s)" in df.columns and "Direção do vento (°)" in df.columns:
        st.markdown("### Distribuição polar do vento")
        st.pyplot(fig_polar_vento(df, "Velocidade do vento (m/s)", "Direção do vento (°)"), clear_figure=True)

    if meteo_disp and poluentes_disp:
        st.markdown("### Relação poluente x meteorologia")
        pol = st.selectbox("Poluente", poluentes_disp, key="pol_meteo")
        met = st.selectbox("Meteorologia", meteo_disp, key="met_meteo")
        par = df[[pol, met]].dropna()
        if len(par) >= 10:
            r = par[pol].corr(par[met])
            st.metric("Correlação de Pearson", f"{r:.2f}")
            st.write(interpretar_meteorologia(met, r))
            st.scatter_chart(par.sample(min(len(par), 5000), random_state=42), x=met, y=pol)
            st.pyplot(fig_meteo_poluente(df, pol, met), clear_figure=True)
        else:
            st.warning("Poucos pares válidos para correlação.")

    st.markdown("### Orientação técnica para meteorologia")
    st.markdown(
        "- Temperatura e radiação tendem a ser relevantes para O3, pois o ozônio é secundário e fotoquímico.\n"
        "- Umidade, chuva e vento costumam alterar dispersão, deposição e ressuspensão de partículas.\n"
        "- Direção do vento deve ser cruzada com fontes no entorno: avenidas, corredores de ônibus, rodovias, indústrias e obras.\n"
        "- Não force causalidade apenas por correlação. Use como evidência auxiliar no relatório."
    )

with aba5:
    st.markdown("### Rascunho técnico para orientar o relatório")
    desc_sel = tabela_descritiva(df, selecionados)
    tend_sel = tabela_tendencias(df, selecionados)

    pior_completude = diagnostico_completude(df, selecionados).sort_values("% válido").head(1)
    pior_txt = ""
    if not pior_completude.empty:
        pior_txt = f"O parâmetro com menor completude foi {pior_completude.iloc[0]['Parâmetro']} ({pior_completude.iloc[0]['% válido']:.1f}% válido)."

    tendencia_txts = []
    for _, row in tend_sel.iterrows():
        if pd.notna(row.get("inclinação por ano")):
            tendencia_txts.append(
                f"{row['Parâmetro']}: {row['leitura exploratória']} ({row['inclinação por ano']:.3g} unidade/ano; R²={row['R²']:.2f})."
            )
    tendencia_txt = " ".join(tendencia_txts) if tendencia_txts else "As tendências não puderam ser estimadas com segurança para todos os parâmetros."

    texto = f"""
**Recorte analisado.** A estação analisada foi **{estacao_nome}**, com dados horários no período de **{periodo_texto(df)}**. Foram identificados **{len(poluentes_disp)}** dos 5 poluentes solicitados no enunciado: {', '.join(poluentes_disp) if poluentes_disp else 'nenhum dos poluentes principais padronizados'}.

**Qualidade da base.** {pior_txt} A estatística descritiva deve ser interpretada considerando as falhas de medição, principalmente quando a completude for inferior a 70%.

**Tendência.** {tendencia_txt} Como o período é curto, a tendência deve ser apresentada como descritiva/exploratória, não como conclusão causal forte.

**Fontes móveis no entorno.** {fontes_moveis}

**Fontes fixas no entorno.** {fontes_fixas}

**Meteorologia.** A interpretação deve cruzar sazonalidade, vento, temperatura, umidade e precipitação. Meses secos e com menor dispersão podem elevar material particulado; períodos quentes e com radiação favorecem análise de O3; vento e direção do vento ajudam a discutir transporte local dos poluentes.

**Conclusão esperada.** O relatório deve responder: quais poluentes foram mais críticos, em quais meses/anos ocorreram maiores valores, se houve aumento ou queda, se as lacunas comprometem a análise, e quais fontes próximas são plausíveis para explicar o padrão observado.
"""
    st.markdown(texto)

    st.markdown("### Checklist do trabalho")
    checklist = pd.DataFrame({
        "Item exigido": [
            "Escolher 1 estação CETESB",
            "Analisar 2021–2025",
            "Usar MP10, MP2.5, NOx, SO2 e O3 quando monitorados",
            "Ter pelo menos 3 poluentes",
            "Tabela com média, desvio padrão e número de medidas",
            "Analisar fontes fixas e móveis no entorno",
            "Usar meteorologia e sazonalidade",
            "Usar relatórios CETESB e bibliografia",
            "Conclusões sobre tendências e limitações",
        ],
        "Status no app": [
            "preencher/confirmar",
            "feito pelo filtro de anos",
            "detectado automaticamente quando os nomes vêm do QUALAR",
            "ver alerta no topo",
            "feito na aba Estatística descritiva",
            "preencher na barra lateral",
            "feito na aba Meteorologia",
            "inserir no texto final do relatório",
            "rascunho gerado nesta aba",
        ],
    })
    st.dataframe(checklist, use_container_width=True)

with aba6:
    st.markdown("### Baixar planilha consolidada")
    st.download_button(
        "⬇️ Baixar Excel consolidado",
        data=excel_bytes(df, selecionados, log),
        file_name="Planilha_Geral_Unica_Tudo.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.markdown("### Baixar gráficos em PNG 300 DPI")
    st.download_button(
        "⬇️ Baixar pacote de gráficos (.zip)",
        data=zip_graficos(df, selecionados),
        file_name="Graficos_QUALAR_300DPI.zip",
        mime="application/zip",
    )

    if modo == "Pasta local do computador":
        if st.button("Salvar Excel na pasta de saída local"):
            try:
                caminho = salvar_excel_em_pasta(df, selecionados, log, pasta_saida)
                st.success(f"Arquivo salvo em: {caminho}")
            except Exception as e:
                st.error(f"Falha ao salvar: {e}")

    st.markdown("### Código de uso")
    st.code("pip install -r requirements.txt\nstreamlit run streamlit_app_qualar.py", language="bash")
