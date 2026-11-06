"""Funções comuns dos testes: deflação, governos, leitura do RTN e cores."""
from datetime import date
from pathlib import Path

import pandas as pd

DADOS = Path(__file__).resolve().parent / "dados"
SAIDA = Path(__file__).resolve().parent / "saida"
SAIDA.mkdir(exist_ok=True)

ANO_BASE = 2025  # valores em reais (média) de 2025

# Paleta categórica de referência (validada: CVD e visão normal, pares adjacentes)
CORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TINTA = {"primaria": "#0b0b0b", "secundaria": "#52514e", "muted": "#898781",
         "grade": "#e1e0d9", "eixo": "#c3c2b7", "superficie": "#fcfcfb", "faixa": "#f0efec"}

GOVERNOS = [
    ("FHC 1", date(1995, 1, 1), date(1998, 12, 31)),
    ("FHC 2", date(1999, 1, 1), date(2002, 12, 31)),
    ("Lula 1", date(2003, 1, 1), date(2006, 12, 31)),
    ("Lula 2", date(2007, 1, 1), date(2010, 12, 31)),
    ("Dilma 1", date(2011, 1, 1), date(2014, 12, 31)),
    ("Dilma 2", date(2015, 1, 1), date(2016, 8, 31)),
    ("Temer", date(2016, 8, 31), date(2018, 12, 31)),
    ("Bolsonaro", date(2019, 1, 1), date(2022, 12, 31)),
    ("Lula 3", date(2023, 1, 1), date(2026, 12, 31)),
]
ANOS_ELEICAO_GERAL = [1998, 2002, 2006, 2010, 2014, 2018, 2022, 2026]


def governo_do_ano(ano: int) -> str:
    """Governo que esteve a maior parte do ano no poder (2016 -> Dilma 2, 8 meses)."""
    meio = date(ano, 7, 1)
    if ano == 2016:
        return "Dilma 2"
    return next(g for g, i, f in GOVERNOS if i <= meio <= f)


def governo_da_data(d: pd.Timestamp) -> str:
    d = d.date()
    return next(g for g, i, f in GOVERNOS if i <= d <= f)


# ------------------------------------------------------------------ IPCA
def _indice_ipca() -> pd.Series:
    ipca = pd.read_parquet(DADOS / "ipca_mensal.parquet").sort_values("data")
    return pd.Series((1 + ipca.variacao_pct.values / 100).cumprod(), index=ipca.data.values)


def fator_anual(ano_base: int = ANO_BASE) -> pd.Series:
    """ano -> fator que leva valores nominais do ano para reais (média) do ano-base."""
    idx = _indice_ipca()
    medio = idx.groupby(idx.index.year).mean()
    return medio[ano_base] / medio


def fator_jul2026_para_base(ano_base: int = ANO_BASE) -> float:
    """O RTN vem em R$ de jul/2026; converte para R$ médios do ano-base."""
    idx = _indice_ipca()
    return idx[idx.index.year == ano_base].mean() / idx[pd.Timestamp("2026-07-01")]


# ------------------------------------------------------------------ RTN (Tesouro)
def rtn_mensal(codigos: dict[str, str]) -> pd.DataFrame:
    """Lê linhas do RTN (tabela 1.2-A, R$ milhões de jul/2026) pelo código da linha.

    `codigos`: {"4.4.1.2": "Bolsa Família", ...}. Devolve R$ milhões do ANO_BASE,
    índice mensal, uma coluna por série.
    """
    bruto = pd.read_excel(DADOS / "rtn" / "rtn_mensal.xlsx", "1.2-A", header=None)
    datas = pd.to_datetime(bruto.iloc[4, 1:].values)
    rotulos = bruto[0].astype(str).str.strip()
    saida = {}
    for cod, nome in codigos.items():
        linha = rotulos[rotulos.str.match(rf"^{cod.replace('.', r'\.')}\s")]
        if len(linha) != 1:
            raise ValueError(f"Linha {cod} encontrada {len(linha)} vezes no RTN")
        saida[nome] = pd.to_numeric(bruto.iloc[linha.index[0], 1:], errors="coerce").values
    df = pd.DataFrame(saida, index=datas).astype(float)
    return df * fator_jul2026_para_base()
