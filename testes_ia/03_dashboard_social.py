"""Teste – Pergunta 2: gasto social por mandato, com análise de séries temporais
(tendência, sazonalidade e ciclos).

Fontes:
  * Tesouro Nacional – RTN, séries MENSAIS da União (pago), R$ de 2025
  * Base dos Dados – br_me_siconfi, função 08 por subfunção (empenhado), 2013–2025

Saída: saida/dashboard_social.html (abre no navegador, funciona offline)
"""
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.signal import periodogram
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.seasonal import STL

from _comum import (ANO_BASE, ANOS_ELEICAO_GERAL, CORES, DADOS, GOVERNOS, SAIDA, TINTA,
                    fator_anual, governo_do_ano, rtn_mensal)

# =============================================================== dados
PROGRAMAS = {
    "4.4.1.2": "Bolsa Família / Auxílio Brasil",
    "4.3.05": "BPC / LOAS",
    "4.3.01.1": "Abono salarial",
    "4.3.01.2": "Seguro-desemprego",
    "4.4.2.8": "Assistência social (discricionária)",
}
NOMES = list(PROGRAMAS.values())
COR = dict(zip(NOMES, CORES))
COR_TOTAL = TINTA["primaria"]
ULTIMO_MES = pd.Timestamp("2026-07-01")

bruto = rtn_mensal({**PROGRAMAS, "4.3.07": "Créditos extraordinários"}) / 1e3  # R$ bi de 2025
bruto = bruto.loc[:ULTIMO_MES]
# Antes de cada série começar a ser publicada o RTN traz zero -> vira "sem dado"
for c in bruto.columns:
    inicio = bruto.index[bruto[c].abs() > 0.001][0]
    bruto.loc[: inicio - pd.offsets.MonthBegin(1), c] = np.nan
prog = bruto[NOMES]
cred_extra = bruto["Créditos extraordinários"]
total = prog.loc["2009":].sum(axis=1)  # todas as 5 séries existem a partir de 2009
total.name = "Total dos 5 programas"

anual = prog.loc["2009":].resample("YE").sum()
anual.index = anual.index.year
anual["Total"] = anual[NOMES].sum(axis=1)
anual["governo"] = anual.index.map(governo_do_ano)
anual["parcial"] = anual.index == 2026

# Maior ano por mandato (anos completos)
completos = anual[~anual.parcial]
maiores = []
for gov, d in completos.groupby("governo", sort=False):
    ano = int(d.Total.idxmax())
    comp = d.loc[ano, NOMES].sort_values(ascending=False)
    maiores.append({
        "governo": gov, "anos": f"{d.index.min()}–{d.index.max()}", "ano": ano,
        "total": d.loc[ano, "Total"],
        "composicao": comp,
    })

# SICONFI – função 08 por subfunção
fator = fator_anual()
sic = pd.read_parquet(DADOS / "siconfi_assistencia.parquet")
sic["real_bi"] = sic.valor * sic.ano.map(fator) / 1e9
# Em 2025 a classificação mudou: o Bolsa Família saiu de 244 (Assistência Comunitária)
# para a nova 246 (Segurança e Renda). Juntamos as duas para manter a série comparável.
NOME_SUB = {
    "3.08.241": "Idoso", "3.08.242": "Pessoa com deficiência", "3.08.243": "Criança e adolescente",
    "3.08.244": "Assist. comunitária / Segurança de renda", "3.08.246": "Assist. comunitária / Segurança de renda",
}
sub = sic[sic.id_conta_bd.str.startswith("3.08.") & (sic.id_conta_bd != "3.08.000") & (sic.ano >= 2013)].copy()
sub["sub"] = sub.id_conta_bd.map(NOME_SUB).fillna("Demais subfunções")
func08 = sic[sic.id_conta_bd == "3.08.000"].pivot_table(index="ano", columns="esfera", values="real_bi")
uniao08 = func08["União"].dropna()


# =============================================================== análise de séries
def stl_log(serie: pd.Series):
    """STL multiplicativa (log), período 12, robusta a outliers (ex.: 2020)."""
    x = np.log(serie.dropna().clip(lower=1e-3))
    r = STL(x, period=12, robust=True).fit()
    forca_saz = max(0.0, 1 - r.resid.var() / (r.seasonal + r.resid).var())
    forca_tend = max(0.0, 1 - r.resid.var() / (r.trend + r.resid).var())
    return r, forca_tend, forca_saz


SERIES_STL = {**{n: prog[n] for n in NOMES}, total.name: total}
stl = {n: stl_log(s) for n, s in SERIES_STL.items()}
perfil = pd.DataFrame({
    n: np.exp(pd.Series(r.seasonal.values, index=r.seasonal.index.month).groupby(level=0).mean()) * 100
    for n, (r, _, _) in stl.items()
}).T
MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
perfil.columns = MESES
forcas = pd.DataFrame({n: {"Tendência": ft, "Sazonalidade": fs} for n, (_, ft, fs) in stl.items()}).T

# Ciclos: crescimento anual real em anos de eleição geral x demais
cresc = completos.Total.pct_change().mul(100).dropna()
eleicao = cresc.index.isin(ANOS_ELEICAO_GERAL)
med_el, med_out = cresc[eleicao].median(), cresc[~eleicao].median()

# Ciclos: periodograma do componente cíclico (filtro HP) do log do total
ciclo, _ = hpfilter(np.log(total), lamb=129600)
freq, pot = periodogram(ciclo.values, detrend="linear")
per = pd.Series(pot[1:], index=1 / freq[1:]).sort_index()
per = per[per.index <= 120]
picos = per.sort_values(ascending=False).head(3)

# =============================================================== estilo comum
FONTE = "system-ui, -apple-system, Segoe UI, sans-serif"


def layout(fig, titulo=None, altura=420, **kw):
    fig.update_layout(
        title=dict(text=titulo, x=0, xanchor="left", font=dict(size=15, color=TINTA["primaria"])) if titulo else None,
        height=altura, margin=dict(l=60, r=30, t=60 if titulo else 30, b=50),
        paper_bgcolor=TINTA["superficie"], plot_bgcolor=TINTA["superficie"],
        font=dict(family=FONTE, size=12, color=TINTA["secundaria"]),
        hoverlabel=dict(bgcolor="white", font=dict(family=FONTE, color=TINTA["primaria"])),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, bgcolor="rgba(0,0,0,0)"),
        **kw,
    )
    fig.update_xaxes(showgrid=False, linecolor=TINTA["eixo"], tickcolor=TINTA["eixo"], ticks="outside")
    fig.update_yaxes(gridcolor=TINTA["grade"], zeroline=False, linecolor=TINTA["eixo"])
    return fig


def faixas(fig, inicio="2008-07-01", rotulos=True, linha=None):
    """Faixas alternadas por mandato no eixo de datas."""
    for k, (nome, ini, fim) in enumerate(GOVERNOS):
        if pd.Timestamp(fim) < pd.Timestamp(inicio):
            continue
        x0 = max(pd.Timestamp(ini), pd.Timestamp(inicio))
        if k % 2 == 0:
            fig.add_vrect(x0=x0, x1=pd.Timestamp(fim), fillcolor=TINTA["faixa"], line_width=0, layer="below",
                          row=linha, col=1 if linha else None)
        if rotulos:
            fig.add_annotation(x=x0 + (pd.Timestamp(fim) - x0) / 2, y=1.0, yref="paper", yanchor="bottom",
                               text=nome, showarrow=False, font=dict(size=11, color=TINTA["secundaria"]))


bi = lambda v: f"R$ {v:,.1f} bi".replace(",", "X").replace(".", ",").replace("X", ".")  # noqa: E731

# =============================================================== 1. séries mensais
f1 = go.Figure()
faixas(f1, inicio="1999-01-01")
for n in NOMES:
    s = prog[n].rolling(12).sum()
    f1.add_scatter(x=s.index, y=s, name=n, line=dict(color=COR[n], width=2),
                   hovertemplate="%{y:,.1f} bi<extra>" + n + "</extra>")
t12 = total.rolling(12).sum()
f1.add_scatter(x=t12.index, y=t12, name="Total dos 5 programas", line=dict(color=COR_TOTAL, width=2.5),
               hovertemplate="%{y:,.1f} bi<extra>Total</extra>")
c12 = cred_extra.rolling(12).sum()
f1.add_scatter(x=c12.index, y=c12, name="Créditos extraordinários (Covid/Auxílio Emergencial)",
               line=dict(color=TINTA["muted"], width=1.5, dash="dot"), visible="legendonly",
               hovertemplate="%{y:,.1f} bi<extra>Créd. extraordinários</extra>")
layout(f1, f"Gasto social federal — soma móvel de 12 meses (R$ bilhões de {ANO_BASE})", altura=520,
       hovermode="x unified")
f1.update_layout(legend=dict(y=-0.32, yanchor="top"), margin=dict(t=80, b=60))
f1.update_xaxes(range=["1999-01-01", "2026-08-01"],
                rangeselector=dict(buttons=[dict(count=5, label="5 anos", step="year", stepmode="backward"),
                                            dict(count=10, label="10 anos", step="year", stepmode="backward"),
                                            dict(step="all", label="Tudo")],
                                   x=1, xanchor="right", y=1.13, bgcolor=TINTA["faixa"],
                                   activecolor=TINTA["grade"]))

# =============================================================== 2. maior ano por mandato (barras empilhadas)
f2 = go.Figure()
for n in NOMES:
    f2.add_bar(x=anual.index, y=anual[n], name=n, marker=dict(color=COR[n], line=dict(color=TINTA["superficie"], width=1)),
               hovertemplate="%{y:,.1f} bi<extra>" + n + "</extra>")
for m in maiores:
    f2.add_annotation(x=m["ano"], y=m["total"], text=f"▲<br>{m['governo']}", yshift=4, yanchor="bottom",
                      showarrow=False, font=dict(size=11, color=TINTA["primaria"]))
f2.add_annotation(x=2026, y=anual.loc[2026, "Total"], text="parcial<br>(jan–jul)", yshift=6, yanchor="bottom",
                  showarrow=False, font=dict(size=10, color=TINTA["muted"]))
layout(f2, f"Gasto anual por programa e o ano de maior gasto em cada mandato (R$ bi de {ANO_BASE})",
       altura=480, barmode="stack", hovermode="x unified", bargap=0.25)
f2.update_layout(legend=dict(y=-0.12, yanchor="top"), margin=dict(t=70, b=40))
f2.update_xaxes(dtick=1, tickangle=0)

# =============================================================== 3. decomposição STL
f3 = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                   subplot_titles=("Observado (R$ bi/mês)", "Tendência (R$ bi/mês)",
                                   "Sazonalidade (% do nível)", "Resíduo (% do nível)"))
traces_por_serie = []
for i, (n, (r, _, _)) in enumerate(stl.items()):
    cor = COR.get(n, COR_TOTAL)
    vis = i == len(stl) - 1  # começa mostrando o total
    idx = r.observed.index
    tr = [
        go.Scatter(x=idx, y=np.exp(r.observed), line=dict(color=cor, width=1.2), visible=vis,
                   hovertemplate="%{y:,.2f} bi<extra>observado</extra>"),
        go.Scatter(x=idx, y=np.exp(r.trend), line=dict(color=cor, width=2.5), visible=vis,
                   hovertemplate="%{y:,.2f} bi<extra>tendência</extra>"),
        go.Scatter(x=idx, y=np.exp(r.seasonal) * 100, line=dict(color=cor, width=1.2), visible=vis,
                   hovertemplate="%{y:.0f}%<extra>sazonal</extra>"),
        go.Scatter(x=idx, y=np.exp(r.resid) * 100, mode="markers", marker=dict(color=cor, size=4), visible=vis,
                   hovertemplate="%{y:.0f}%<extra>resíduo</extra>"),
    ]
    for linha, t in enumerate(tr, start=1):
        f3.add_trace(t, row=linha, col=1)
    traces_por_serie.append(len(tr))
botoes = []
n_total = sum(traces_por_serie)
for i, n in enumerate(stl):
    vis = [False] * n_total
    for k in range(4):
        vis[i * 4 + k] = True
    botoes.append(dict(label=n, method="update", args=[{"visible": vis}]))
layout(f3, None, altura=760, showlegend=False, hovermode="x unified")
f3.update_layout(margin=dict(t=90),
                 updatemenus=[dict(buttons=botoes, active=len(stl) - 1, x=0, y=1.1, xanchor="left",
                                   bgcolor="white", bordercolor=TINTA["eixo"])])
for linha in (3, 4):
    f3.add_hline(y=100, line=dict(color=TINTA["eixo"], width=1), row=linha, col=1)
f3.update_annotations(font=dict(size=12, color=TINTA["secundaria"]))

# =============================================================== 4. perfil sazonal (heatmap)
z = np.log2(perfil.values / 100).clip(-2, 2)
f4 = go.Figure(go.Heatmap(
    z=z, x=MESES, y=perfil.index, text=perfil.round(0).astype(int).astype(str) + "%",
    texttemplate="%{text}", textfont=dict(size=11),
    colorscale=[[0, "#1c5cab"], [0.25, "#86b6ef"], [0.5, "#f0efec"], [0.75, "#f2a09f"], [1, "#b02e2e"]],
    zmin=-2, zmax=2, xgap=2, ygap=2,
    colorbar=dict(title="", tickvals=[-2, -1, 0, 1, 2], ticktext=["≤25%", "50%", "100%", "200%", "≥400%"], len=0.9),
    hovertemplate="%{y}<br>%{x}: <b>%{text}</b> do nível normal<extra></extra>",
))
layout(f4, "Perfil sazonal: quanto cada mês gasta em relação ao nível de tendência (100% = mês típico)",
       altura=380)
f4.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
f4.update_layout(margin=dict(l=250))

# =============================================================== 5. força de tendência x sazonalidade
f5 = go.Figure()
for k, comp in enumerate(["Tendência", "Sazonalidade"]):
    f5.add_bar(y=forcas.index, x=forcas[comp], name=comp, orientation="h", marker_color=CORES[k],
               text=forcas[comp].map(lambda v: f"{v:.2f}"), textposition="outside",
               hovertemplate="%{x:.2f}<extra>" + comp + "</extra>")
layout(f5, "Força da tendência e da sazonalidade (0 = nenhuma, 1 = domina a série)", altura=400,
       barmode="group", bargap=0.3)
f5.update_xaxes(range=[0, 1.12], showgrid=True, gridcolor=TINTA["grade"])
f5.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
f5.update_layout(margin=dict(l=250, b=70), legend=dict(y=-0.1, yanchor="top"))

# =============================================================== 6. ciclos
f6 = make_subplots(rows=1, cols=2, column_widths=[0.58, 0.42], horizontal_spacing=0.1,
                   subplot_titles=("Crescimento real do total por ano (%)",
                                   "Periodograma do componente cíclico (filtro HP)"))
for flag, nome, cor in [(True, "Ano de eleição geral", CORES[1]), (False, "Demais anos", CORES[0])]:
    sel = cresc[eleicao == flag]
    f6.add_bar(x=sel.index, y=sel, name=nome, marker_color=cor, row=1, col=1,
               text=sel.map(lambda v: f"{v:+.0f}%"), textposition="outside",
               hovertemplate="%{x}: %{y:+.1f}%<extra>" + nome + "</extra>")
for v, nome, cor in [(med_el, "mediana eleição", CORES[1]), (med_out, "mediana demais", CORES[0])]:
    f6.add_hline(y=v, line=dict(color=cor, width=1.5, dash="dot"), row=1, col=1,
                 annotation_text=f"{nome}: {v:+.1f}%", annotation_position="top right",
                 annotation_font=dict(size=10, color=TINTA["secundaria"]))
f6.add_scatter(x=per.index, y=per / per.max(), mode="lines", line=dict(color=CORES[2], width=2), row=1, col=2,
               showlegend=False, hovertemplate="período %{x:.0f} meses<br>potência %{y:.2f}<extra></extra>")
for p, v in picos.items():
    f6.add_annotation(x=np.log10(p), y=v / per.max(), text=f"{p:.0f} meses", showarrow=True, arrowhead=0,
                      ax=0, ay=-22, font=dict(size=10, color=TINTA["primaria"]), row=1, col=2)
layout(f6, None, altura=440, bargap=0.2)
f6.update_xaxes(type="log", title_text="período (meses, escala log)", row=1, col=2,
                tickvals=[3, 6, 12, 24, 48, 96])
f6.update_yaxes(title_text="potência relativa", row=1, col=2)
f6.update_xaxes(dtick=1, tickangle=-45, row=1, col=1)
f6.update_layout(legend=dict(y=-0.2, yanchor="top"), margin=dict(t=50))

# =============================================================== 7. SICONFI – subfunções
f7 = go.Figure()
esferas = ["União", "Estados + DF", "Municípios"]
nomes_sub = list(dict.fromkeys(NOME_SUB.values())) + ["Demais subfunções"]
for e_i, esfera in enumerate(esferas):
    d = (sub[sub.esfera == esfera].pivot_table(index="ano", columns="sub", values="real_bi", aggfunc="sum")
         .reindex(columns=nomes_sub).fillna(0))  # NaN quebra o empilhamento no Plotly
    d.index = d.index.astype(int)
    for s_i, s in enumerate(nomes_sub):
        f7.add_bar(x=d.index, y=d[s], name=s, marker=dict(color=CORES[s_i], line=dict(color=TINTA["superficie"], width=1)),
                   visible=e_i == 0, showlegend=e_i == 0, legendgroup=s,
                   hovertemplate="%{y:,.1f} bi<extra>" + s + "</extra>")
botoes7 = []
for e_i, esfera in enumerate(esferas):
    vis = []
    for j in range(len(esferas)):
        vis += [j == e_i] * len(nomes_sub)
    botoes7.append(dict(label=esfera, method="update", args=[{"visible": vis, "showlegend": vis}]))
layout(f7, None, altura=460, barmode="stack", hovermode="x unified", bargap=0.25)
f7.update_layout(updatemenus=[dict(buttons=botoes7, active=0, x=0, y=1.16, xanchor="left",
                                   bgcolor="white", bordercolor=TINTA["eixo"])],
                 legend=dict(y=-0.12, yanchor="top"), margin=dict(t=70))
f7.update_xaxes(dtick=1)

# =============================================================== HTML
def fig_html(fig, primeira=False):
    return pio.to_html(fig, include_plotlyjs=primeira, full_html=False,
                       config={"displaylogo": False, "responsive": True, "locale": "pt-BR"})


def tiles():
    out = []
    for m in maiores:
        top = m["composicao"].head(2)
        comp = " · ".join(f"{html.escape(n)} {v / m['total'] * 100:.0f}%" for n, v in top.items())
        out.append(f"""<div class="tile"><div class="tile-gov">{html.escape(m['governo'])}
            <span>({m['anos']})</span></div><div class="tile-ano">{m['ano']}</div>
            <div class="tile-val">{bi(m['total'])}</div><div class="tile-comp">{comp}</div></div>""")
    return "\n".join(out)


def tabela_mandatos():
    linhas = []
    for m in maiores:
        anos_gov = [a for a in uniao08.index if governo_do_ano(a) == m["governo"]]
        if anos_gov:
            a08 = int(uniao08.loc[anos_gov].idxmax())
            f08 = f"{a08} — {bi(uniao08[a08])}"
        else:
            f08 = "sem dado (SICONFI União só desde 2013)"
        comp = "<br>".join(f"{html.escape(n)}: {bi(v)}" for n, v in m["composicao"].items())
        linhas.append(f"<tr><td>{html.escape(m['governo'])}</td><td>{m['anos']}</td><td><b>{m['ano']}</b></td>"
                      f"<td class=num>{bi(m['total'])}</td><td class=comp>{comp}</td><td>{f08}</td></tr>")
    return "\n".join(linhas)


def tabela_anual():
    cab = "".join(f"<th>{html.escape(c)}</th>" for c in ["Ano", "Governo", *NOMES, "Total"])
    corpo = []
    for ano, r in anual.iterrows():
        cel = "".join(f"<td class=num>{r[n]:,.1f}</td>" for n in [*NOMES, "Total"])
        corpo.append(f"<tr><td>{ano}{'*' if r.parcial else ''}</td><td>{html.escape(r.governo)}</td>{cel}</tr>")
    return f"<table><thead><tr>{cab}</tr></thead><tbody>{''.join(corpo)}</tbody></table>"


tend_forte = forcas["Tendência"].idxmax()
saz_forte = forcas["Sazonalidade"].idxmax()
pico_abono = perfil.loc["Abono salarial"].idxmax()
fmt_pct = lambda v: f"{v:.0f}%"  # noqa: E731

pagina = f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gasto social federal</title>
<style>
  :root {{ color-scheme: light; --sup:{TINTA['superficie']}; --pag:#f9f9f7; --t1:{TINTA['primaria']};
          --t2:{TINTA['secundaria']}; --t3:{TINTA['muted']}; --borda:rgba(11,11,11,0.10); }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--pag); color:var(--t1); font-family:{FONTE}; line-height:1.5; }}
  main {{ max-width:1180px; margin:0 auto; padding:28px 16px 60px; }}
  h1 {{ font-size:26px; margin:0 0 4px; }} h2 {{ font-size:18px; margin:0 0 6px; }}
  .sub {{ color:var(--t2); margin:0 0 20px; max-width:860px; }}
  .card {{ background:var(--sup); border:1px solid var(--borda); border-radius:10px; padding:18px 18px 8px; margin:18px 0; }}
  .card p {{ color:var(--t2); margin:4px 0 10px; max-width:900px; }}
  .tiles {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(165px, 1fr)); gap:12px; }}
  .tile {{ background:var(--sup); border:1px solid var(--borda); border-radius:10px; padding:14px; }}
  .tile-gov {{ font-weight:600; font-size:14px; }} .tile-gov span {{ color:var(--t3); font-weight:400; font-size:12px; }}
  .tile-ano {{ font-size:28px; font-weight:700; margin-top:4px; }}
  .tile-val {{ font-size:15px; color:var(--t1); }} .tile-comp {{ font-size:12px; color:var(--t2); margin-top:6px; }}
  .aviso {{ border-left:3px solid {CORES[3]}; padding:8px 12px; background:#fffaf0; color:var(--t2); border-radius:4px; font-size:14px; }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; margin:8px 0 12px; }}
  th, td {{ padding:6px 8px; border-bottom:1px solid #e1e0d9; text-align:left; vertical-align:top; }}
  th {{ color:var(--t2); font-weight:600; }} td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  td.comp {{ font-size:12px; color:var(--t2); }}
  .scroll {{ overflow-x:auto; }}
  details summary {{ cursor:pointer; color:var(--t2); margin:8px 0; }}
  ul {{ color:var(--t2); margin:6px 0 12px; padding-left:20px; }} li {{ margin:3px 0; }}
  footer {{ color:var(--t3); font-size:12px; margin-top:24px; }}
</style></head><body><main>
<h1>Gasto social federal por mandato, 1999–2026</h1>
<p class="sub">Bolsa Família/Auxílio Brasil, BPC/LOAS, abono salarial, seguro-desemprego e assistência social
discricionária: valores pagos pela União, em <b>R$ de {ANO_BASE}</b> (IPCA). Análise de tendência,
sazonalidade e ciclos em séries mensais do Tesouro Nacional.</p>

<div class="aviso"><b>Teste feito pela IA (Claude)</b> para validar fontes e métodos. Limitações: as cinco séries só
existem juntas a partir de 2009 (FHC e Lula 1 ficam de fora; o BPC vem desde 1999); a parte da pergunta sobre
<b>execução acima da dotação inicial</b> não foi respondida, porque nem o RTN nem o SICONFI (DCA) trazem a dotação
— ela está no RREO/SIOP.</div>

<div class="card"><h2>Ano de maior gasto em cada mandato</h2>
<p>Soma dos 5 programas, só anos completos. Abaixo de cada ano, os dois programas com maior peso.</p>
<div class="tiles">{tiles()}</div>
<details><summary>Ver tabela completa (com a composição e a função 08 do SICONFI)</summary>
<div class="scroll"><table><thead><tr><th>Mandato</th><th>Anos cobertos</th><th>Maior ano</th><th>Total</th>
<th>No que foi gasto</th><th>Maior ano na função 08 – União (SICONFI, empenhado, inclui créditos extraordinários)</th></tr></thead>
<tbody>{tabela_mandatos()}</tbody></table></div></details>
{fig_html(f2, primeira=True)}</div>

<div class="card"><h2>Séries temporais</h2>
<p>Soma móvel de 12 meses: elimina a sazonalidade e mostra o nível anualizado mês a mês. Clique na legenda para
mostrar/ocultar séries; “Créditos extraordinários” (Covid, Auxílio Emergencial) começa oculto porque em 2020
chega a {bi(cred_extra.loc['2020'].sum())} e achata o resto.</p>
{fig_html(f1)}</div>

<div class="card"><h2>Tendência: decomposição STL</h2>
<p>Decomposição sazonal STL multiplicativa (log, período de 12 meses, robusta a outliers como 2020). Escolha a série no
menu. A <b>tendência</b> é o nível de longo prazo; a <b>sazonalidade</b>, o padrão que se repete todo ano; o
<b>resíduo</b>, o que sobra (choques, mudanças de regra).</p>
{fig_html(f3)}</div>

<div class="card"><h2>Sazonalidade</h2>
<ul>
<li><b>{html.escape(saz_forte)}</b> é a série mais sazonal (força {forcas.loc[saz_forte, 'Sazonalidade']:.2f}): o
pico é em <b>{pico_abono}</b>, com {fmt_pct(perfil.loc['Abono salarial'].max())} do nível normal, por causa do calendário anual de pagamento.</li>
<li><b>{html.escape(tend_forte)}</b> é praticamente só tendência (força {forcas.loc[tend_forte, 'Tendência']:.2f}): cresce de forma
contínua com o envelhecimento da população e o salário mínimo.</li>
<li>Bolsa Família/Auxílio Brasil quase não tem sazonalidade (força {forcas.loc[NOMES[0], 'Sazonalidade']:.2f}); suas
variações vêm de mudanças de regra (2020, 2022, 2023), que aparecem no resíduo e na tendência.</li>
</ul>
{fig_html(f4)}
{fig_html(f5)}</div>

<div class="card"><h2>Ciclos</h2>
<ul>
<li><b>Ciclo eleitoral:</b> a mediana do crescimento real em anos de eleição geral foi
<b>{med_el:+.1f}%</b>, contra <b>{med_out:+.1f}%</b> nos demais anos. São só {eleicao.sum()} anos eleitorais
na amostra e 2022 (Auxílio Brasil de R$ 600) pesa muito: é um indício, não uma prova.</li>
<li><b>Periodograma:</b> os picos ficam em {', '.join(f'{p:.0f}' for p in picos.index)} meses. O de ~12 meses é a
sazonalidade que sobrou; os de longo período (4–6 anos) lembram o ciclo político/econômico, mas com
{len(total) // 12} anos de dados há poucas repetições para confirmar.</li>
</ul>
{fig_html(f6)}</div>

<div class="card"><h2>No que foi gasto: função 08 (Assistência Social) por subfunção</h2>
<p>SICONFI/Base dos Dados, despesa empenhada, 2013–2025, por esfera (escolha no menu). Na União, os créditos
extraordinários da Covid (Auxílio Emergencial) entram em 2020 e 2021. Em 2025 o Bolsa Família passou da
subfunção 244 (Assistência Comunitária) para a nova 246 (Segurança e Renda); as duas aparecem somadas.</p>
{fig_html(f7)}</div>

<div class="card"><h2>Tabela de dados (R$ bi de {ANO_BASE})</h2>
<details><summary>Mostrar valores anuais por programa</summary><div class="scroll">{tabela_anual()}</div>
<p>* 2026: janeiro a julho.</p></details></div>

<footer>Fontes: Tesouro Nacional – Resultado do Tesouro Nacional (série histórica mensal, tabela 1.2-A, valores pagos);
Base dos Dados – br_me_siconfi (DCA, função 08); BCB/SGS 433 (IPCA). Valores do RTN convertidos de R$ de jul/2026
para R$ médios de {ANO_BASE}. Governo de cada ano = quem governou a maior parte dele (2016 = Dilma 2).</footer>
</main></body></html>"""

saida = SAIDA / "dashboard_social.html"
saida.write_text(pagina, encoding="utf-8")
print("salvo:", saida)
for m in maiores:
    print(f"{m['governo']:>10} {m['anos']}  maior ano {m['ano']}  {m['total']:.1f} bi  ->",
          ", ".join(f"{n} {v:.1f}" for n, v in m["composicao"].head(3).items()))
print(forcas.round(2).to_string())
print(f"mediana eleição {med_el:.1f}  demais {med_out:.1f}; picos {list(picos.index.round(1))}")
