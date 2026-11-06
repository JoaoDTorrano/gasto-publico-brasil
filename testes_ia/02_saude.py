"""Teste – Pergunta 1: como evoluiu o gasto em saúde (função 10), em reais de 2025
e per capita, por esfera de governo, e a variação média anual em cada mandato?

Saída: saida/saude_series_temporais.png e saida/saude_crescimento_mandato.csv
"""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from _comum import ANO_BASE, CORES, DADOS, GOVERNOS, SAIDA, TINTA, fator_anual, governo_do_ano, rtn_mensal

ESFERAS = {"Municípios": CORES[0], "Estados + DF": CORES[1], "União": CORES[2]}

# ------------------------------------------------------------------ dados
fator = fator_anual()
pop = pd.read_parquet(DADOS / "populacao_brasil.parquet").set_index("ano")["populacao"]

saude = pd.read_parquet(DADOS / "siconfi_saude.parquet")
saude["real_bi"] = saude.valor * saude.ano.map(fator) / 1e9
saude["per_capita"] = saude.real_bi * 1e9 / saude.ano.map(pop)
saude["classif_antiga"] = saude.conta.eq("Saúde e Saneamento")
saude["governo"] = saude.ano.map(governo_do_ano)

# União mensal pelo RTN (pago): obrigatórias com controle de fluxo + discricionárias
rtn = rtn_mensal({"4.4.1.3": "obrig", "4.4.2.1": "discr"})
uniao_mensal = (rtn.obrig + rtn.discr).loc["2008":] / 1e3  # R$ bi de 2025
uniao_12m = uniao_mensal.rolling(12).sum()

# Crescimento real médio anual por mandato (média das variações ano a ano).
# Municípios só a partir de 2003: até 2001 a função era "Saúde e Saneamento"
# e 2002 é o ano da quebra de série.
cresc = saude.sort_values("ano").copy()
cresc["var_pct"] = cresc.groupby("esfera").real_bi.pct_change() * 100
cresc = cresc[~((cresc.esfera == "Municípios") & (cresc.ano <= 2002))].dropna(subset=["var_pct"])
tabela = (
    cresc.groupby(["governo", "esfera"])
    .agg(var_media_pct=("var_pct", "mean"), anos=("ano", lambda a: f"{a.min()}–{a.max()}"))
    .reset_index()
)
ordem_gov = [g for g, *_ in GOVERNOS]
tabela["ordem"] = tabela.governo.map(ordem_gov.index)
tabela = tabela.sort_values(["ordem", "esfera"]).drop(columns="ordem")
tabela.round(1).to_csv(SAIDA / "saude_crescimento_mandato.csv", index=False, encoding="utf-8-sig")
print(tabela.round(1).to_string(index=False))

# ------------------------------------------------------------------ estilo
plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
    "axes.edgecolor": TINTA["eixo"], "axes.labelcolor": TINTA["secundaria"],
    "xtick.color": TINTA["muted"], "ytick.color": TINTA["muted"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": TINTA["grade"], "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": TINTA["superficie"], "axes.facecolor": TINTA["superficie"],
})

fig = plt.figure(figsize=(13, 15.5))
gs = fig.add_gridspec(4, 1, height_ratios=[1.15, 1, 1, 0.95], hspace=0.42, top=0.925, bottom=0.06)
ax_a, ax_b, ax_c, ax_d = (fig.add_subplot(gs[i]) for i in range(4))


def faixas_governo(ax, rotular=False, anual=True):
    """Faixas alternadas por mandato (eixo em anos ou em datas)."""
    for k, (nome, ini, fim) in enumerate(GOVERNOS):
        x0 = ini.year + (ini.timetuple().tm_yday - 1) / 365 - 0.5 if anual else mdates.date2num(ini)
        x1 = fim.year + fim.timetuple().tm_yday / 365 - 0.5 if anual else mdates.date2num(fim)
        if k % 2 == 0:
            ax.axvspan(x0, x1, color=TINTA["faixa"], zorder=0, lw=0)
        if rotular:
            ax.text((x0 + x1) / 2, 1.0, nome, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8.5, color=TINTA["secundaria"])


def linhas_esferas(ax, coluna, fmt):
    for esfera, cor in ESFERAS.items():
        d = saude[saude.esfera == esfera].sort_values("ano")
        antiga, nova = d[d.classif_antiga], d[~d.classif_antiga]
        if len(antiga):  # classificação antiga em traço fino e esmaecido
            ponte = pd.concat([antiga, nova.head(1)])
            ax.plot(ponte.ano, ponte[coluna], color=cor, lw=1.5, alpha=0.45, ls=(0, (2, 2)))
        ax.plot(nova.ano, nova[coluna], color=cor, lw=2, marker="o", ms=3.5)
        ult = nova.iloc[-1]
        ax.annotate(f"{esfera}  {fmt(ult[coluna])}", (ult.ano, ult[coluna]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color=TINTA["primaria"])
    ax.set_xlim(1994.5, 2029.5)
    ax.set_xticks(range(1995, 2027, 2))


# A – valores reais
faixas_governo(ax_a, rotular=True)
linhas_esferas(ax_a, "real_bi", lambda v: f"R$ {v:,.0f} bi".replace(",", "."))
ax_a.set_title(f"A. Gasto anual em saúde por esfera — R$ bilhões de {ANO_BASE} (despesa empenhada)",
               loc="left", fontsize=11.5, color=TINTA["primaria"], pad=22)
ax_a.annotate("1996–2001: “Saúde e Saneamento”\n(classificação antiga, inclui saneamento)",
              (1998, saude.query("esfera=='Municípios' and ano==1998").real_bi.iloc[0]),
              xytext=(1996, 175), fontsize=8, color=TINTA["secundaria"],
              arrowprops=dict(arrowstyle="-", color=TINTA["muted"], lw=0.7))

# B – per capita
faixas_governo(ax_b)
linhas_esferas(ax_b, "per_capita", lambda v: f"R$ {v:,.0f}".replace(",", "."))
ax_b.set_title(f"B. Gasto em saúde por habitante — R$ de {ANO_BASE}", loc="left",
               fontsize=11.5, color=TINTA["primaria"])

# C – União mensal (RTN)
faixas_governo(ax_c, anual=False)
ax_c.plot(uniao_mensal.index, uniao_mensal * 12, color=ESFERAS["União"], lw=0.8, alpha=0.35,
          label="Mês × 12 (anualizado)")
ax_c.plot(uniao_12m.index, uniao_12m, color=ESFERAS["União"], lw=2, label="Soma móvel de 12 meses")
pico = uniao_12m.idxmax()
ax_c.annotate(f"Pico: R$ {uniao_12m.max():.0f} bi\n(12 meses até {pico:%m/%Y})", (pico, uniao_12m.max()),
              xytext=(-120, 8), textcoords="offset points", fontsize=8.5, color=TINTA["primaria"],
              arrowprops=dict(arrowstyle="-", color=TINTA["muted"], lw=0.7))
ax_c.set_xlim(pd.Timestamp("2008-01-01"), pd.Timestamp("2026-12-31"))
ax_c.xaxis.set_major_locator(mdates.YearLocator(2))
ax_c.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax_c.set_ylim(0, max(250, (uniao_mensal * 12).quantile(0.99) * 1.1))
ax_c.legend(loc="upper left", frameon=False, fontsize=9)
ax_c.set_title(f"C. União — gasto mensal em saúde (Tesouro/RTN, valor pago), R$ bilhões de {ANO_BASE}",
               loc="left", fontsize=11.5, color=TINTA["primaria"])
for nome, ini, fim in GOVERNOS:
    if fim.year >= 2008:
        x = mdates.date2num(max(ini, pd.Timestamp("2008-01-01").date()))
        ax_c.text(x, 0.02, f" {nome}", transform=ax_c.get_xaxis_transform(), fontsize=8,
                  color=TINTA["secundaria"], ha="left")

# D – crescimento médio por mandato
govs = [g for g in ordem_gov if g in set(tabela.governo)]
larg = 0.26
for j, (esfera, cor) in enumerate(ESFERAS.items()):
    t = tabela[tabela.esfera == esfera].set_index("governo").reindex(govs)
    xs = [i + (j - 1) * larg for i in range(len(govs))]
    barras = ax_d.bar(xs, t.var_media_pct, width=larg - 0.03, color=cor, label=esfera)
    for x, v in zip(xs, t.var_media_pct):
        if pd.notna(v):
            ax_d.text(x, v + (0.4 if v >= 0 else -0.4), f"{v:.1f}%", ha="center",
                      va="bottom" if v >= 0 else "top", fontsize=7.5, color=TINTA["secundaria"])
ax_d.axhline(0, color=TINTA["eixo"], lw=0.8)
ax_d.set_xticks(range(len(govs)), govs)
ax_d.grid(axis="x", visible=False)
ax_d.set_ylim(tabela.var_media_pct.min() - 2.5, tabela.var_media_pct.max() + 3)
ax_d.legend(loc="upper center", frameon=False, ncols=3, fontsize=9)
ax_d.set_ylabel("% ao ano (real)")
ax_d.set_title("D. Crescimento real médio por ano em cada mandato (média das variações anuais)",
               loc="left", fontsize=11.5, color=TINTA["primaria"])

fig.suptitle("Gasto público em saúde no Brasil, 1996–2026", x=0.065, y=0.992, ha="left", va="top",
             fontsize=16, fontweight="bold", color=TINTA["primaria"])
fig.text(0.065, 0.966,
         "Função 10 (Saúde). Esferas não devem ser somadas: repasses do SUS da União aparecem também "
         "no gasto de estados e municípios.", fontsize=9.5, color=TINTA["secundaria"])
fig.text(0.065, 0.025,
         "Fontes: Base dos Dados – br_me_siconfi (FINBRA até 2012, SICONFI desde 2013); Tesouro Nacional – RTN; "
         "IBGE (população); BCB/SGS 433 (IPCA).\nEstados e União só disponíveis no SICONFI a partir de 2013; "
         "nº de municípios declarantes varia por ano (4.270 a 5.552). 2026: RTN até jul/2026.",
         fontsize=8, color=TINTA["muted"])

saida = SAIDA / "saude_series_temporais.png"
fig.savefig(saida, dpi=150, bbox_inches="tight")
print("salvo:", saida)
