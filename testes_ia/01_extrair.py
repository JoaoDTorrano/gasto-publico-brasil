"""Teste: baixa os dados usados nos gráficos (SICONFI, população, IPCA).

Saída em testes_ia/dados/*.parquet
"""
import time

import pandas as pd
import requests

from _bq import DADOS, q

DS = "basedosdados.br_me_siconfi"

# ---------------------------------------------------------------- Saúde (função 10)
# Municípios: 1996–2001 só existe "Saúde e Saneamento" (classificação antiga);
# 2002–2003 conta "Saúde" sem código; 2004+ id_conta_bd = 3.10.000.
# Antes de 2013 (FINBRA) o estágio é a despesa empenhada.
saude_mun = q(f"""
SELECT ano, 'Municípios' AS esfera,
       ANY_VALUE(conta_bd) AS conta,
       COUNT(DISTINCT id_municipio) AS n_entes,
       SUM(valor) AS valor
FROM `{DS}.municipio_despesas_funcao`
WHERE (id_conta_bd = '3.10.000'
       OR (ano < 2004 AND conta_bd IN ('Saúde', 'Saúde e Saneamento')))
  AND (estagio_bd = 'Despesas Empenhadas' OR ano < 2013)
GROUP BY ano
""")

saude_uf_br = q(f"""
SELECT ano, 'Estados + DF' AS esfera, ANY_VALUE(conta_bd) AS conta,
       COUNT(DISTINCT sigla_uf) AS n_entes, SUM(valor) AS valor
FROM `{DS}.uf_despesas_funcao`
WHERE id_conta_bd = '3.10.000' AND estagio_bd = 'Despesas Empenhadas'
GROUP BY ano
UNION ALL
SELECT ano, 'União' AS esfera, ANY_VALUE(conta_bd) AS conta,
       1 AS n_entes, SUM(valor) AS valor
FROM `{DS}.brasil_despesas_funcao`
WHERE id_conta_bd = '3.10.000' AND estagio_bd = 'Despesas Empenhadas'
GROUP BY ano
""")

saude = pd.concat([saude_mun, saude_uf_br]).sort_values(["esfera", "ano"])
saude.to_parquet(DADOS / "siconfi_saude.parquet", index=False)
print(saude.assign(bi=lambda d: (d.valor / 1e9).round(1)).drop(columns="valor").to_string(index=False))

# ---------------------------------------------- Assistência Social (função 08) por subfunção
assist = q(f"""
SELECT ano, 'Municípios' AS esfera, id_conta_bd, conta_bd, SUM(valor) AS valor
FROM `{DS}.municipio_despesas_funcao`
WHERE STARTS_WITH(id_conta_bd, '3.08.') AND estagio_bd = 'Despesas Empenhadas'
GROUP BY 1, 2, 3, 4
UNION ALL
SELECT ano, 'Estados + DF', id_conta_bd, conta_bd, SUM(valor)
FROM `{DS}.uf_despesas_funcao`
WHERE STARTS_WITH(id_conta_bd, '3.08.') AND estagio_bd = 'Despesas Empenhadas'
GROUP BY 1, 2, 3, 4
UNION ALL
SELECT ano, 'União', id_conta_bd, conta_bd, SUM(valor)
FROM `{DS}.brasil_despesas_funcao`
WHERE STARTS_WITH(id_conta_bd, '3.08.') AND estagio_bd = 'Despesas Empenhadas'
GROUP BY 1, 2, 3, 4
""")
assist.to_parquet(DADOS / "siconfi_assistencia.parquet", index=False)
print(
    assist[assist.id_conta_bd == "3.08.000"]
    .pivot_table(index="ano", columns="esfera", values="valor", aggfunc="sum")
    .div(1e9).round(1).to_string()
)

# ---------------------------------------------------------------- População
pop = q("SELECT ano, populacao FROM `basedosdados.br_ibge_populacao.brasil` ORDER BY ano")
pop.to_parquet(DADOS / "populacao_brasil.parquet", index=False)
print("população:", pop.ano.min(), "–", pop.ano.max())

# ---------------------------------------------------------------- IPCA mensal (BCB SGS 433)
def sgs(ini: int, fim: int, tentativas: int = 5) -> pd.DataFrame:
    """A API do BCB às vezes devolve HTML em vez de JSON; tenta de novo."""
    for t in range(tentativas):
        r = requests.get(
            "https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados",
            params={"formato": "json", "dataInicial": f"01/01/{ini}", "dataFinal": f"31/12/{fim}"},
            timeout=60,
        )
        if r.ok and "json" in r.headers.get("content-type", ""):
            return pd.DataFrame(r.json())
        time.sleep(2 * (t + 1))
    raise RuntimeError(f"API do BCB falhou para {ini}-{fim}")


ipca = pd.concat(sgs(ini, min(ini + 9, 2026)) for ini in range(1995, 2027, 10))
ipca["data"] = pd.to_datetime(ipca["data"], dayfirst=True)
ipca["variacao_pct"] = pd.to_numeric(ipca["valor"])
ipca[["data", "variacao_pct"]].drop_duplicates("data").to_parquet(DADOS / "ipca_mensal.parquet", index=False)
print("IPCA:", ipca.data.min().date(), "–", ipca.data.max().date())
