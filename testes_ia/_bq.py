"""Helper de acesso ao BigQuery (Base dos Dados) para os testes."""
import os
from pathlib import Path

import basedosdados as bd
from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parents[1]
load_dotenv(RAIZ / ".env")
PROJECT_ID = os.environ["GCP_PROJECT_ID"]
DADOS = Path(__file__).resolve().parent / "dados"
DADOS.mkdir(exist_ok=True)


def q(sql: str):
    return bd.read_sql(sql, billing_project_id=PROJECT_ID)
