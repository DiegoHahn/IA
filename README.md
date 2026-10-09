# IA: detecção de anomalias em telemetria de motores

> **English summary.** Unsupervised anomaly detection for electric motor telemetry
> (setpoint frequency, current, input voltage, status) using scikit-learn's Isolation
> Forest. `treinamentoModelo.py` trains the model from a CSV export;
> `previsao.py` scores rows from a PostgreSQL table (`log_motor`) and writes a 0/1
> `anomalia` flag back, or works CSV-to-CSV without a database. Code and docs are in
> Portuguese. Trained models are pickle files: only load ones you trust.

## O problema

Três motores enviam leituras periódicas para uma tabela `log_motor` no PostgreSQL.
O objetivo é marcar automaticamente as leituras fora do comportamento habitual,
como queda de tensão ou corrente inesperada, sem ter um conjunto de dados rotulado.

Por isso a abordagem é não supervisionada: um **Isolation Forest** aprende o formato
dos dados históricos e marca como anomalia os pontos mais fáceis de isolar.

## Esquema dos dados

Os dois scripts esperam as colunas abaixo, com estes nomes exatos (incluindo
espaços e acento):

| Coluna | Tipo | Uso |
|---|---|---|
| `ID Log` | inteiro | Identificador da leitura. Não entra no modelo. É a chave usada no `UPDATE`. |
| `ID Motor` | inteiro | Motor que gerou a leitura (1, 2 ou 3 no conjunto de treino). Entra no modelo. |
| `Status` | 0 ou 1 | Motor parado ou ligado. Entra no modelo. |
| `SP Frequencia` | float | Frequência de setpoint. Entra no modelo. |
| `Feed Corrente` | float | Corrente medida. Entra no modelo. |
| `Feed Tensão Entrada` | float | Tensão de entrada medida. Entra no modelo. |
| `Timestamp` | data e hora | Momento da leitura. Não entra no modelo. |
| `anomalia` | 0, 1 ou nulo | Só no banco. Nulo significa "ainda não classificado". O `previsao.py` preenche com 1 (anomalia) ou 0 (normal). |

A lista de colunas do modelo fica em `FEATURE_COLUMNS`, em `treinamentoModelo.py`,
e a previsão importa a mesma lista. Assim treino e previsão sempre usam a mesma ordem.

## Como funciona

**Treino (`treinamentoModelo.py`)**

1. Lê o CSV de treino.
2. Seleciona as colunas de `FEATURE_COLUMNS` e normaliza com `StandardScaler`.
3. Treina um `IsolationForest` com `contamination=0.1`, `n_estimators=200`,
   `max_samples=256` e `random_state=42`.
4. Salva o modelo e o scaler com `joblib`.

**Previsão (`previsao.py`)**

1. Lê do banco as linhas com `anomalia IS NULL`, ou lê de um CSV com `--input-csv`.
2. Aplica o scaler e o modelo salvos no treino.
3. Converte a saída do Isolation Forest, em que -1 é anomalia e 1 é normal, para 1 e 0.
4. Grava a coluna `anomalia` no banco em uma única transação e, se pedido,
   salva também um CSV com `--output-csv`.

## Como rodar

Requisitos: Python 3.10 ou mais novo. Testado com Python 3.14.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # e preencha DATABASE_URL se for usar o banco
```

### 1. Obter os dados de treino

O CSV de treino (`arquivoMotorIA.csv`, cerca de 46,7 mil leituras de 06/06/2024
a 13/06/2024) não está mais na `main`, mas continua no histórico do Git.
Para recuperá-lo (funciona igual no Bash e no PowerShell):

```bash
git restore --source=7bc7c3f^ -- arquivoMotorIA.csv
mkdir data
mv arquivoMotorIA.csv data/
```

A pasta `data/` está no `.gitignore`, então o arquivo não volta a ser commitado por engano.

Qualquer outro CSV com as colunas da tabela acima também serve, por exemplo um
export da própria tabela `log_motor`.

### 2. Treinar

```bash
python treinamentoModelo.py
# ou, com opções:
python treinamentoModelo.py --data data/arquivoMotorIA.csv --contamination 0.05
```

O modelo e o scaler vão para `models/`. Rode `python treinamentoModelo.py --help`
para ver todas as opções.

### 3. Classificar

Sem banco, de CSV para CSV:

```bash
python previsao.py --input-csv data/arquivoMotorIA.csv --output-csv previsoes.csv --no-db-write
```

Com banco, lendo e gravando em `log_motor`:

```bash
python previsao.py   # usa DATABASE_URL do ambiente ou do .env
```

## Configuração

Cada opção pode vir de argumento de linha de comando ou de variável de ambiente.
O argumento tem prioridade. Um arquivo `.env` na pasta onde o script roda é lido
automaticamente.

| Variável | Argumento | Padrão | Usada em |
|---|---|---|---|
| `DATABASE_URL` | `--database-url` | nenhum | `previsao.py`, obrigatória quando lê ou grava no banco |
| `IA_DATA_PATH` | `--data` | `data/arquivoMotorIA.csv` | `treinamentoModelo.py` |
| `IA_MODEL_PATH` | `--model` | `models/isolation_forest_model.pkl` | os dois scripts |
| `IA_SCALER_PATH` | `--scaler` | `models/scaler.pkl` | os dois scripts |

Opções só da linha de comando: `--contamination`, `--n-estimators` e `--max-samples`
no treino; `--input-csv`, `--output-csv` e `--no-db-write` na previsão.

## Resultados e limitações

- **Não há rótulos**, então não existe métrica de acerto como precisão ou recall.
  O que dá para afirmar é o comportamento do modelo nos dados de treino.
- Com `contamination=0.1`, o modelo marca por construção cerca de 10% do próprio
  conjunto de treino como anomalia. No CSV de treino, foram 4.671 de 46.711 leituras.
  Esse número reflete o parâmetro escolhido, não uma taxa real de falhas.
- Nesse mesmo treino, a proporção marcada varia por motor: cerca de 9,7% no motor 1,
  5,6% no motor 2 e 14,7% no motor 3.
- O arquivo `previsao_anomalias.csv` que existia no repositório tinha 5 leituras,
  todas marcadas como anomalia. É uma amostra pequena demais para concluir algo.
- `ID Motor` é um identificador, mas entra no modelo como número. Um modelo por
  motor, ou codificar o motor como categoria, provavelmente seria mais adequado.
- O treino considera todo o histórico como "majoritariamente normal". Se o CSV
  tiver muitas falhas, elas passam a fazer parte do que o modelo considera normal.

## Segurança: arquivos pickle

O modelo e o scaler são salvos com `joblib`, que usa `pickle`. **Carregar um arquivo
pickle executa código**, então só carregue modelos que você mesmo treinou ou que
vieram de uma fonte confiável. Por isso os `.pkl` não ficam mais no repositório:
cada um treina o próprio modelo seguindo os passos acima.

## Estrutura

```
treinamentoModelo.py   treino do Isolation Forest a partir de CSV
previsao.py            classificação via PostgreSQL ou CSV
requirements.txt       dependências
.env.example           modelo de configuração
```
