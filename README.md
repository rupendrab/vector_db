# Vector DB · Local Semantic Search

Search financial documents by meaning using PostgreSQL, pgvector, Ollama, and a Streamlit UI. Generate embeddings locally, store them alongside document text, and retrieve similar documents with cosine distance.

This repository is a learning project with reusable Python helpers and a Jupyter notebook exploring different embedding workflows. The UI retrieves documents; it does not generate answers or provide financial advice.

## Features

- Local embeddings through Ollama, with configurable request batches.
- Progress logs after increases of at least 10 percentage points, plus final completion.
- Transactional inserts from Polars DataFrames into PostgreSQL.
- Semantic search returning document IDs, text, distance, and similarity.
- Streamlit search with 1–20 results, 50-word previews, and a **More — full document** option.
- Notebook experiments with Sentence Transformers and Ollama.

## How it works

```text
Documents → Ollama embeddings → PostgreSQL + pgvector
                                      ↑
Search question → Ollama embedding → cosine-distance search → Streamlit results
```

Documents are embedded during ingestion. Each search generates only the question's embedding and searches the stored vectors.

## Requirements

- Python 3.12 or later.
- [uv](https://docs.astral.sh/uv/) for dependency management.
- Docker with its daemon running, or an existing PostgreSQL instance with pgvector.
- [Ollama](https://ollama.com/) running locally.

Allow disk space for Python dependencies, model downloads, and database storage. Initial installation and model downloads require internet access. Local inference can use substantial CPU/GPU resources.

## Quick start

Run commands from the repository root.

### 1. Install dependencies

```bash
git clone https://github.com/rupendrab/vector_db.git
cd vector_db
uv sync
```

The dependency set includes Jupyter, PyTorch, and Sentence Transformers for notebook experiments as well as the Streamlit application.

### 2. Start PostgreSQL

```bash
bash pull_pgvector.sh
bash run_pg_database.sh
```

The scripts create a PostgreSQL 16 container named `postgres-vector`, expose port `5432`, and store database files in the Docker volume `postgres_vector_data`.

The current connection settings in `src/vector_db/create_db.py` match the container script:

| Setting | Local default |
| --- | --- |
| Host | `localhost` |
| Port | `5432` |
| Database | `vectordb` |
| Username | `postgres` |
| Password | `postgres123` |

These are development defaults. Change both the script and connection URL for your environment before deployment. The current code does not load a database URL from `.env`.

For an existing container, use:

```bash
bash manage_pg_database.sh start
# Also supported: stop, restart, logs
```

Check the connection:

```bash
PYTHONPATH=src uv run python -m vector_db.check_connection
```

### 3. Prepare Ollama

Start the Ollama application or run `ollama serve` in a separate terminal if it is not already running. Then download the model:

```bash
ollama pull nomic-embed-text
```

The search function defaults to `nomic-embed-text`; the UI uses that default. Use the same embedding model for document ingestion and queries.

### 4. Create the search table and ingest sample documents

Save the following as `seed_demo.py` in the repository root and run `uv run python seed_demo.py`. You can also execute it in a notebook started from the root.

```python
import logging
import sys

import polars as pl
from sqlalchemy import text

from create_embeddings import create_and_save_embeddings
from src.vector_db.create_db import get_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
)
logging.getLogger("httpx").setLevel(logging.WARNING)

engine = get_engine()

with engine.begin() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS fiqa_documents (
            document_id INTEGER PRIMARY KEY,
            content TEXT NOT NULL,
            embedding VECTOR(768)
        )
    """))

documents = pl.DataFrame({
    "document_id": [1, 2, 3],
    "content": [
        "An emergency fund provides accessible savings for unexpected expenses.",
        "Diversification spreads investments across assets to reduce concentration risk.",
        "Credit card interest increases the cost of carrying an unpaid balance.",
    ],
})

embedded = create_and_save_embeddings(
    df=documents,
    column="content",
    model="nomic-embed-text",
    engine=engine,
    table_name="fiqa_documents",
    batch_size=32,
)

print(f"Saved {embedded.height} documents")
```

The sample is for an empty table. Inserts append rows and do not perform an upsert: rerunning with the same document IDs raises a primary-key error. If you already loaded FIQA data, skip this sample ingestion.

**Table distinction:** `create_db()` creates the separate `documents` table with `VECTOR(384)` for earlier experiments. It does not create `fiqa_documents`. The UI requires the `fiqa_documents` schema above, with 768-dimensional embeddings for the default model.

### 5. Launch the UI

```bash
uv run streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`. Ask a question, choose the number of results, and select **Search**. Documents longer than 50 words have a **More — full document** expander.

Try:

- “How can I prepare for unexpected expenses?”
- “Why should I diversify my investments?”
- “What is the cost of keeping a credit card balance?”

## Use your own data or FIQA

The ingestion function accepts a Polars DataFrame whose columns match the destination table. For `fiqa_documents`, provide:

| Column | Type | Purpose |
| --- | --- | --- |
| `document_id` | Integer | Unique document identifier |
| `content` | String | Text to embed and display |

`create_embeddings()` adds an `embedding` column. `save_embeddings_to_db()` expects that column to already exist, while `create_and_save_embeddings()` performs both steps. The destination table must exist before saving.

`Test_01.ipynb` includes downloading and preparing a 10,000-document subset of the [BeIR FIQA dataset](https://huggingface.co/datasets/BeIR/fiqa). Its generated `fiqa_10000.parquet` file is excluded from Git. Once prepared, replace the sample DataFrame with:

```python
documents = (
    pl.read_parquet("fiqa_10000.parquet")
    .with_columns(pl.col("document_id").cast(pl.Int64))
    .select("document_id", "content")
)
```

Use a fresh table or nonconflicting IDs. Refer to the dataset's own terms before redistributing its contents.

## Python API

From a script or notebook in the repository root:

```python
from src.vector_db.create_db import get_engine
from src.vector_db.db import query_to_df, semantic_search_fiqa

engine = get_engine()

results = semantic_search_fiqa(
    engine,
    query="Should I pay off credit card debt before investing?",
    limit=5,
)
print(results)

count = query_to_df(engine, "SELECT COUNT(*) AS documents FROM fiqa_documents")
print(count)
```

`get_engine()` caches the engine within the Python process. Database connections are opened as needed and returned to its pool.

Search results contain `document_id`, `content`, `distance`, and `similarity`. Lower cosine distance means a closer match; similarity is `1 - distance`, not a probability or confidence percentage.

## Optional: add a vector index

For larger datasets, create an HNSW cosine index once after ingestion:

```python
with engine.begin() as conn:
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS fiqa_documents_embedding_hnsw_idx
        ON fiqa_documents USING hnsw (embedding vector_cosine_ops)
    """))
    conn.execute(text("ANALYZE fiqa_documents"))
```

This example uses `engine` and `text` from the setup code. HNSW provides approximate nearest-neighbor search, trading some recall for speed. Index creation consumes CPU and memory but does not regenerate embeddings. PostgreSQL chooses whether to use the index; small tables may still use a sequential scan.

## Notebooks

```bash
uv run jupyter lab
```

Open `Test_01.ipynb` to explore embedding generation, SQL queries, and similarity search. Some cells use different models and table schemas; run the relevant setup cells before each experiment.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| PostgreSQL connection refused | Start the container and confirm port `5432` is available. |
| Ollama connection or model error | Start Ollama and run `ollama pull nomic-embed-text`. |
| `fiqa_documents` does not exist | Run the search-table setup in the quick start. |
| Vector dimension mismatch | Match the column dimensions to the model output, and use the same model for ingestion and search. |
| Duplicate document IDs | Ingestion appends; it does not skip or replace existing rows. |
| No results | Check that the search table contains rows. |
| Notebook logs appear in red | Configure logging with `stream=sys.stdout`; stderr styling does not imply failure. |
| Laptop gets hot during ingestion | Start with fewer documents, reuse saved embeddings, and reduce sustained load with pauses between batches if needed. |

Batching limits each Ollama request, but the current ingestion implementation retains all generated embeddings in memory before inserting them. Large datasets can therefore still require substantial RAM.

## Project layout

```text
app.py                         Streamlit search interface
create_embeddings.py           Embedding generation and database inserts
src/vector_db/
    create_db.py               Cached engine and 384-dimension example table
    check_connection.py        PostgreSQL connectivity check
    db.py                      SQL-to-Polars and FIQA semantic search
Test_01.ipynb                  Notebook experiments and FIQA preparation
run_pg_database.sh             Create the PostgreSQL container
manage_pg_database.sh          Manage the existing container
pull_pgvector.sh               Pull the PostgreSQL/pgvector image
psql.sh                        Open psql inside the container
pyproject.toml                 Python dependencies
uv.lock                        Dependency lockfile
```

Local environments, model files, generated Parquet datasets, notebook checkpoints, and secret configuration files are excluded by `.gitignore`.

## License

This project is licensed under the [Apache License 2.0](LICENSE). Third-party models and datasets retain their own licenses and terms.
