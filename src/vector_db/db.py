"""Read database query results as Polars DataFrames."""

import polars as pl
from sqlalchemy import Engine, text
import ollama


def query_to_df(engine: Engine, sql: str) -> pl.DataFrame:
    """Execute a SQL query and return its results as a Polars DataFrame.

    Args:
        engine: SQLAlchemy engine connected to the database to query.
        sql: SQL query that returns rows.

    Returns:
        A DataFrame containing the query's result columns and rows.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: If connecting or executing SQL fails.
        polars.exceptions.PolarsError: If converting the results fails.
    """
    with engine.connect() as conn:
        return pl.read_database(query=text(sql), connection=conn)


def semantic_search_fiqa(
    engine: Engine,
    query: str,
    model: str = "nomic-embed-text",
    limit: int = 5,
) -> pl.DataFrame:
    embedding = ollama.embed(
        model=model,
        input=query
    )["embeddings"][0]

    sql = text("""
        SELECT
            document_id,
            content,
            embedding <=> CAST(:embedding AS vector) AS distance,
            1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
        FROM fiqa_documents
        ORDER BY distance
        LIMIT :limit
    """)

    with engine.connect() as conn:
        rows = conn.execute(
            sql,
            {
                "embedding": str(embedding),
                "limit": limit
            }
        ).mappings().all()

    return pl.DataFrame(
        rows,
        schema={
            "document_id": pl.Int64,
            "content": pl.String,
            "distance": pl.Float64,
            "similarity": pl.Float64
        }
    )