import logging

import ollama
import polars as pl
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Engine, MetaData, Table

logger = logging.getLogger(__name__)

def create_embeddings(
    df: pl.DataFrame,
    column: str,
    model: str,
    batch_size: int = 32
) -> pl.DataFrame:
    """Generate Ollama embeddings in batches while preserving row order.

    Log completion at INFO level after increases of at least 10 percentage
    points, and always when the final batch completes.

    Args:
        df: DataFrame containing the text to embed.
        column: Name of the column containing the input text.
        model: Name of the Ollama embedding model to use.
        batch_size: Maximum number of texts per request. Defaults to 32.

    Returns:
        A DataFrame with an added or replaced ``embedding`` column containing
        a list of floats for each input row.

    Raises:
        ValueError: If batch_size is less than or equal to zero.
        polars.exceptions.ColumnNotFoundError: If column is not in df.
        ollama.ResponseError: If the Ollama API returns an error response.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    texts = df[column]
    total = len(texts)
    embeddings = []
    last_logged_completed = 0
    if total == 0:
        logger.info("No documents to embed")
    for offset in range(0, total, batch_size):
        response = ollama.embed(
            model=model,
            input=texts.slice(offset, batch_size).to_list(),
        )
        embeddings.extend(response["embeddings"])
        completed = min(offset + batch_size, total)
        if (completed - last_logged_completed) * 10 >= total or completed == total:
            logger.info(
                "Embeddings completed: %.1f%% (%d/%d documents)",
                completed / total * 100,
                completed,
                total,
            )
            last_logged_completed = completed

    return df.with_columns(
        pl.Series("embedding", embeddings, dtype=pl.List(pl.Float64))
    )


def save_embeddings_to_db(
    df: pl.DataFrame, 
    engine: Engine,
    table_name: str
) -> None:
    """Append a DataFrame containing embeddings to an existing database table.

    DataFrame column names must match the destination table. For the documents
    table, supply ``content`` and ``embedding``; omit ``id`` to generate it.
    All inserts run in one transaction and roll back if any insert fails.

    Args:
        df: DataFrame containing an ``embedding`` column of numeric lists and
            any other columns to insert.
        engine: SQLAlchemy engine connected to the destination database.
        table_name: Name of an existing table in the default database schema.

    Raises:
        ValueError: If the embedding column is missing or a DataFrame column
            does not exist in the destination table.
        sqlalchemy.exc.SQLAlchemyError: If table lookup or insertion fails.
    """
    if "embedding" not in df.columns:
        raise ValueError("DataFrame must contain an 'embedding' column")
    if df.is_empty():
        return

    with engine.begin() as conn:
        table = Table(
            table_name,
            MetaData(),
            Column("embedding", Vector()),
            autoload_with=conn,
        )
        unknown_columns = set(df.columns) - set(table.columns.keys())
        if unknown_columns:
            raise ValueError(
                f"Columns not found in table: {', '.join(sorted(unknown_columns))}"
            )

        for batch in df.iter_slices(n_rows=1000):
            conn.execute(table.insert(), batch.to_dicts())


def create_and_save_embeddings(
    df: pl.DataFrame,
    column: str,
    model: str,
    engine: Engine,
    table_name: str,
    batch_size: int = 32,
) -> pl.DataFrame:
    """Generate embeddings, append them to a table, and return the DataFrame.

    DataFrame column names must match the existing destination table.
    Embeddings are generated before the database insert transaction begins.

    Args:
        df: DataFrame containing the text and any other columns to insert.
        column: Name of the column containing the input text.
        model: Name of the Ollama embedding model to use.
        engine: SQLAlchemy engine connected to the destination database.
        table_name: Name of an existing table in the default database schema.
        batch_size: Maximum number of texts per embedding request. Defaults to 32.

    Returns:
        The DataFrame with its ``embedding`` column, after insertion succeeds.

    Raises:
        ValueError: If batch_size is not positive or a DataFrame column does
            not exist in the destination table.
        polars.exceptions.ColumnNotFoundError: If column is not in df.
        ollama.ResponseError: If the Ollama API returns an error response.
        sqlalchemy.exc.SQLAlchemyError: If table lookup or insertion fails.
    """
    result = create_embeddings(df, column, model, batch_size=batch_size)
    save_embeddings_to_db(result, engine, table_name)
    return result
