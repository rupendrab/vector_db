"""Create the pgvector extension and documents table."""

from functools import lru_cache

from sqlalchemy import Engine, create_engine, text


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(
        "postgresql+psycopg://postgres:postgres123@localhost:5432/vectordb"
    )

def create_db() -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS documents (
                id BIGSERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding VECTOR(384)
            )
        """))


if __name__ == "__main__":
    create_db()
