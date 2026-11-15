"""Check the PostgreSQL connection and print the server version."""

from sqlalchemy import text

from vector_db.create_db import get_engine


def main() -> None:
    engine = get_engine()

    with engine.connect() as conn:
        result = conn.execute(text("SELECT version()"))
        print(result.scalar())


if __name__ == "__main__":
    main()
