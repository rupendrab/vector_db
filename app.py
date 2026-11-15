"""Streamlit interface for semantic search over FIQA documents."""

import logging

import ollama
import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from src.vector_db.create_db import get_engine
from src.vector_db.db import semantic_search_fiqa


def main() -> None:
    """Render the search form and document previews."""
    st.set_page_config(page_title="FIQA Search", page_icon="🔎", layout="centered")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    st.title("FIQA Search")
    st.caption("Find financial answers by meaning. Ask a question in your own words.")

    with st.form("search"):
        query = st.text_input(
            "Your question",
            placeholder="Should I pay off credit card debt before investing?",
        )
        limit = st.slider("Number of results", min_value=1, max_value=20, value=5)
        submitted = st.form_submit_button("Search", type="primary")

    if submitted:
        st.session_state.pop("search_results", None)
        st.session_state.pop("searched_query", None)
        if not query.strip():
            st.warning("Enter a question to search.")
        else:
            try:
                with st.spinner("Finding relevant documents…"):
                    results = semantic_search_fiqa(
                        get_engine(), query.strip(), limit=limit
                    )
                st.session_state.search_results = results
                st.session_state.searched_query = query.strip()
            except (ollama.ResponseError, ConnectionError):
                st.error(
                    "Could not generate the query embedding. Check that Ollama is "
                    "running and nomic-embed-text is available."
                )
            except SQLAlchemyError:
                st.error(
                    "Could not search the database. Check that PostgreSQL is "
                    "running and fiqa_documents contains your embeddings."
                )

    if "search_results" not in st.session_state:
        return

    results = st.session_state.search_results
    st.subheader("Search results")
    st.caption(f"{results.height} results for: {st.session_state.searched_query}")
    if results.is_empty():
        st.info("No documents found.")
        return

    for rank, row in enumerate(results.iter_rows(named=True), start=1):
        with st.container(border=True):
            st.markdown(f"**{rank}. Document {row['document_id']}**")
            if row["similarity"] is not None:
                st.caption(f"Cosine similarity: {row['similarity']:.3f}")
            content = row["content"] or ""
            words = content.split()
            preview = " ".join(words[:50])
            st.text(preview + ("…" if len(words) > 50 else ""))
            if len(words) > 50:
                with st.expander("More — full document"):
                    st.text(content)


if __name__ == "__main__":
    main()
