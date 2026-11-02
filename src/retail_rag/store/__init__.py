from retail_rag.store.base import IndexMismatchError, VectorStore


def build_store(settings) -> VectorStore:
    """Create the configured store. Postgres dependencies are only imported when used."""
    if settings.vector_store == "postgres":
        from retail_rag.store.postgres import PostgresStore

        return PostgresStore(settings.database_url.get_secret_value(), settings.db_pool_min, settings.db_pool_max)
    from retail_rag.store.memory import MemoryStore

    return MemoryStore(settings.memory_store_path)


__all__ = ["IndexMismatchError", "VectorStore", "build_store"]
