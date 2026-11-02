"""Postgres + pgvector store.

* `products` holds the catalogue (relational filters run here: price, brand, category, stock).
* `chunks` holds text, embedding (HNSW cosine index) and a generated `tsvector` (GIN index).
* Hybrid search runs both retrievers in ONE SQL statement and fuses them with
  weighted reciprocal-rank fusion, so filters, vector search and keyword search
  share one round trip and one transaction snapshot.
* `index_metadata` records which embedding model built the vectors; the service
  refuses to start against an index built by a different model.

Connections come from a pool. On Cloud Run with Cloud SQL, point
`RAG_DATABASE_URL` at the unix socket:
`postgresql://USER:PASS@/retail?host=/cloudsql/PROJECT:REGION:INSTANCE`.
"""

from __future__ import annotations

import numpy as np
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from retail_rag.schemas import Chunk, Product, RetrievedChunk, SearchFilters
from retail_rag.store.base import VectorStore

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS index_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS products (
    product_id   TEXT PRIMARY KEY,
    product_name TEXT NOT NULL,
    description  TEXT NOT NULL,
    list_price   NUMERIC(10, 2),
    sale_price   NUMERIC(10, 2),
    brand        TEXT,
    category     TEXT,
    available    BOOLEAN,
    product_url  TEXT,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS products_price_idx ON products ((COALESCE(sale_price, list_price)));
CREATE INDEX IF NOT EXISTS products_brand_idx ON products (lower(brand));
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    product_id   TEXT NOT NULL REFERENCES products(product_id) ON DELETE CASCADE,
    chunk_index  INT NOT NULL,
    content      TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    embedding    vector({dim}) NOT NULL,
    -- the chunk's first line is its product header (name, brand, category): weight it above the body
    tsv          tsvector GENERATED ALWAYS AS (
                     setweight(to_tsvector('english', split_part(content, E'\n', 1)), 'A')
                  || setweight(to_tsvector('english', content), 'B')) STORED,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS chunks_product_idx ON chunks (product_id);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
"""

_PRODUCT_COLS = (
    "p.product_id, p.product_name, p.description, p.list_price, p.sale_price, p.brand, p.category, "
    "p.available, p.product_url"
)


def _filter_sql(f: SearchFilters) -> tuple[str, dict]:
    clauses, params = ["TRUE"], {}
    if f.min_price is not None:
        clauses.append("COALESCE(p.sale_price, p.list_price) >= %(min_price)s")
        params["min_price"] = f.min_price
    if f.max_price is not None:
        clauses.append("COALESCE(p.sale_price, p.list_price) <= %(max_price)s")
        params["max_price"] = f.max_price
    if f.category:
        clauses.append("p.category ILIKE %(category)s")
        params["category"] = f"%{f.category}%"
    if f.brand:
        clauses.append("lower(p.brand) = lower(%(brand)s)")
        params["brand"] = f.brand
    if f.in_stock_only:
        clauses.append("p.available IS NOT FALSE")
    return " AND ".join(clauses), params


class PostgresStore(VectorStore):
    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 10):
        self.pool = ConnectionPool(
            dsn,
            min_size=min_size,
            max_size=max_size,
            open=True,
            configure=self._configure,
            kwargs={"autocommit": False},
        )

    @staticmethod
    def _configure(conn):
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        register_vector(conn)

    def close(self):
        self.pool.close()

    def ping(self):
        with self.pool.connection() as conn:
            return conn.execute("SELECT 1").fetchone()[0] == 1

    # ------------------------------------------------------------ lifecycle
    def init(self, embedding_id, dim, rebuild=False):
        with self.pool.connection() as conn:
            if rebuild:
                conn.execute("DROP TABLE IF EXISTS chunks, products, index_metadata CASCADE")
            else:
                meta = self._meta(conn)
                if meta.get("embedding_id") not in (None, embedding_id):
                    self.check_compatible(embedding_id)
            conn.execute(SCHEMA.format(dim=int(dim)))
            conn.execute(
                "INSERT INTO index_metadata VALUES ('embedding_id', %s), ('dim', %s) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                (embedding_id, str(dim)),
            )

    @staticmethod
    def _meta(conn) -> dict:
        exists = conn.execute("SELECT to_regclass('index_metadata') IS NOT NULL").fetchone()[0]
        if not exists:
            return {}
        return dict(conn.execute("SELECT key, value FROM index_metadata").fetchall())

    def metadata(self):
        with self.pool.connection() as conn:
            return self._meta(conn)

    # -------------------------------------------------------------- writes
    def upsert_products(self, products):
        rows = [
            (
                p.product_id,
                p.product_name,
                p.description,
                p.list_price,
                p.sale_price,
                p.brand,
                p.category,
                p.available,
                p.product_url,
            )
            for p in products
        ]
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO products (product_id, product_name, description, list_price, sale_price, brand,
                                         category, available, product_url)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (product_id) DO UPDATE SET
                     product_name = EXCLUDED.product_name, description = EXCLUDED.description,
                     list_price = EXCLUDED.list_price, sale_price = EXCLUDED.sale_price, brand = EXCLUDED.brand,
                     category = EXCLUDED.category, available = EXCLUDED.available,
                     product_url = EXCLUDED.product_url, updated_at = now()""",
                rows,
            )

    def existing_hashes(self):
        with self.pool.connection() as conn:
            if conn.execute("SELECT to_regclass('chunks') IS NULL").fetchone()[0]:
                return {}
            return dict(conn.execute("SELECT chunk_id, content_hash FROM chunks").fetchall())

    def upsert_chunks(self, chunks, vectors):
        rows = [
            (c.chunk_id, c.product_id, c.chunk_index, c.content, c.content_hash, np.asarray(v, dtype="float32"))
            for c, v in zip(chunks, vectors, strict=True)
        ]
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO chunks (chunk_id, product_id, chunk_index, content, content_hash, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT (chunk_id) DO UPDATE SET content = EXCLUDED.content,
                     content_hash = EXCLUDED.content_hash, embedding = EXCLUDED.embedding,
                     chunk_index = EXCLUDED.chunk_index""",
                rows,
            )

    def delete_chunks(self, chunk_ids):
        if not chunk_ids:
            return
        with self.pool.connection() as conn:
            conn.execute("DELETE FROM chunks WHERE chunk_id = ANY(%s)", (list(chunk_ids),))

    def delete_products_except(self, keep_ids):
        with self.pool.connection() as conn:
            cur = conn.execute("DELETE FROM products WHERE NOT (product_id = ANY(%s))", (list(keep_ids),))
            return cur.rowcount

    # -------------------------------------------------------------- search
    def search(self, query_text, query_vec, filters, *, k, limit, mode="hybrid", w_vec=1.0, w_kw=1.0, rrf_k=60):
        where, params = _filter_sql(filters)
        params.update(
            {
                "q": np.asarray(query_vec, dtype="float32"),
                "text": query_text,
                "k": k,
                "limit": limit,
                "w_vec": w_vec if mode != "keyword" else 0.0,
                "w_kw": w_kw if mode != "vector" else 0.0,
                "rrf_k": rrf_k,
            }
        )
        vec_cte = (
            f"""
            SELECT c.chunk_id, row_number() OVER (ORDER BY c.embedding <=> %(q)s) AS rank
            FROM chunks c JOIN products p USING (product_id)
            WHERE {where}
            ORDER BY c.embedding <=> %(q)s LIMIT %(k)s"""
            if mode != "keyword"
            else "SELECT NULL::text AS chunk_id, NULL::bigint AS rank WHERE FALSE"
        )
        kw_cte = (
            f"""
            SELECT c.chunk_id, row_number() OVER (ORDER BY ts_rank(c.tsv, q.query, 1) DESC) AS rank
            FROM chunks c JOIN products p USING (product_id),
                 (SELECT to_tsquery('english', replace(plainto_tsquery('english', %(text)s)::text, '&', '|'))
                  AS query) q
            WHERE {where} AND c.tsv @@ q.query
            ORDER BY ts_rank(c.tsv, q.query, 1) DESC LIMIT %(k)s"""
            if mode != "vector"
            else "SELECT NULL::text AS chunk_id, NULL::bigint AS rank WHERE FALSE"
        )
        sql = f"""
            WITH vec AS ({vec_cte}), kw AS ({kw_cte}),
            ids AS (SELECT chunk_id FROM vec UNION SELECT chunk_id FROM kw)
            SELECT c.chunk_id, c.product_id, c.chunk_index, c.content, c.content_hash, {_PRODUCT_COLS},
                   1 - (c.embedding <=> %(q)s) AS vector_similarity,
                   vec.rank AS vector_rank, kw.rank AS keyword_rank,
                   COALESCE(%(w_vec)s / (%(rrf_k)s + vec.rank), 0)
                 + COALESCE(%(w_kw)s / (%(rrf_k)s + kw.rank), 0) AS score
            FROM ids JOIN chunks c USING (chunk_id) JOIN products p ON p.product_id = c.product_id
            LEFT JOIN vec USING (chunk_id) LEFT JOIN kw USING (chunk_id)
            ORDER BY score DESC, c.chunk_id
            LIMIT %(limit)s"""
        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            # Filters are applied after the HNSW scan, so widen the scan to keep k results under filters.
            cur.execute(f"SET LOCAL hnsw.ef_search = {max(40, int(k) * 4)}")
            rows = cur.execute(sql, params).fetchall()
        out = []
        for r in rows:
            product = Product(
                product_id=r["product_id"],
                product_name=r["product_name"],
                description=r["description"],
                list_price=float(r["list_price"]) if r["list_price"] is not None else None,
                sale_price=float(r["sale_price"]) if r["sale_price"] is not None else None,
                brand=r["brand"],
                category=r["category"],
                available=r["available"],
                product_url=r["product_url"],
            )
            chunk = Chunk(
                chunk_id=r["chunk_id"],
                product_id=r["product_id"],
                chunk_index=r["chunk_index"],
                content=r["content"],
                content_hash=r["content_hash"],
            )
            out.append(
                RetrievedChunk(
                    chunk=chunk,
                    product=product,
                    score=float(r["score"]),
                    vector_similarity=float(r["vector_similarity"]),
                    vector_rank=r["vector_rank"],
                    keyword_rank=r["keyword_rank"],
                )
            )
        return out

    def stats(self):
        with self.pool.connection() as conn:
            if conn.execute("SELECT to_regclass('chunks') IS NULL").fetchone()[0]:
                return {"backend": "postgres", "products": 0, "chunks": 0}
            products = conn.execute("SELECT count(*) FROM products").fetchone()[0]
            chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
            return {"backend": "postgres", "products": products, "chunks": chunks, **self._meta(conn)}
