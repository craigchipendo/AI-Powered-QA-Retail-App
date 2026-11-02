from retail_rag.ingestion.chunking import chunk_product, pack, split_sentences
from retail_rag.ingestion.loader import clean_text, load_catalogue
from retail_rag.ingestion.pipeline import ingest
from retail_rag.providers.embeddings import HashingEmbedder
from retail_rag.schemas import Product
from retail_rag.store.memory import MemoryStore


def test_clean_text():
    assert clean_text("<b>Safe</b> &amp; fun.Great  for 3+") == "Safe & fun. Great for 3+"
    assert clean_text('Pool 58\\" x 13\\"') == 'Pool 58" x 13"'


def test_load_catalogue_report(catalogue_csv):
    products, report = load_catalogue(catalogue_csv)
    assert len(products) == 5 and report["products"] == 5
    assert products[3].price == 11.5  # sale price wins


def test_chunks_carry_product_header():
    p = Product(
        product_id="x", product_name="Kite", description="Flies high. " * 200, brand="Acme", category="A | B | C"
    )
    chunks = chunk_product(p, size=300, overlap=40)
    assert len(chunks) > 1
    assert all(c.content.startswith("Product: Kite | Brand: Acme | Category: B > C") for c in chunks)
    assert len({c.content_hash for c in chunks}) >= 1 and chunks[1].chunk_id == "x:1"


def test_pack_respects_size_and_overlap():
    sents = split_sentences("One two three. Four five six. Seven eight nine. Ten eleven twelve.")
    chunks = pack(sents, size=32, overlap=16)
    assert all(len(c) <= 32 for c in chunks)
    assert chunks[1].startswith("Four five six.")  # overlap carried forward


def test_ingest_is_incremental(catalogue_csv, tmp_path):
    store, emb = MemoryStore(tmp_path / "idx"), HashingEmbedder(64)
    first = ingest(catalogue_csv, store, emb, rebuild=True)
    second = ingest(catalogue_csv, store, emb)
    assert first.chunks_embedded == first.chunks_total > 0
    assert second.chunks_embedded == 0 and second.chunks_unchanged == first.chunks_total

    reloaded = MemoryStore(tmp_path / "idx")  # persisted to disk
    assert reloaded.stats()["chunks"] == first.chunks_total

    import pandas as pd

    df = pd.read_csv(catalogue_csv)
    df.loc[0, "description"] = "A brand new description of the kiddie pool."
    df = df[df.product_id != "p3"]
    df.to_csv(catalogue_csv, index=False)
    third = ingest(catalogue_csv, store, emb)
    assert third.chunks_embedded == 1 and third.products_deleted == 1 and third.chunks_deleted >= 1
