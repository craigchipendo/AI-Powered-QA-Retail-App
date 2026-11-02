import os

import pandas as pd
import pytest

from retail_rag.config import Settings
from retail_rag.ingestion.pipeline import ingest
from retail_rag.providers.embeddings import HashingEmbedder
from retail_rag.providers.llm import OfflineLLM
from retail_rag.service import RAGService
from retail_rag.store.memory import MemoryStore

ROWS = [
    (
        "p1",
        "Intex Kiddie Pool 58 inch",
        "A round inflatable kiddie pool for toddlers. Easy to set up in the backyard.",
        24.99,
        None,
        "Intex",
        "Toys | Outdoor Play | Swimming Pools & Spas | Kiddie Pools",
        True,
    ),
    (
        "p2",
        "Schwinn Training Wheels 12-20 inch",
        "Training wheels that help kids learn to ride. Fits 12 to 20 inch bikes.",
        28.17,
        None,
        "Schwinn",
        "Toys | Kids Bikes & Riding Toys | Accessories",
        True,
    ),
    (
        "p3",
        "Top Trumps Snakes Card Game",
        "A card game packed with facts about 30 snakes. Fun and educational.",
        9.95,
        None,
        "Top Trumps",
        "Toys | Games & Puzzles | Card Games",
        True,
    ),
    (
        "p4",
        "Pool Leaf Skimmer Net",
        "Fine mesh net for scooping leaves and bugs out of the swimming pool.",
        13.99,
        11.5,
        "Purity Pool",
        "Toys | Outdoor Play | Swimming Pools & Spas | Pool Cleaning Tools",
        False,
    ),
    (
        "p5",
        "Toddler Bike Helmet",
        "Lightweight helmet for toddlers with adjustable straps and bright colours.",
        37.83,
        None,
        "Schwinn",
        "Toys | Kids Bikes & Riding Toys | Protective Gear",
        True,
    ),
]


@pytest.fixture
def catalogue_csv(tmp_path):
    df = pd.DataFrame(
        ROWS,
        columns=[
            "product_id",
            "product_name",
            "description",
            "list_price",
            "sale_price",
            "brand",
            "category",
            "available",
        ],
    )
    path = tmp_path / "catalogue.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def settings(tmp_path, catalogue_csv):
    return Settings(
        vector_store="memory",
        memory_store_path=tmp_path / "index",
        catalogue_csv=catalogue_csv,
        log_json=False,
        log_level="WARNING",
    )


@pytest.fixture
def service(settings):
    store, embedder = MemoryStore(settings.memory_store_path), HashingEmbedder(64)
    ingest(settings.catalogue_csv, store, embedder, rebuild=True)
    return RAGService(settings, store, embedder, OfflineLLM())


def postgres_url():
    return os.environ.get("RAG_TEST_DATABASE_URL")
