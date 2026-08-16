# tests/conftest.py
from pathlib import Path
import pytest
import orcachip

DIMENSIONS = 128
EMBEDDINGS_PATH = Path(__file__).parent / "embeddings.bin"

@pytest.fixture(scope="module")
def dimensions():
    return DIMENSIONS

@pytest.fixture(scope="module")
def embeddings_path():
    return EMBEDDINGS_PATH

@pytest.fixture(scope="module")
def engine(embeddings_path, dimensions):
    assert embeddings_path.exists(), f"File missing: {embeddings_path}"
    return orcachip.VectorSearchEngine(str(embeddings_path), dimensions)

@pytest.fixture
def query(dimensions):
    return [0.1] * dimensions