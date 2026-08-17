# orcachip

**orcachip** is a fast, lightweight vector similarity search library for Python powered by C++. It allows you to search vector embeddings directly from disk with near-zero RAM overhead and instant startup times.

## How It Works

Instead of loading large vector datasets into system memory, **orcachip** memory-maps flat binary files (**float32**) directly from disk. The operating system manages page caching automatically. When you execute a search, the C++ engine computes distance metrics directly on raw pointer addresses and returns top results back to Python with virtually no binding overhead.

## Installation

```bash
pip install orcachip
```

## Quick Start

Initialize **VectorSearchEngine** with the path to your binary embedding file and your embedding dimension:

```python
# Sample initialization
from orcachip import VectorSearchEngine, SimilarityMetric

engine = VectorSearchEngine("embeddings.bin", dimension=128)

```

Run a vector search using **vector_search()**:

```python
results = engine.vector_search(
    query,                                     # 1. Query vector (list or np.ndarray of floats)
    max_results=5,                             # 2. Number of top results to return
    similarity_metric=SimilarityMetric.COSINE, # 3. Options: COSINE, EUCLIDEAN, DOT_PRODUCT
    show_diagnostics=False                     # Optional: enable diagnostics profiling
)
```

### Return Structure

**vector_search()** returns a **SearchResult** object containing search matches and profiling data:

**Matches (SearchResult.matches)**

| Field | Type | Description |
| --- | --- | --- |
| **index** | **int** | Zero-based index of the matching vector in your binary file |
| **score** | **float** | Computed similarity or distance score |

**Diagnostics (SearchResult.diagnostics)** *(populated when show_diagnostics=True)*

| Field | Type | Description |
| --- | --- | --- |
| **search_time** | **float** | Search execution time in seconds |
| **total_vectors** | **int** | Total number of vectors evaluated |
| **vectors_per_second** | **float** | Search throughput (vectors processed per second) |
| **similarity_metric_used** | **str** | Name of the active metric (**COSINE**, **EUCLIDEAN**, **DOT_PRODUCT**) |

## Data Preparation

In case you need a guide for preparing your data for the search engine, save your matrix as a contiguous **float32** binary file:

```python
import numpy as np

# Export float32 embeddings matrix to binary
embeddings = np.array(your_vectors, dtype=np.float32)
embeddings.tofile("embeddings.bin")
```

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Brontoad/Orcachip/issues).

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.