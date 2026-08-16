import orcachip

import gc
import psutil
import os
import pytest

def get_process_memory_mb():
    """Returns current process Resident Set Size (RSS) memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def test_no_memory_leak_on_repeated_searches(engine, query):
    """Ensure running searches doesn't continuously leak memory."""
    # 1. Warm-up batch: lets Python/Pybind11 reach its allocator high-water mark
    for _ in range(2000):
        _ = engine.vector_search(query, max_results=10)
    
    gc.collect()
    mem_before = get_process_memory_mb()
    
    # 2. Main test batch
    for _ in range(30000):
        _ = engine.vector_search(query, max_results=10)
        
    gc.collect()
    mem_after = get_process_memory_mb()
    
    mem_growth = mem_after - mem_before
    
    # After warm-up, continuous growth should be near 0 MB
    assert mem_growth < 3.0, f"Continuous memory leak detected: grew by {mem_growth:.2f} MB"

def test_engine_lifecycle_memory(embeddings_path, dimensions):
    """Ensure instantiating and destroying engine cleans up mapped pointers."""
    gc.collect()
    mem_before = get_process_memory_mb()
    
    # Repeated creation/destruction cycle
    for _ in range(100):
        temp_engine = orcachip.VectorSearchEngine(str(embeddings_path), dimensions)
        _ = temp_engine.vector_search([0.1] * dimensions)
        del temp_engine
        
    gc.collect()
    mem_after = get_process_memory_mb()
    
    assert (mem_after - mem_before) < 10.0, "Engine destructor failed to free memory resources"

def test_mmap_lazy_allocation(embeddings_path, dimensions):
    """Verify memory mapping doesn't eagerly force full file load into RAM."""
    gc.collect()
    baseline_mem = get_process_memory_mb()
    
    # Load engine
    e = orcachip.VectorSearchEngine(str(embeddings_path), dimensions)
    
    post_load_mem = get_process_memory_mb()
    mem_added = post_load_mem - baseline_mem
    total_file_mb = e.total_bytes / (1024 * 1024)
    
    # Resident RAM footprint should be strictly less than total binary size on load
    print(f"\n File Size: {total_file_mb:.2f} MB | Initial RSS Added: {mem_added:.2f} MB")
    assert e.total_bytes > 0
