import time
import numpy as np

def test_search_speed_benchmark(engine, query):
    """Measures latency statistics and throughput over 1,000 iterations."""
    latencies_ms = []
    iterations = 1000
    
    for _ in range(iterations):
        start = time.perf_counter()
        result = engine.vector_search(query, max_results=10, show_diagnostics=True)
        end = time.perf_counter()
        
        latencies_ms.append((end - start) * 1000)
        
    p50 = np.percentile(latencies_ms, 50)
    p99 = np.percentile(latencies_ms, 99)
    avg_throughput = (engine.total_vectors / (p50 / 1000))
    
    print(f"\n Latency p50: {p50:.3f} ms | p99: {p99:.3f} ms")
    print(f" Throughput: {avg_throughput:,.0f} vectors/sec")
    print(f" C++ Reported: {result.diagnostics.vectors_per_second:,.0f} vectors/sec")
    
    # Assert sub-50ms p50 requirement (adjust threshold as needed)
    assert p50 < 50.0