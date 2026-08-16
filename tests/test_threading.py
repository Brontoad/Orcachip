from concurrent.futures import ThreadPoolExecutor

def test_thread_safety_concurrent_searches(engine, query):
    """Executes searches across 8 concurrent Python threads."""
    num_threads = 8
    searches_per_thread = 200
    
    def worker():
        for _ in range(searches_per_thread):
            res = engine.vector_search(query, max_results=5)
            assert len(res.matches) <= 5

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker) for _ in range(num_threads)]
        for future in futures:
            future.result()  # Re-raises exceptions thrown in threads