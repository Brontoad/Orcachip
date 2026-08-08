import json
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer

from api import vector_search
from api.request import query
from pre_processing.encoder import encode_to_vector

# Load the model, in this case, the all-MiniLM-L6-v2, 
# for quick text embedding
print("Loading model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load the metadata
print("Loading metadata...")
with open("metadata.json", "r", encoding="utf-8") as file: metadata = json.load(file)

# Load the search engine
try:
    vector_search_engine = vector_search.VectorSearchEngine("embeddings.bin", 384)
except Exception as e:
    print(f"{str(e)}")
    embeddings_data = None

# API Handlers
app = FastAPI()

# Loads the homepage
app.frontend("/", directory="api/client")

# Handler for vector searching
@app.post("/search")
async def search(query_request: query.QueryRequest):
    try:
        embeddings = encode_to_vector(query_request.query, model=model, to_list=True)
        raw_results = vector_search_engine.vector_search(
            query=embeddings, 
            max_results=10,
            similarity_metric=vector_search.SimilarityMetric.DOT_PRODUCT,
            show_diagnostics=True)
        results = {
            "search_results" : [{
                "score": match.score, 
                "index": match.index, 
                "data": metadata[match.index]} for match in raw_results.matches],
            "diagnostics" : {
                "search_time": raw_results.diagnostics.search_time,
                "total_vectors": raw_results.diagnostics.total_vectors,
                "vectors_per_second": raw_results.diagnostics.vectors_per_second,
                "similarity_metric_used": raw_results.diagnostics.similarity_metric_used,
            } if raw_results.diagnostics else None
        }
        return {"results": results}
    except Exception as e: return HTTPException(status_code=500, detail=str(e))
