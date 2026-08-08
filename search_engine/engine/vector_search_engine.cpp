#include "vector_search_engine.h"

using namespace Orcachip;
/**
 * Constructs a new vector search engine, by loading the embeddings file to it
 * 
 * @param file_name file  name of the .bin file
 * @param dimensions dimensions produced by the selected model
 * 
 * @returns Vector search engine object
 */
VectorSearchEngine::VectorSearchEngine(const char* file_name, size_t dimensions) {
    if (dimensions == 0) { throw std::invalid_argument("Dimensions must be greater than 0."); }
    this->dimensions = dimensions;
    this->get_embeddings_from_file(file_name);
}

VectorSearchEngine::~VectorSearchEngine() {this->clean_embeddings_ptr();}

/**
 * Function for vector searching
 * 
 * @param query query of floats
 * @param max_results max results returned
 * @param similarity_metric type of similarity metric to use
 * @param show_diagnostics option to show diagnostics
 * 
 * @returns Vector of matches
 */
SearchResult VectorSearchEngine::vector_search(
    const std::vector<float>& query,
    size_t max_results = 10,
    SimilarityMetric similarity_metric = SimilarityMetric::DOT_PRODUCT,
    bool show_diagnostics = false
) {
    SearchResult search_result = {};

    // Handle zero max results, or zero vectors
    if (max_results == 0 || this->total_vectors == 0) { return {}; }

    // Handle improper dimensions
    if (query.size() != this->dimensions) {
        throw std::invalid_argument("Query size (" + std::to_string(query.size()) + ") does not match engine dimensions (" + std::to_string(this->dimensions) + ").");
    }

    // Create a priority queue to arrange the lowest scored vector
    auto compare = [](const Match &a, const Match &b) { return a.score > b.score; };
    std::priority_queue<Match, std::vector<Match>, decltype(compare)> matches(compare);

    // Before proceeding to the search loop, record the searching time if possible
    std::chrono::high_resolution_clock::time_point start_time;

    if (show_diagnostics) { start_time = std::chrono::high_resolution_clock::now(); }

    // Get the most relevant search item in the dataset
    // by getting lowest scored vector, and getting the maximum amount
    // based on the results
    float min_score_threshold = -1e9f;
    for (size_t i = 0; i < this->total_vectors; i++) {
        float score = 0.0f;
        switch (similarity_metric) {
            case SimilarityMetric::COSINE:
                score = this->cosine_similarity(query, i);
                break;
            case SimilarityMetric::EUCLIDEAN:
                score = this->euclidean_distance(query, i);
                break;
            default:
                score = this->dot_product(query, i);
                break;
        }
        
        if (matches.size() < max_results) {
            matches.push({i, score});
            if (matches.size() == max_results) { min_score_threshold = matches.top().score; }
        } else if (score > min_score_threshold) {
            matches.pop();
            matches.push({i, score});
            min_score_threshold = matches.top().score;
        }
    }

    if (show_diagnostics) {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;

        Diagnostics diagnostics;
        diagnostics.search_time = duration.count();
        diagnostics.total_vectors = this->total_vectors;
        diagnostics.similarity_metric_used = metric_to_string(similarity_metric);

        diagnostics.vectors_per_second = (diagnostics.search_time > 0.0) 
            ? (static_cast<double>(this->total_vectors) / diagnostics.search_time) 
            : 0.0;
        
        search_result.diagnostics = diagnostics;
    }

    size_t result_count = matches.size();
    
    // Load all of the results from priority queue to the output vector
    // output vector of resulting indices
    std::vector<Match> results(result_count);

    size_t idx = result_count;
    while (!matches.empty()) {
        results[--idx] = matches.top();
        matches.pop();
    }

    search_result.matches = results;

    return search_result;
}

// Loads the embeddings from the memory and memory mapped to the RAM
// TODO: Add another option for non windows system, for now Linux
void VectorSearchEngine::get_embeddings_from_file(const char* file_name) {
    std::filesystem::path file_path(file_name);

    // Check whether the file is a .bin file or not
    if (file_path.extension() != ".bin") {
        throw std::runtime_error("Invalid file extension. Expected a '.bin' file, got: " + file_path.extension().string());
    }

    // Open file
    HANDLE hFile = CreateFileA(file_name, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_READONLY, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("Error opening file. Code: " + std::to_string(GetLastError()));
    }

    // Get file size, total floats, total vectors
    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(hFile, &file_size)) {
        CloseHandle(hFile);
        throw std::runtime_error("Error fetching file size. Code: " + std::to_string(GetLastError()));
    }

    this->total_bytes = static_cast<size_t>(file_size.QuadPart);
    // Ensure total_bytes can safely contain at least one complete vector
    size_t bytes_per_vector = this->dimensions * sizeof(float);
    if (this->total_bytes < bytes_per_vector) {
        CloseHandle(hFile);
        throw std::runtime_error("File size is smaller than a single vector dimension.");
    }

    // Create file mapping object
    HANDLE hFileMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (hFileMap == NULL) {
        CloseHandle(hFile);
        throw std::runtime_error("Error creating file mapping. Code: " + std::to_string(GetLastError()));
    }

    // Create file map view
    this->embeddings_ptr = static_cast<EmbeddingsPtr>(MapViewOfFileEx(hFileMap, FILE_MAP_READ, 0, 0, 0, NULL));
    if (embeddings_ptr == nullptr) {
        CloseHandle(hFile);
        CloseHandle(hFileMap);
        throw std::runtime_error("Error create map view of the file map. Code: " + std::to_string(GetLastError()));
    }

    this->embedding_floats = total_bytes / sizeof(float);
    this->total_vectors = embedding_floats / this->dimensions;

    CloseHandle(hFile);
    CloseHandle(hFileMap);
}

// Cleans the mapped memory
void VectorSearchEngine::clean_embeddings_ptr() { UnmapViewOfFile(this->embeddings_ptr); }

// Helper function to just display the metric to string in diagnostics
const char* VectorSearchEngine::metric_to_string(SimilarityMetric similarity_metric) {
    switch(similarity_metric) {
        case SimilarityMetric::COSINE: return "Cosine Similarity";
        case SimilarityMetric::EUCLIDEAN: return "Euclidean Distance";
        default: return "Dot Product";
    }
}

// Method used to get the distances of vectors
float VectorSearchEngine::dot_product(const std::vector<float>& query, size_t index) {
    EmbeddingsPtr target_vector = this->embeddings_ptr + this->dimensions * index;

    float score = 0.0f;
    
    for (size_t i = 0; i < this->dimensions; i++) { score += target_vector[i] * query[i]; }

    return score;
}

float VectorSearchEngine::cosine_similarity(const std::vector<float>& query, size_t index) {
    EmbeddingsPtr target_vector = this->embeddings_ptr + this->dimensions * index;
    float dot_product = 0.0f;
    float magnitude_query = 0.0f;
    float magnitude_target = 0.0f;

    for (size_t i = 0; i < this->dimensions; i++) { 
        dot_product += target_vector[i] * query[i]; 
        magnitude_query += query[i] * query[i]; 
        magnitude_target += target_vector[i] * target_vector[i]; 
    }

    if (magnitude_query == 0.0f || magnitude_target == 0.0f) { return 0.0f; }

    return dot_product / (std::sqrt(magnitude_query) * std::sqrt(magnitude_target));
}

float VectorSearchEngine::euclidean_distance(const std::vector<float>& query, size_t index) {
    EmbeddingsPtr target_vector = this->embeddings_ptr + this->dimensions * index;
    float score = 0.0f;
    float difference = 0.0f;

    for (size_t i = 0; i < this->dimensions; i++) { 
        difference = target_vector[i] - query[i]; 
        score += difference * difference;
    }

    return std::sqrt(score);
}

