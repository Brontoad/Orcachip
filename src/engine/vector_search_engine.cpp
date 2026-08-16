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

VectorSearchEngine::~VectorSearchEngine() { this->clean_embeddings_ptr(); }

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
    size_t max_results,
    SimilarityMetric similarity_metric,
    bool show_diagnostics
) {
    SearchResult search_result = {};

    // Handle zero max results, or zero vectors
    if (max_results == 0 || this->total_vectors == 0) { return {}; }

    // Handle improper dimensions
    if (query.size() != this->dimensions) {
        throw std::invalid_argument("Query size (" + std::to_string(query.size()) + ") does not match engine dimensions (" + std::to_string(this->dimensions) + ").");
    }

    // Lower scores are prioritized in Euclidean distance
    bool is_higher_better = (similarity_metric != SimilarityMetric::EUCLIDEAN);

    // If similarity metric is cosine similarity, precalculate the normalized query magnitude
    float sqrt_magnitude_query = 0.0f;
    if (similarity_metric == SimilarityMetric::COSINE) {
        for (float val : query) { sqrt_magnitude_query += val * val; }
        sqrt_magnitude_query = std::sqrt(sqrt_magnitude_query);
    }

    // Create a priority queue to arrange the lowest scored vector
    auto compare = [is_higher_better](const Match &a, const Match &b) { 
        return is_higher_better ? (a.score > b.score) : (a.score < b.score); 
    };
    std::priority_queue<Match, std::vector<Match>, decltype(compare)> matches(compare);

    // Before proceeding to the search loop, record the searching time if possible
    std::chrono::high_resolution_clock::time_point start_time;

    if (show_diagnostics) { start_time = std::chrono::high_resolution_clock::now(); }

    // Get the most relevant search item in the dataset
    // by getting lowest scored vector, and getting the maximum amount
    // based on the results
    for (size_t i = 0; i < this->total_vectors; i++) {
        float score = 0.0f;
        switch (similarity_metric) {
            case SimilarityMetric::COSINE:
                score = this->cosine_similarity(query, i, sqrt_magnitude_query);
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
        } else {
            bool is_better = is_higher_better ? (score > matches.top().score) : (score < matches.top().score);
            if (is_better) {
                matches.pop();
                matches.push({i, score});
            }
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

    this->clean_embeddings_ptr();

    std::filesystem::path file_path(file_name);

    // Check whether the file is a .bin file or not
    if (file_path.extension() != ".bin") {
        throw std::runtime_error("Invalid file extension. Expected a '.bin' file, got: " + file_path.extension().string());
    }

    size_t bytes_per_vector = this->dimensions * sizeof(float);

    #ifdef _WIN32
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
        if (this->total_bytes == 0 || this->total_bytes < bytes_per_vector || (this->total_bytes % bytes_per_vector != 0)) {
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

        CloseHandle(hFile);
        CloseHandle(hFileMap);
    #else
        // Open file
        int fd = open(file_name, O_RDONLY);
        if (fd == -1) {
            throw std::runtime_error("Error opening file. Code: " + std::string(strerror(errno)));
        }

        // Get file size, total floats, total vectors
        struct stat buf;
        if (fstat(fd, &buf) == -1) {
            close(fd);
            throw std::runtime_error("Error fetching file size. Code: " + std::string(strerror(errno)));
        }
        
        // Ensure total_bytes can safely contain at least one complete vector
        this->total_bytes = static_cast<size_t>(buf.st_size);
        if (this->total_bytes == 0 || this->total_bytes < bytes_per_vector || (this->total_bytes % bytes_per_vector != 0)) {
            close(fd);
            throw std::runtime_error("File size is smaller than a single vector dimension.");
        }

        // Create file mapping object
        void* fileMap = mmap(NULL, this->total_bytes, PROT_READ, MAP_SHARED, fd);
        if (fileMap == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("Error creating file mapping. Code: " + std::string(strerror(errno)));
        }

        this->embeddings_ptr = static_cast<EmbeddingsPtr>(fileMap);

        close(fd);
    #endif

    this->embedding_floats = total_bytes / sizeof(float);
    this->total_vectors = embedding_floats / this->dimensions;
}

// Cleans the mapped memory
void VectorSearchEngine::clean_embeddings_ptr() { 
    if (this->embeddings_ptr != nullptr) {
        #ifdef _WIN32
            UnmapViewOfFile(static_cast<LPCVOID>(this->embeddings_ptr));
        #else
            munmap(static_cast<void*>(this->embeddings_ptr), this->total_bytes);
        #endif

        this->embeddings_ptr = nullptr;
        this->total_bytes = 0;
        this->embedding_floats = 0;
        this->total_vectors = 0;
    }
}

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

float VectorSearchEngine::cosine_similarity(const std::vector<float>& query, size_t index, float sqrt_magnitude_query) {
    EmbeddingsPtr target_vector = this->embeddings_ptr + this->dimensions * index;
    float dot_product = 0.0f;
    float magnitude_target = 0.0f;

    for (size_t i = 0; i < this->dimensions; i++) { 
        dot_product += target_vector[i] * query[i];
        magnitude_target += target_vector[i] * target_vector[i]; 
    }

    if (sqrt_magnitude_query == 0.0f || magnitude_target == 0.0f) { return 0.0f; }

    return dot_product / (sqrt_magnitude_query * std::sqrt(magnitude_target));
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
