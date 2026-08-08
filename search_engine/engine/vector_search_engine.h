#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <queue>
#include <string>

#include "windows.h"

#include "../utils/utils.h"

namespace Orcachip {
    /**
     * The vector search engine
     */
    class VectorSearchEngine {
        public:
            EmbeddingsPtr embeddings_ptr;
            size_t dimensions;
            size_t total_bytes;
            size_t embedding_floats;
            size_t total_vectors;
        
        public:
            /**
             * Constructs a new vector search engine, by loading the embeddings file to it
             * 
             * @param file_name file  name of the .bin file
             * @param dimensions dimensions produced by the selected model
             * 
             * @returns Vector search engine object
             */
            VectorSearchEngine(const char* file_name, size_t dimensions);
            
            ~VectorSearchEngine();

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
            SearchResult vector_search(
                const std::vector<float>& query,
                size_t max_results = 10,
                SimilarityMetric similarity_metric = SimilarityMetric::DOT_PRODUCT,
                bool show_diagnostics = false
            );

        private:
            // Loads the embeddings from the memory and memory mapped to the RAM
            // TODO: Add another option for non windows system, for now Linux
            void get_embeddings_from_file(const char* file_name);

            // Cleans the mapped memory
            void clean_embeddings_ptr() { UnmapViewOfFile(this->embeddings_ptr); }

            // Helper function to just display the metric to string in diagnostics
            const char* metric_to_string(SimilarityMetric similarity_metric);

            // Method used to get the distances of vectors
            float dot_product(const std::vector<float>& query, size_t index);
            float cosine_similarity(const std::vector<float>& query, size_t index);
            float euclidean_distance(const std::vector<float>& query, size_t index);
        };

}
