#pragma once

#include <vector>

namespace Orcachip { 
    typedef const float* EmbeddingsPtr;

    /**
     * The search matches from vector searching
     * 
     * @param index index from the metadata
     * @param score similarity score from the metrics
     */
    struct Match {
        size_t index;
        float score;
    };

    /**
     * Diagnostics for profiling
     * 
     * @param search_time time in seconds
     * @param total_vectors total vectors search
     * @param vectors_per_second total vectors per second, throughput
     * @param similarity_metric_used similarity metric used as string
     */
    struct Diagnostics {
        double search_time;
        size_t total_vectors;
        float vectors_per_second;
        const char* similarity_metric_used;
    };

    /**
     * The results, output of the vector searching
     * 
     * @param matches vector of matches
     * @param diagnostics optional diagnostics for profiling
     */
    struct SearchResult {
        std::vector<Match> matches;
        Diagnostics diagnostics;
    };

    enum SimilarityMetric {
        COSINE,
        EUCLIDEAN,
        DOT_PRODUCT
    };
}