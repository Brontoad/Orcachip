/**
 * Orcachip Vector Search Engine
 * 
 * Created by: Joshua Miguel Calulut, Brontoad | Founder & Developer | Brontoad Games
 */

#include "pybind11/stl.h"
#include "pybind11/pybind11.h"

#include "engine/vector_search_engine.h"

namespace py = pybind11;
using namespace Orcachip;

PYBIND11_MODULE(vector_search, m) {
    m.doc() = "Orcachip Vector Search Engine";

    py::class_<SearchResult>(m, "SearchResult")
        .def_readwrite("matches", &SearchResult::matches)
        .def_readwrite("diagnostics", &SearchResult::diagnostics);

    py::class_<Diagnostics>(m, "Diagnostics")
        .def_readwrite("search_time", &Diagnostics::search_time)
        .def_readwrite("total_vectors", &Diagnostics::total_vectors)
        .def_readwrite("vectors_per_second", &Diagnostics::vectors_per_second)
        .def_readwrite("similarity_metric_used", &Diagnostics::similarity_metric_used);

    py::class_<Match>(m, "Match")
        .def_readwrite("index", &Match::index)
        .def_readwrite("score", &Match::score);
    
    py::enum_<SimilarityMetric>(m, "SimilarityMetric")
        .value("DOT_PRODUCT", SimilarityMetric::DOT_PRODUCT)
        .value("COSINE", SimilarityMetric::COSINE)
        .value("EUCLIDEAN", SimilarityMetric::EUCLIDEAN);

    py::class_<VectorSearchEngine>(m, "VectorSearchEngine")
        .def(py::init<const char*, size_t>(), py::arg("file_name"), py::arg("dimensions"))
        .def("vector_search", &VectorSearchEngine::vector_search,
            py::arg("query"),
            py::arg("max_results") = 10,
            py::arg("similarity_metric")  = SimilarityMetric::DOT_PRODUCT,
            py::arg("show_diagnostics") = false);
}