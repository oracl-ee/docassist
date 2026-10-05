#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <omp.h>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using Results = std::vector<std::pair<int, float>>;

static void check_shapes(const FloatArray& vectors, const FloatArray& query) {
    if (vectors.ndim() != 2 || query.ndim() != 1)
        throw std::invalid_argument("vectors must be 2-D and query must be 1-D");
    if (query.shape(0) != vectors.shape(1))
        throw std::invalid_argument("query length does not match vector length");
}

// Pick the k highest scores without sorting the whole array: O(n log k)
static Results select_top_k(const std::vector<float>& scores, int k) {
    const int n = static_cast<int>(scores.size());
    const int kk = std::min(k, n);
    if (kk <= 0) return {};
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);  // 0, 1, 2, ..., n-1
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                      [&](int a, int b) { return scores[a] > scores[b]; });
    Results out;
    out.reserve(kk);
    for (int i = 0; i < kk; ++i) out.emplace_back(idx[i], scores[idx[i]]);
    return out;
}

// One thread
Results top_k(FloatArray vectors, FloatArray query, int k) {
    check_shapes(vectors, query);
    const int n = static_cast<int>(vectors.shape(0));
    const int d = static_cast<int>(vectors.shape(1));
    const float* V = vectors.data();
    const float* q = query.data();

    std::vector<float> scores(n);
    for (int i = 0; i < n; ++i) {
        const float* row = V + static_cast<size_t>(i) * d;
        float s = 0.0f;
        for (int j = 0; j < d; ++j) s += row[j] * q[j];
        scores[i] = s;
    }
    return select_top_k(scores, k);
}

// All CPU cores: each thread scores its own slice of rows
Results top_k_parallel(FloatArray vectors, FloatArray query, int k) {
    check_shapes(vectors, query);
    const int n = static_cast<int>(vectors.shape(0));
    const int d = static_cast<int>(vectors.shape(1));
    const float* V = vectors.data();
    const float* q = query.data();

    std::vector<float> scores(n);
    {
        py::gil_scoped_release release;  // let threads run without Python's lock
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            const float* row = V + static_cast<size_t>(i) * d;
            float s = 0.0f;
            for (int j = 0; j < d; ++j) s += row[j] * q[j];
            scores[i] = s;  // each i is written by exactly one thread: no race
        }
    }
    return select_top_k(scores, k);
}

int num_threads() { return omp_get_max_threads(); }

PYBIND11_MODULE(fastsearch, m) {
    m.doc() = "Fast top-k dot-product search for DocAssist";
    m.def("top_k", &top_k, py::arg("vectors"), py::arg("query"), py::arg("k") = 4);
    m.def("top_k_parallel", &top_k_parallel, py::arg("vectors"), py::arg("query"),
          py::arg("k") = 4);
    m.def("num_threads", &num_threads, "Number of OpenMP threads available");
}