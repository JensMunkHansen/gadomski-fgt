// fgt — fast Gauss transforms
// Copyright (C) 2016 Peter J. Gadomski <pete.gadomski@gmail.com>
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA

#include <cmath>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4820) // paddding
#pragma warning(disable : 4626) // assignment implicitly deleted
#pragma warning(disable : 5027) // move assignment implicitly deleted
#pragma warning(disable : 4127) // conditional expression is constant
#pragma warning(disable : 4365) // signed/unsigned
#endif
#include "nanoflann.hpp"
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include "fgt.hpp"

namespace fgt {
namespace {

template <typename M, typename V>
struct MatrixAdaptor {
    size_t kdtree_get_point_count() const { return size_t(m_rows); }
    typename M::Scalar kdtree_distance(const typename M::Scalar* p1,
                                       const typename M::Index idx_p2,
                                       typename M::Index) const {
        typename M::Scalar distance = 0.0;
        for (typename M::Index k = 0; k < m_cols; ++k) {
            typename M::Scalar temp = p1[k] - m_data[idx_p2 * m_cols + k];
            distance += temp * temp;
        }
        return distance;
    }
    typename M::Scalar kdtree_get_pt(const typename M::Index idx,
                                     typename M::Index dim) const {
        return m_data[m_cols * idx + dim];
    }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const {
        return false;
    }

    const typename M::Scalar* m_data;
    typename M::Index m_rows;
    typename M::Index m_cols;
};
} // namespace

template <typename M, typename V>
V direct_tree(const Eigen::Ref<const M> source,
              const Eigen::Ref<const M> target, typename M::Scalar bandwidth,
              typename M::Scalar epsilon) {
    return DirectTree<M, V>(source, bandwidth, epsilon).compute(target);
}

template <typename M, typename V>
V direct_tree(const Eigen::Ref<const M> source,
              const Eigen::Ref<const M> target, typename M::Scalar bandwidth,
              typename M::Scalar epsilon, const Eigen::Ref<const V> weights) {
    return DirectTree<M, V>(source, bandwidth, epsilon)
        .compute(target, weights);
}

template <typename M, typename V>
struct DirectTree<M, V>::NanoflannTree {
    typedef nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<typename M::Scalar, MatrixAdaptor<M, V>>,
        MatrixAdaptor<M, V>>
        tree_t;

    NanoflannTree(const Eigen::Ref<const M> source)
        : matrix_adaptor({source.data(), source.rows(), source.cols()}),
          tree(int(source.cols()), matrix_adaptor) {}

    NanoflannTree(const NanoflannTree&) = delete;
    NanoflannTree& operator=(const NanoflannTree&) = delete;
    NanoflannTree& operator=(NanoflannTree&&) = delete;

    MatrixAdaptor<M, V> matrix_adaptor;
    tree_t tree;
};

template <typename M, typename V>
DirectTree<M, V>::DirectTree(const Eigen::Ref<const M> source,
                             typename M::Scalar bandwidth,
                             typename M::Scalar epsilon)
    : Transform<M, V>(source, bandwidth),
      m_epsilon(epsilon),
      m_tree(new DirectTree<M, V>::NanoflannTree(this->source())) {
    m_tree->tree.buildIndex();
}

template <typename M, typename V>
DirectTree<M, V>::~DirectTree() {}

template <typename M, typename V>
V DirectTree<M, V>::compute_impl(const Eigen::Ref<const M> target,
                                 const Eigen::Ref<const V> weights) const {
    typename M::Scalar h2 = this->bandwidth() * this->bandwidth();
    typename M::Scalar cutoff_radius =
        this->bandwidth() * std::sqrt(std::log(1.0 / epsilon()));
    typename M::Scalar r2 = cutoff_radius * cutoff_radius;
    typename M::Index rows_source = this->source().rows();
    typename M::Index rows_target = target.rows();
    V g = V::Zero(rows_target);
    typename M::Index cols = this->source().cols();

    nanoflann::SearchParams params;
    params.sorted = false;

#pragma omp parallel for
    for (typename M::Index j = 0; j < rows_target; ++j) {
        std::vector<std::pair<size_t, typename M::Scalar>> indices_distances;
        indices_distances.reserve(unsigned(rows_source));
        size_t nfound = m_tree->tree.radiusSearch(&target.data()[j * cols], r2,
                                                  indices_distances, params);
        for (size_t i = 0; i < nfound; ++i) {
            auto entry = indices_distances[i];
            g[j] += weights[signed(entry.first)] * std::exp(-entry.second / h2);
        }
    }
    return g;
}
// Explicit instantiations
template class DirectTree<Matrix, Vector>;

template Vector
direct_tree<Matrix, Vector>(const Eigen::Ref<const Matrix> source,
                            const Eigen::Ref<const Matrix> target,
                            Matrix::Scalar bandwidth, Matrix::Scalar epsilon);

template Vector
direct_tree<Matrix, Vector>(const Eigen::Ref<const Matrix> source,
                            const Eigen::Ref<const Matrix> target,
                            Matrix::Scalar bandwidth, Matrix::Scalar epsilon,
                            const Eigen::Ref<const Vector> weights);

} // namespace fgt
