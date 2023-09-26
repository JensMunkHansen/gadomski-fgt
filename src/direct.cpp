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

#include <cassert>
#include <cmath>

#include "fgt.hpp"

namespace fgt {

template <typename M, typename V>
Direct<M, V>::Direct(const MatrixRef source, typename M::Scalar bandwidth)
    : Transform<M, V>(source, bandwidth) {}

template <typename M, typename V>
V Direct<M, V>::compute_impl(const MatrixRef target,
                             const VectorRef weights) const {
    typename M::Scalar h2 = this->bandwidth() * this->bandwidth();
    MatrixRef source = this->source();
    auto rows_source = source.rows();
    auto rows_target = target.rows();
    V g = V::Zero(rows_target);
#pragma omp parallel for
    for (typename M::Index j = 0; j < rows_target; ++j) {
        for (typename M::Index i = 0; i < rows_source; ++i) {
            typename M::Scalar distance =
                (source.row(i) - target.row(j)).array().pow(2).sum();
            g[j] += weights[i] * std::exp(-distance / h2);
        }
    }
    return g;
}

template <typename M, typename V>
V direct(const Eigen::Ref<const M> source, const Eigen::Ref<const M> target,
         typename M::Scalar bandwidth) {
    return Direct<M, V>(source, bandwidth).compute(target);
}

template <typename M, typename V>
V direct(const Eigen::Ref<const M> source, const Eigen::Ref<const M> target,
         typename M::Scalar bandwidth, const Eigen::Ref<const V> weights) {
    return Direct<M, V>(source, bandwidth).compute(target, weights);
}
// Explicit instantiations
template class Direct<Matrix, Vector>;
template Vector direct(const Eigen::Ref<const Matrix> source,
                       const Eigen::Ref<const Matrix> target,
                       Matrix::Scalar bandwidth,
                       const Eigen::Ref<const Vector> weights);

template Vector direct(const Eigen::Ref<const Matrix> source,
                       const Eigen::Ref<const Matrix> target,
                       Matrix::Scalar bandwidth);

} // namespace fgt
