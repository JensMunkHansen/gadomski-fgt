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
#include <limits>
#include <random>

#include "cluster.hpp"
#include "fgt.hpp"

namespace fgt {

template <typename M, typename V>
M pick_cluster_centers(const Eigen::Ref<const M> points,
                       typename M::Index nclusters) {
    std::default_random_engine generator;
    std::uniform_int_distribution<typename M::Index> distribution(
        0, points.rows() - 1);
    typename M::Index cols = points.cols();
    M clusters(nclusters, cols);
    for (typename M::Index j = 0; j < nclusters; ++j) {
        typename M::Index index = distribution(generator);
        for (typename M::Index k = 0; k < cols; ++k) {
            clusters(j, k) = points(index, k);
        }
    }
    return clusters;
}

template <typename M, typename V>
Clustering<M, V> cluster(const Eigen::Ref<const M> points,
                         typename M::Index nclusters,
                         typename M::Scalar epsilon) {
    M clusters = pick_cluster_centers<M, V>(points, nclusters);
    return cluster<M, V>(points, nclusters, epsilon, clusters);
}

// Explicit Instantiations
template class Clustering<Matrix, Vector>;

template Clustering<Matrix, Vector>
cluster(const Eigen::Ref<const Matrix> points, Matrix::Index nclusters,
        Matrix::Scalar epsilon);

template Matrix
pick_cluster_centers<Matrix, Vector>(const Eigen::Ref<const Matrix> points,
                                     Matrix::Index nclusters);

} // namespace fgt
