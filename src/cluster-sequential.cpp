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

#include "cluster.hpp"

namespace fgt {

template <typename M, typename V>
Clustering<M, V> cluster(const Eigen::Ref<const M> points,
                         typename M::Index nclusters,
                         typename M::Scalar epsilon,
                         const Eigen::Ref<const M> starting_clusters) {
    auto cols = points.cols();
    auto rows = points.rows();
    M clusters(starting_clusters);
    M temp_clusters(nclusters, cols);
    Eigen::Matrix<typename M::Index, Eigen::Dynamic, 1> counts(nclusters);
    Eigen::Matrix<typename M::Index, Eigen::Dynamic, 1> labels(rows);
    typename M::Scalar error = 0.0;
    typename M::Scalar old_error = 0.0;

    do {
        old_error = error;
        error = 0.0;
        counts.setZero();
        temp_clusters.setZero();

        for (typename M::Index i = 0; i < rows; ++i) {
            typename M::Scalar min_distance =
                std::numeric_limits<typename M::Scalar>::max();
            for (typename M::Index j = 0; j < nclusters; ++j) {
                typename M::Scalar distance =
                    (points.row(i) - clusters.row(j)).array().pow(2).sum();
                if (distance < min_distance) {
                    labels[i] = j;
                    min_distance = distance;
                }
            }

            temp_clusters.row(labels[i]) += points.row(i);
            ++counts[labels[i]];
            error += min_distance;
        }

        for (typename M::Index j = 0; j < nclusters; ++j) {
            for (typename M::Index k = 0; k < cols; ++k) {
                clusters(j, k) = counts[j] ? temp_clusters(j, k) / counts[j]
                                           : temp_clusters(j, k);
            }
        }
    } while (std::abs(error - old_error) > epsilon);

    typename M::Scalar max_radius =
        std::numeric_limits<typename M::Scalar>::min();
    V radii =
        V::Constant(nclusters, std::numeric_limits<typename M::Scalar>::min());
    for (typename M::Index i = 0; i < rows; ++i) {
        typename M::Scalar distance = std::sqrt(
            (points.row(i) - clusters.row(labels[i])).array().pow(2).sum());
        if (distance > radii[labels[i]]) {
            radii[labels[i]] = distance;
        }
        if (distance > max_radius) {
            max_radius = distance;
        }
    }

    return {max_radius, labels, clusters, counts, radii};
}
template Clustering<Matrix, Vector>
cluster(const Eigen::Ref<const Matrix> points, Matrix::Index nclusters,
        Matrix::Scalar epsilon,
        const Eigen::Ref<const Matrix> starting_clusters);
} // namespace fgt
