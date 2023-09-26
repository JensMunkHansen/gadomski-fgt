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

#include <cstddef>

#include "fgt.hpp"

namespace fgt {

// typedef Eigen::Matrix<Matrix::Index, Eigen::Dynamic, 1> VectorXs;

/// The results from k-means clustering.
template <typename M, typename V>
class Clustering {
public:
    /// The maximum cluster radius.
    typename M::Scalar max_radius;
    /// The cluster membership ids for each points.
    Eigen::Matrix<typename M::Index, Eigen::Dynamic, 1> indices;
    /// The centers of each cluster.
    M clusters;
    /// The number of points in each cluster.
    Eigen::Matrix<typename M::Index, Eigen::Dynamic, 1> npoints;
    /// The radius of each cluster.
    V radii;
};

/// Runs k-means clustering on a set of points.
template <typename M, typename V>
Clustering<M, V> cluster(const Eigen::Ref<const M> points,
                         typename M::Index nclusters,
                         typename M::Scalar epsilon);

/// Runs k-means clustering, specifying the starting cluster centers.
template <typename M, typename V>
Clustering<M, V> cluster(const Eigen::Ref<const M> points,
                         typename M::Index nclusters,
                         typename M::Scalar epsilon,
                         const Eigen::Ref<const M> starting_clusters);
} // namespace fgt
