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

#pragma once

#include <cstddef>

#include "fgt.hpp"

namespace fgt {

/// Tuning parameters for the IFGT.
///
/// In general, you shouldn't need to create these yourself.
template <typename M, typename V>
struct IfgtParameters {
    /// The number of clusters that should be used for the IFGT.
    typename M::Index nclusters;
    /// The cutoff radius.
    typename M::Scalar cutoff_radius;
};

/// Chooses appropriate parameters for an IFGT.
template <typename M, typename V>
IfgtParameters<M, V>
ifgt_choose_parameters(typename M::Index cols, typename M::Scalar bandwidth,
                       typename M::Scalar epsilon,
                       typename M::Index max_num_clusters,
                       typename M::Index truncation_number_ul);
/// Chooses the appropriate truncation number for IFGT, given a max clustering
/// radius.
template <typename M, typename V>
typename M::Index ifgt_choose_truncation_number(
    typename M::Index cols, typename M::Scalar bandwidth,
    typename M::Scalar epsilon, typename M::Scalar max_radius,
    typename M::Index truncation_number_ul);
} // namespace fgt
