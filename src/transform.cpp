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

#include "fgt.hpp"

namespace fgt {

template <typename M, typename V>
Transform<M, V>::Transform(const MatrixRef source, typename M::Scalar bandwidth)
    : m_source(source), m_bandwidth(bandwidth) {}

template <typename M, typename V>
V Transform<M, V>::compute(const MatrixRef target) {
    V weights = V::Ones(this->source().rows());
    return compute(target, weights);
}

template <typename M, typename V>
V Transform<M, V>::compute(const MatrixRef target, const VectorRef weights) {
    return compute_impl(target, weights);
}

// Explicit instantiations
template class Transform<Matrix, Vector>;
} // namespace fgt
