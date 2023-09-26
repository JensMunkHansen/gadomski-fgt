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
#include <vector>

#include "cluster.hpp"
#include "fgt.hpp"
#include "ifgt.hpp"

namespace fgt {
namespace {

// TODO make this configurable
const size_t TRUNCATION_NUMBER_UL = 200;

template <typename M, typename V>
typename M::Index nchoosek(typename M::Index n, typename M::Index k) {
    auto k_orig = k;
    typename M::Index n_k = n - k;
    if (k < n_k) {
        k = n_k;
        n_k = n - k;
    }
    typename M::Scalar nchsk = 1;
    for (typename M::Index i = 1; i <= n_k; ++i) {
        nchsk *= ++k;
        nchsk /= i;
    }
    if (nchsk > std::numeric_limits<typename M::Index>::max()) {
        std::stringstream ss;
        ss << "n choose k for " << n << " and " << k_orig
           << " caused an overflow. Dimensionality of the data might be "
              "too high.";
        throw fgt_error(ss.str());
    }
    return typename M::Index(nchsk);
}
} // namespace

template <typename M, typename V>
V ifgt(const Eigen::Ref<const M> source, const Eigen::Ref<const M> target,
       typename M::Scalar bandwidth, typename M::Scalar epsilon) {
    return Ifgt<M, V>(source, bandwidth, epsilon).compute(target);
}

template <typename M, typename V>
V ifgt(const Eigen::Ref<const M> source, const Eigen::Ref<const M> target,
       typename M::Scalar bandwidth, typename M::Scalar epsilon,
       const Eigen::Ref<const V> weights) {
    return Ifgt<M, V>(source, bandwidth, epsilon).compute(target, weights);
}

template <typename M, typename V>
IfgtParameters<M, V>
ifgt_choose_parameters(typename M::Index cols, typename M::Scalar bandwidth,
                       typename M::Scalar epsilon,
                       typename M::Index max_num_clusters,
                       typename M::Index truncation_number_ul) {
    typename M::Scalar h2 = bandwidth * bandwidth;
    typename M::Scalar radius = bandwidth * std::sqrt(std::log(1.0 / epsilon));
    typename M::Scalar complexity_min =
        std::numeric_limits<typename M::Scalar>::max();
    typename M::Index nclusters = 0;

    for (typename M::Index i = 0; i < max_num_clusters; ++i) {
        typename M::Scalar rx = std::pow(typename M::Scalar(i + 1),
                                         -1.0 / typename M::Scalar(cols));
        typename M::Scalar rx2 = rx * rx;
        typename M::Scalar n =
            std::min(typename M::Scalar(i + 1),
                     std::pow(radius / rx, typename M::Scalar(cols)));
        typename M::Scalar error =
            std::numeric_limits<typename M::Scalar>::max();
        typename M::Scalar temp = 1.0;
        typename M::Index p = 0;
        while ((error > epsilon) && (p <= truncation_number_ul)) {
            ++p;
            typename M::Scalar b = std::min(
                (rx + std::sqrt(rx2 + 2.0 * typename M::Scalar(p) * h2)) / 2.0,
                rx + radius);
            typename M::Scalar c = rx - b;
            temp *= 2 * rx * b / h2 / typename M::Scalar(p);
            error = temp * std::exp(-(c * c) / h2);
        }
        typename M::Scalar complexity =
            i + 1 + std::log(typename M::Scalar(i + 1)) +
            (n + 1) * nchoosek<M, V>(p - 1 + cols, cols);
        if (complexity < complexity_min) {
            complexity_min = complexity;
            nclusters = i + 1;
        }
    }
    return {nclusters, radius};
}

template <typename M, typename V>
typename M::Index
ifgt_choose_truncation_number(typename M::Index cols,
                              typename M::Scalar bandwidth,
                              typename M::Scalar epsilon, typename M::Scalar rx,
                              typename M::Index truncation_number_ul) {
    typename M::Scalar h2 = bandwidth * bandwidth;
    typename M::Scalar rx2 = rx * rx;
    typename M::Scalar r = std::min(
        std::sqrt(cols), bandwidth * std::sqrt(std::log(1.0 / epsilon)));
    typename M::Scalar error = std::numeric_limits<typename M::Scalar>::max();
    typename M::Index p = 0;
    typename M::Scalar temp = 1.0;
    while ((error > epsilon) && (p <= truncation_number_ul)) {
        ++p;
        typename M::Scalar b = std::min(
            (rx + std::sqrt(rx2 + 2 * typename M::Scalar(p) * h2)) / 2.0,
            rx + r);
        typename M::Scalar c = rx - b;
        temp *= 2 * rx * b / h2 / typename M::Scalar(p);
        error = temp * std::exp(-(c * c) / h2);
    }
    return p;
}

template <typename M, typename V>
Ifgt<M, V>::Ifgt(const MatrixRef source, typename M::Scalar bandwidth,
                 typename M::Scalar epsilon)
    : Transform<M, V>(source, bandwidth),
      m_epsilon(epsilon),
      m_nclusters(0),
      m_clustering(),
      m_truncation_number(0),
      m_p_max_total(0),
      m_constant_series() {
    // TODO max num clusters should be configurable
    typename M::Index max_num_clusters(
        typename M::Index(std::round(0.2 * 100 / bandwidth)));
    auto params =
        ifgt_choose_parameters<M, V>(source.cols(), bandwidth, epsilon,
                                     max_num_clusters, TRUNCATION_NUMBER_UL);
    if (params.nclusters == 0) {
        throw ifgt_no_clusters();
    }
    m_nclusters = params.nclusters;
    // TODO make the clustering constructor do the work?
    m_clustering.reset(
        new Clustering<M, V>(cluster<M, V>(source, m_nclusters, epsilon)));
    m_truncation_number = ifgt_choose_truncation_number<M, V>(
        source.cols(), bandwidth, epsilon, m_clustering->max_radius,
        TRUNCATION_NUMBER_UL);
    m_p_max_total =
        nchoosek<M, V>(m_truncation_number - 1 + source.cols(), source.cols());
    m_constant_series = compute_constant_series();
    m_ry_square.resize(m_nclusters);
    for (typename M::Index j = 0; j < m_nclusters; ++j) {
        typename M::Scalar ry = params.cutoff_radius + m_clustering->radii[j];
        m_ry_square[j] = ry * ry;
    }
}

template <typename M, typename V>
Ifgt<M, V>::~Ifgt() {}

template <typename M, typename V>
V Ifgt<M, V>::compute_monomials(const VectorRef d) const {
    auto cols = this->source().cols();
    std::vector<typename M::Index> heads(unsigned(cols), 0);
    V monomials = V::Ones(p_max_total());
    for (typename M::Index k = 1, t = 1, tail = 1; k < m_truncation_number;
         ++k, tail = t) {
        for (typename M::Index i = 0; i < cols; ++i) {
            typename M::Index head = heads[unsigned(i)];
            heads[unsigned(i)] = t;
            for (typename M::Index j = head; j < tail; ++j, ++t) {
                monomials[t] = d[i] * monomials[j];
            }
        }
    }
    return monomials;
}

template <typename M, typename V>
V Ifgt<M, V>::compute_constant_series() const {
    auto cols = this->source().cols();
    std::vector<typename M::Index> heads(unsigned(cols + 1), 0);
    heads[unsigned(cols)] = std::numeric_limits<typename M::Index>::max();
    std::vector<typename M::Index> cinds(unsigned(p_max_total()), 0);
    V monomials = V::Ones(p_max_total());

    for (typename M::Index k = 1, t = 1, tail = 1; k < m_truncation_number;
         ++k, tail = t) {
        for (typename M::Index i = 0; i < cols; ++i) {
            typename M::Index head = heads[unsigned(i)];
            heads[unsigned(i)] = t;
            for (typename M::Index j = head; j < tail; ++j, ++t) {
                cinds[unsigned(t)] =
                    (j < heads[unsigned(i) + 1]) ? cinds[unsigned(j)] + 1 : 1;
                monomials[t] = 2.0 * monomials[j];
                monomials[t] /= typename M::Scalar(cinds[unsigned(t)]);
            }
        }
    }
    return monomials;
}

template <typename M, typename V>
V Ifgt<M, V>::compute_impl(const MatrixRef target,
                           const VectorRef weights) const {
    auto source = this->source();
    auto rows_source = source.rows();
    auto rows_target = target.rows();
    auto cols = source.cols();
    auto bandwidth = this->bandwidth();
    auto nclusters = this->nclusters();
    auto p_max_total = this->p_max_total();

    typename M::Scalar h2 = bandwidth * bandwidth;

    M C = M::Zero(nclusters, p_max_total);
    for (typename M::Index i = 0; i < rows_source; ++i) {
        typename M::Scalar distance = 0.0;
        V dx = V::Zero(cols);
        for (typename M::Index k = 0; k < cols; ++k) {
            typename M::Scalar delta =
                source(i, k) -
                m_clustering->clusters(m_clustering->indices[i], k);
            distance += delta * delta;
            dx[k] = delta / bandwidth;
        }

        auto monomials = compute_monomials(dx);
        typename M::Scalar f = weights[i] * std::exp(-distance / h2);
        for (typename M::Index alpha = 0; alpha < p_max_total; ++alpha) {
            C(m_clustering->indices[i], alpha) += f * monomials[alpha];
        }
    }

#pragma omp parallel for
    for (typename M::Index j = 0; j < nclusters; ++j) {
        for (typename M::Index alpha = 0; alpha < p_max_total; ++alpha) {
            C(j, alpha) *= m_constant_series[alpha];
        }
    }

    V G = V::Zero(rows_target);
#pragma omp parallel for
    for (typename M::Index i = 0; i < rows_target; ++i) {
        for (typename M::Index j = 0; j < nclusters; ++j) {
            typename M::Scalar distance = 0.0;
            V dy = V::Zero(cols);
            for (typename M::Index k = 0; k < cols; ++k) {
                typename M::Scalar delta =
                    target(i, k) - m_clustering->clusters(j, k);
                distance += delta * delta;
                if (distance > m_ry_square[j]) {
                    break;
                }
                dy[k] = delta / bandwidth;
            }
            if (distance <= m_ry_square[j]) {
                auto monomials = compute_monomials(dy);
                typename M::Scalar g = std::exp(-distance / h2);
                G[i] += C.row(j) * g * monomials;
            }
        }
    }

    return G;
}

template Vector ifgt<Matrix, Vector>(const Eigen::Ref<const Matrix> source,
                                     const Eigen::Ref<const Matrix> target,
                                     Matrix::Scalar bandwidth,
                                     Matrix::Scalar epsilon);

template Vector ifgt<Matrix, Vector>(const Eigen::Ref<const Matrix> source,
                                     const Eigen::Ref<const Matrix> target,
                                     Matrix::Scalar bandwidth,
                                     Matrix::Scalar epsilon,
                                     const Eigen::Ref<const Vector> weights);

template Matrix::Index ifgt_choose_truncation_number<Matrix, Vector>(
    Matrix::Index cols, Matrix::Scalar bandwidth, Matrix::Scalar epsilon,
    Matrix::Scalar rx, Matrix::Index truncation_number_ul);
} // namespace fgt
