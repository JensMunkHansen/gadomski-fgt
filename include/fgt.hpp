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

/// \file fgt.hpp
/// \brief The header file for the fgt library.
///
/// This header includes both a functional interface and a class-based interface
/// to the fgt library.

#pragma once

#include <cstddef>
#include <memory>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4365) // unsigned/signed
#pragma warning(disable : 5031) // pragma warning mismatch
#pragma warning(disable : 4820) // padding
#pragma warning(disable : 4626) // assignment operator deleted
#pragma warning(disable : 5027) // move assignment operator deleted
#endif
#include <Eigen/Core>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

/// Top-level namespace for all things fgt.
namespace fgt {

/// fgt exceptions.
class fgt_error : public std::runtime_error {
public:
    explicit fgt_error(const std::string what_arg)
        : std::runtime_error(what_arg) {}
};

/// Thrown when an IFGT run asks for no clusters.
///
/// This usually means the data parameters aren't set up well for IFGT, for
/// whatever reason.
class ifgt_no_clusters : public fgt_error {
public:
    ifgt_no_clusters()
        : fgt_error("IFGT decided that it didn't need any clusters. These "
                    "parameters cannot be used for IFGT, try another method "
                    "instead.") {}
};
/// Convenience typedef for our type of matrix.
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Matrix;

/// Convenience typedef for our flavor of Eigen::Vector.
typedef Eigen::VectorXd Vector;

#if 0
  
/// Convenience typedef for a reference to our type of Eigen::Matrix.
///
/// We accept references to our functions in case the argument isn't exactly a
/// Eigen::Matrix, for whatever reason.
typedef Eigen::Ref<const Matrix> MatrixRef;


/// Convenience typedef for a reference to a Vector.
typedef Eigen::Ref<const Vector> VectorRef;
#endif

/// Returns the version of the fgt library as a string.
const char* version();

/// Returns true if the library was compiled with OpenMP support.
bool with_openmp();

/// Computes the direct Gauss transform with equal weights.
template <typename M, typename V>
V direct(const Eigen::Ref<const M> source, const Eigen::Ref<const M> target,
         typename M::Scalar bandwidth);

/// Computes the direct Gauss transform with provided weights.
template <typename M, typename V>
V direct(const Eigen::Ref<const M> source, const Eigen::Ref<const M> target,
         typename M::Scalar bandwidth, const Eigen::Ref<const V> weights);

/// Computes the direct Gauss transform using a kd-tree.
template <typename M, typename V>
V direct_tree(const Eigen::Ref<const M> source,
              const Eigen::Ref<const M> target, typename M::Scalar bandwidth,
              typename M::Scalar epsilon);

/// Computes the direct Gauss transform using a kd-tree and weights.
template <typename M, typename V>
V direct_tree(const Eigen::Ref<const M> source,
              const Eigen::Ref<const M> target, typename M::Scalar bandwidth,
              typename M::Scalar epsilon, const Eigen::Ref<const V> weights);

/// Computes the Improved Fast Gauss Transform.
template <typename M, typename V>
V ifgt(const Eigen::Ref<const M> source, const Eigen::Ref<const M> target,
       typename M::Scalar bandwidth, typename M::Scalar epsilon);

/// Computes the Improved Fast Gauss Transform with the provided weights.
template <typename M, typename V>
V ifgt(const Eigen::Ref<const M> source, const Eigen::Ref<const M> target,
       typename M::Scalar bandwidth, typename M::Scalar epsilon,
       const Eigen::Ref<const V> weights);

/// Abstract base class for all supported variants of the Gauss transform.
///
/// Some flavors of transform can pre-compute some data, e.g. the `DirectTree`
/// can pre-compute the KD-tree.
/// This pre-computation allows reuse of those data structure for multiple runs
/// of the transform, potentially with different target data sets.
template <typename M, typename V>
class Transform {
public:
    using MatrixRef = Eigen::Ref<const M>;
    using VectorRef = Eigen::Ref<const V>;
    /// Constructs a new transform that can be re-used with different targets.
    Transform(const Eigen::Ref<const M> source, typename M::Scalar bandwidth);

    /// Explicitly deleted copy constructor.
    Transform(const Transform&) = delete;
    /// Explicitly deleted assgnment operator.
    auto& operator=(const Transform&) = delete;
    /// Explicitly deleted move assignment operator.
    auto& operator=(Transform&&) = delete;

    /// Destroys a transform.
    virtual ~Transform() {}

    /// Returns the pointer to the source dataset.
    const Eigen::Ref<const M> source() const { return m_source; }
    /// Returns the bandwidth of the transform.
    typename M::Scalar bandwidth() const { return m_bandwidth; }

    /// Computes the Gauss transform for the given target dataset.
    V compute(const Eigen::Ref<const M> target);
    /// Computes the Gauss transform with the given weights.
    V compute(const Eigen::Ref<const M> target,
              const Eigen::Ref<const V> weights);

private:
    virtual V compute_impl(const Eigen::Ref<const M> target,
                           const Eigen::Ref<const V> weights) const = 0;

    const M m_source;
    typename M::Scalar m_bandwidth;
};

/// Direct Gauss transform.
template <typename M, typename V>
class Direct : public Transform<M, V> {
public:
    using MatrixRef = Eigen::Ref<const M>;
    using VectorRef = Eigen::Ref<const V>;
    /// Creates a new direct transform.
    Direct(const Eigen::Ref<const M> source, typename M::Scalar bandwidth);

    /// Explicitly deleted copy constructor.
    Direct(const Direct&) = delete;
    /// Explicitly deleted assgnment operator.
    auto& operator=(const Direct&) = delete;
    /// Explicitly deleted move assignment operator.
    auto& operator=(Direct&&) = delete;

private:
    virtual V compute_impl(const Eigen::Ref<const M> target,
                           const Eigen::Ref<const V> weights) const;
};

/// Direct Gauss transform using a KD-tree truncation.
template <typename M, typename V>
class DirectTree : public Transform<M, V> {
public:
    using MatrixRef = Eigen::Ref<const M>;
    using VectorRef = Eigen::Ref<const V>;
    /// Creates a new direct tree transform.
    ///
    /// This constructor pre-computes the KD-tree, so subsequent calls to
    /// `compute()` will re-use the same tree.
    DirectTree(const Eigen::Ref<const M> source, typename M::Scalar bandwidth,
               typename M::Scalar epsilon);

    /// Explicitly deleted copy constructor.
    DirectTree(const DirectTree&) = delete;
    /// Explicitly deleted assgnment operator.
    auto& operator=(const DirectTree&) = delete;
    /// Explicitly deleted move assignment operator.
    auto& operator=(DirectTree&&) = delete;

    /// Destroys a DirectTree.
    ///
    /// Required because of the unique pointer to a incomplete class.
    virtual ~DirectTree();

    /// Returns the error tolerance value.
    typename M::Scalar epsilon() const { return m_epsilon; }

private:
    struct NanoflannTree;

    virtual V compute_impl(const Eigen::Ref<const M> target,
                           const Eigen::Ref<const V> weights) const;

    typename M::Scalar m_epsilon;
    std::unique_ptr<NanoflannTree> m_tree;
};

/// Opaque k-means clustering structure.
template <typename M, typename V>
class Clustering;

/// Improved Fast Gauss Transform.
template <typename M, typename V>
class Ifgt : public Transform<M, V> {
public:
    using MatrixRef = Eigen::Ref<const M>;
    using VectorRef = Eigen::Ref<const V>;
    /// Creates a new Ifgt.
    ///
    /// This constructor will precompute some values, including the clusters and
    /// monomials, in hopes of speeding up subsequent runs.
    Ifgt(const Eigen::Ref<const M> source, typename M::Scalar bandwidth,
         typename M::Scalar epsilon);

    /// Explicitly deleted copy constructor.
    Ifgt(const Ifgt&) = delete;
    /// Explicitly deleted assgnment operator.
    auto& operator=(const Ifgt&) = delete;
    /// Explicitly deleted move assignment operator.
    auto& operator=(Ifgt&&) = delete;

    /// Destroys this transform.
    ///
    /// Required because PIMPL.
    virtual ~Ifgt();

    /// Returns the error tolerance value.
    typename M::Scalar epsilon() const { return m_epsilon; }
    /// Returns the number of clusters.
    typename M::Index nclusters() const { return m_nclusters; }
    /// Returns the truncation number.
    typename M::Index truncation_number() const { return m_truncation_number; }
    /// Returns the length of each monomial.
    typename M::Index p_max_total() const { return m_p_max_total; }

private:
    virtual V compute_impl(const Eigen::Ref<const M> target,
                           const Eigen::Ref<const V> weights) const;
    V compute_monomials(const Eigen::Ref<const V> d) const;
    V compute_constant_series() const;

    typename M::Scalar m_epsilon;
    typename M::Index m_nclusters;
    std::unique_ptr<Clustering<M, V>> m_clustering;
    typename M::Index m_truncation_number;
    typename M::Index m_p_max_total;
    V m_constant_series;
    V m_ry_square;
};
} // namespace fgt
