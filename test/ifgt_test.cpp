#include "gtest/gtest.h"

#include "ifgt.hpp"
#include "test/support.hpp"

namespace fgt {

TEST(Ifgt, Reference) {
    auto source = load_ascii_test_matrix<Matrix, Vector>("X.txt");
    auto target = load_ascii_test_matrix<Matrix, Vector>("Y.txt");
    double bandwidth = 0.5;
    double epsilon = 1e-4;
    auto expected = direct<Matrix, Vector>(source, target, bandwidth);
    auto actual = ifgt<Matrix, Vector>(source, target, bandwidth, epsilon);
    ASSERT_EQ(expected.size(), actual.size());
    EXPECT_LT((expected - actual).array().abs().maxCoeff() / actual.size(),
              epsilon);
}

TEST(Ifgt, ClassBased) {
    auto source = load_ascii_test_matrix<Matrix, Vector>("X.txt");
    auto target = load_ascii_test_matrix<Matrix, Vector>("Y.txt");
    double bandwidth = 0.5;
    double epsilon = 1e-4;
    auto expected = direct<Matrix, Vector>(source, target, bandwidth);
    auto actual =
        Ifgt<Matrix, Vector>(source, bandwidth, epsilon).compute(target);
    ASSERT_EQ(expected.size(), actual.size());
    EXPECT_LT((expected - actual).array().abs().maxCoeff() / actual.size(),
              epsilon);
}

TEST(Ifgt, ChooseParameters) {
    IfgtParameters params =
        ifgt_choose_parameters<Matrix, Vector>(2, 0.3, 1e-6, 189, 200);
    EXPECT_EQ(13, params.nclusters);
    EXPECT_NEAR(1.1151, params.cutoff_radius, 1e-4);
}

TEST(Ifgt, ChooseTruncationNumber) {
    size_t truncation_number =
        ifgt_choose_truncation_number<Matrix, Vector>(2, 0.3, 1e-6, 0.1, 200);
    EXPECT_EQ(9, truncation_number);
}

TEST(Ifgt, HighBandwidth) {
    auto source = load_ascii_test_matrix<Matrix, Vector>("X.txt");
    auto target = load_ascii_test_matrix<Matrix, Vector>("Y.txt");
    target = target.array() + 2;
    double bandwidth = 3.5;
    double epsilon = 1e-4;
    auto expected = direct<Matrix, Vector>(source, target, bandwidth);
    auto actual = ifgt<Matrix, Vector>(source, target, bandwidth, epsilon);
    ASSERT_EQ(expected.size(), actual.size());
    EXPECT_LT((expected - actual).array().abs().maxCoeff() / actual.size(),
              epsilon);
}

class WhatMismatch : public std::runtime_error {
public:
    WhatMismatch(const std::string& expectedWhat, const std::exception& e)
        : std::runtime_error(std::string("expected: '") + expectedWhat +
                             "', actual: '" + e.what() + '\'') {}
};
template <typename F>
auto call(const F& f, const std::string& expectedWhat) {
    try {
        return f();
    } catch (const std::exception& e) {
        if (expectedWhat != e.what())
            throw WhatMismatch(expectedWhat, e);
        throw;
    }
}

TEST(Ifgt, UTM) {
    auto source = load_ascii_test_matrix<Matrix, Vector>("utm.txt");
    auto target = source;
    EXPECT_THROW(call([&] { ifgt<Matrix, Vector>(source, target, 100, 1e-4); },
                      "IFGT decided that it didn't need any clusters. These "
                      "parameters cannot be used for IFGT, try another method "
                      "instead."),
                 fgt::ifgt_no_clusters);
}

TEST(Ifgt, ManyDimensionsManyPoints) {
    Matrix source = Matrix::Random(10, 60);
    EXPECT_THROW(call([&] { Ifgt<Matrix, Vector>(source, 0.4, 1e-4); },
                      "n choose k for 122 and 60 caused an overflow. "
                      "Dimensionality of the data might be too high."),
                 fgt::fgt_error);
}
} // namespace fgt
