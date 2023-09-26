#include "gtest/gtest.h"

#include "test/support.hpp"

namespace fgt {

TEST(Direct, Reference) {
    auto source = load_ascii_test_matrix<Matrix, Vector>("X.txt");
    auto target = load_ascii_test_matrix<Matrix, Vector>("Y.txt");
    auto actual = direct<Matrix, Vector>(source, target, 0.5);
    auto expected = load_ascii_test_matrix<Matrix, Vector>("direct.txt");
    EXPECT_TRUE(expected.isApprox(actual, 1e-6));
}

TEST(Direct, WithWeights) {
    auto source = load_ascii_test_matrix<Matrix, Vector>("X.txt");
    auto target = load_ascii_test_matrix<Matrix, Vector>("Y.txt");
    auto no_weights = direct<Matrix, Vector>(source, target, 0.5);
    Vector weights = Vector::Ones(source.rows());
    auto with_weights = direct<Matrix, Vector>(source, target, 0.5, weights);
    EXPECT_TRUE(no_weights.isApprox(with_weights));
}

TEST(Direct, ClassBased) {
    auto source = load_ascii_test_matrix<Matrix, Vector>("X.txt");
    auto target = load_ascii_test_matrix<Matrix, Vector>("Y.txt");
    Direct<Matrix, Vector> direct(source, 0.5);
    auto actual = direct.compute(target);
    auto expected = load_ascii_test_matrix<Matrix, Vector>("direct.txt");
    EXPECT_TRUE(expected.isApprox(actual, 1e-6));
}
} // namespace fgt
