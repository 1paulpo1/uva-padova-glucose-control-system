#include <gtest/gtest.h>
#include <cmath>

#include "uva_padova/cost_function.hpp"
#include "mocks/mock_simulator.h"

using namespace uva_padova;

namespace {

// All 32 param names — tested in two batches (≤ kMaxJetDim = 30 each).
constexpr const char* const kParamBatchA[] = {
  "kp1", "kp2", "kp3", "Vm0", "Vmx", "p2U",
  "ka1", "ka2", "kd", "kmax", "kmin", "kabs", "kgri",
  "VG", "VI", "Gb"
};
constexpr const char* const kParamBatchB[] = {
  "Ib", "k1", "k2", "m1", "m2", "m4", "ki",
  "f", "b", "d",
  "ke1", "ke2", "Ts", "r1", "Gth", "BW"
};

std::vector<ParamDescriptor> make_descs(const char* const* names, int count) {
  std::vector<ParamDescriptor> descs;
  for (int i = 0; i < count; ++i) descs.push_back({names[i]});
  return descs;
}

void test_batch(const char* const* names, int count) {
  auto descs = make_descs(names, count);
  IdentificationData data;
  data.finalize();
  ObservationCost<MockSimulator> cost(data, Params::defaults_t1d(), descs);

  std::vector<double> theta(count, std::log(2.0));
  BasicParams<double> p = Params::defaults_t1d();
  auto orig = p;
  cost.apply_adjustments(theta.data(), p);

  for (int i = 0; i < count; ++i) {
    std::string n = names[i];
    auto check = [&](double val, double expected) {
      ASSERT_DOUBLE_EQ(val, expected) << " param=" << n;
    };
    if (n == "kp1")  check(p.kp1,  orig.kp1 * 2.0);
    if (n == "kp2")  check(p.kp2,  orig.kp2 * 2.0);
    if (n == "kp3")  check(p.kp3,  orig.kp3 * 2.0);
    if (n == "Vm0")  check(p.Vm0,  orig.Vm0 * 2.0);
    if (n == "Vmx")  check(p.Vmx,  orig.Vmx * 2.0);
    if (n == "p2U")  check(p.p2U,  orig.p2U * 2.0);
    if (n == "ka1")  check(p.ka1,  orig.ka1 * 2.0);
    if (n == "ka2")  check(p.ka2,  orig.ka2 * 2.0);
    if (n == "kd")   check(p.kd,   orig.kd  * 2.0);
    if (n == "kmax") check(p.kmax, orig.kmax * 2.0);
    if (n == "kmin") check(p.kmin, orig.kmin * 2.0);
    if (n == "kabs") check(p.kabs, orig.kabs * 2.0);
    if (n == "kgri") check(p.kgri, orig.kgri * 2.0);
    if (n == "VG")   check(p.VG,   orig.VG  * 2.0);
    if (n == "VI")   check(p.VI,   orig.VI  * 2.0);
    if (n == "Gb")   check(p.Gb,   orig.Gb  * 2.0);
    if (n == "Ib")   check(p.Ib,   orig.Ib  * 2.0);
    if (n == "k1")   check(p.k1,   orig.k1  * 2.0);
    if (n == "k2")   check(p.k2,   orig.k2  * 2.0);
    if (n == "m1")   check(p.m1,   orig.m1  * 2.0);
    if (n == "m2")   check(p.m2,   orig.m2  * 2.0);
    if (n == "m4")   check(p.m4,   orig.m4  * 2.0);
    if (n == "ki")   check(p.ki,   orig.ki  * 2.0);
    if (n == "f")    check(p.f,    orig.f   * 2.0);
    if (n == "b")    check(p.b,    orig.b   * 2.0);
    if (n == "d")    check(p.d,    orig.d   * 2.0);
    if (n == "ke1")  check(p.ke1,  orig.ke1 * 2.0);
    if (n == "ke2")  check(p.ke2,  orig.ke2 * 2.0);
    if (n == "Ts")   check(p.Ts,   orig.Ts  * 2.0);
    if (n == "r1")   check(p.r1,   orig.r1  * 2.0);
    if (n == "Gth")  check(p.Gth,  orig.Gth * 2.0);
    if (n == "BW")   check(p.BW,   orig.BW  * 2.0);
  }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
//  apply_adjustments: all params in two batches (kMaxJetDim = 30)
// ---------------------------------------------------------------------------
TEST(AdjustmentsTest, apply_adjustments_batch_A) {
  test_batch(kParamBatchA, sizeof(kParamBatchA) / sizeof(kParamBatchA[0]));
}

TEST(AdjustmentsTest, apply_adjustments_batch_B) {
  test_batch(kParamBatchB, sizeof(kParamBatchB) / sizeof(kParamBatchB[0]));
}

// ---------------------------------------------------------------------------
//  apply_adjustments: theta = 0 → multiplier = 1
// ---------------------------------------------------------------------------
TEST(AdjustmentsTest, apply_adjustments_theta_zero) {
  auto descs = make_descs(kParamBatchA, sizeof(kParamBatchA) / sizeof(kParamBatchA[0]));
  IdentificationData data;
  data.finalize();
  ObservationCost<MockSimulator> cost(data, Params::defaults_t1d(), descs);

  std::vector<double> theta(descs.size(), 0.0);

  BasicParams<double> p = Params::defaults_t1d();
  auto orig = p;
  cost.apply_adjustments(theta.data(), p);

  EXPECT_DOUBLE_EQ(p.kp1, orig.kp1);
  EXPECT_DOUBLE_EQ(p.VG,  orig.VG);
  EXPECT_DOUBLE_EQ(p.Gb,  orig.Gb);
  EXPECT_DOUBLE_EQ(p.BW,  orig.BW);  // BW not in batch A, verify unchanged
}

// ---------------------------------------------------------------------------
//  apply_adjustments: unlisted params unchanged
// ---------------------------------------------------------------------------
TEST(AdjustmentsTest, apply_adjustments_unlisted_unchanged) {
  // Only adjust kp1 and Vm0
  std::vector<ParamDescriptor> descs = {{"kp1"}, {"Vm0"}};
  IdentificationData data;
  data.finalize();
  ObservationCost<MockSimulator> cost(data, Params::defaults_t1d(), descs);

  std::vector<double> theta = {std::log(3.0), std::log(0.5)};

  BasicParams<double> p = Params::defaults_t1d();
  auto orig = p;
  cost.apply_adjustments(theta.data(), p);

  EXPECT_DOUBLE_EQ(p.kp1, orig.kp1 * 3.0);
  EXPECT_DOUBLE_EQ(p.Vm0, orig.Vm0 * 0.5);

  // All other params unchanged
  EXPECT_DOUBLE_EQ(p.VG,   orig.VG);
  EXPECT_DOUBLE_EQ(p.k1,   orig.k1);
  EXPECT_DOUBLE_EQ(p.k2,   orig.k2);
  EXPECT_DOUBLE_EQ(p.VI,   orig.VI);
  EXPECT_DOUBLE_EQ(p.Gb,   orig.Gb);
  EXPECT_DOUBLE_EQ(p.BW,   orig.BW);
}

// ---------------------------------------------------------------------------
//  to_double: Jet → double, derivatives dropped
// ---------------------------------------------------------------------------
TEST(AdjustmentsTest, to_double_conversion) {
  using JetT = ceres::Jet<double, 10>;
  BasicParams<JetT> p_jet;

  auto make_jet = [](double val, int idx, double deriv) {
    JetT j(val);
    j.v[idx] = deriv;
    return j;
  };
  p_jet.kp1 = make_jet(2.5, 0, 1.0);
  p_jet.VG  = make_jet(1.88, 1, 2.0);
  p_jet.Gb  = make_jet(120.0, 2, 3.0);

  auto p_double = ObservationCost<MockSimulator>::to_double(p_jet);

  EXPECT_DOUBLE_EQ(p_double.kp1, 2.5);
  EXPECT_DOUBLE_EQ(p_double.VG,  1.88);
  EXPECT_DOUBLE_EQ(p_double.Gb,  120.0);
}

// ---------------------------------------------------------------------------
//  copy_params: Params → BasicParams<T>
// ---------------------------------------------------------------------------
TEST(AdjustmentsTest, copy_params) {
  Params src;
  src.kp1 = 1.11;
  src.VG  = 2.22;
  src.Gb  = 99.0;

  BasicParams<double> dst;
  ObservationCost<MockSimulator>::copy_params(src, dst);

  EXPECT_DOUBLE_EQ(dst.kp1, 1.11);
  EXPECT_DOUBLE_EQ(dst.VG,  2.22);
  EXPECT_DOUBLE_EQ(dst.Gb,  99.0);
  EXPECT_DOUBLE_EQ(dst.k2,  src.k2);
  EXPECT_DOUBLE_EQ(dst.BW,  src.BW);
}
