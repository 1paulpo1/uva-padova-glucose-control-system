#include "uva_padova/integrals.hpp"
#include "uva_padova/odesystem.hpp"
#include "uva_padova/types.hpp"
#include "uva_padova/config_loader.hpp"
#include "virtual_patient.hpp"

#include <boost/numeric/odeint.hpp>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>

namespace odeint = boost::numeric::odeint;
using namespace uva_padova;
using namespace uva_padova::crossval;

// ---------------------------------------------------------------------------
// State array for odeint — 14 components matching BasicState order
// ---------------------------------------------------------------------------
using OdeState = std::array<double, 14>;

static OdeState to_ode(const State& s) {
    return {s.Gp, s.Gt, s.Qsto1, s.Qsto2, s.Qgut,
            s.Ip, s.Ii, s.Isc1, s.Isc2, s.I_prime,
            s.Xl, s.X, s.Gsc, s.mealD};
}

static State from_ode(const OdeState& a) {
    State s;
    s.Gp = a[0]; s.Gt = a[1]; s.Qsto1 = a[2]; s.Qsto2 = a[3]; s.Qgut = a[4];
    s.Ip = a[5]; s.Ii = a[6]; s.Isc1 = a[7]; s.Isc2 = a[8];
    s.I_prime = a[9]; s.Xl = a[10]; s.X = a[11]; s.Gsc = a[12]; s.mealD = a[13];
    return s;
}

// ---------------------------------------------------------------------------
// Odeint RHS functor — wraps UvaPadovaODESystem::rhs with fixed inputs & params
// ---------------------------------------------------------------------------
struct UvaPadovaOde {
    const Inputs& u_;
    const Params& p_;
    void operator()(const OdeState& x, OdeState& dxdt, double /*t*/) const {
        State sx = from_ode(x);
        State sdxdt;
        UvaPadovaODESystem::rhs(sx, sdxdt, u_, p_);
        dxdt = to_ode(sdxdt);
    }
};

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    // ---- 1. Config — identical to 9-active zero-noise experiment ----
    VirtualPatientConfig cfg = default_config();
    cfg.cgm_noise.fixed_sd = 0.0;
    cfg.seed = 42;
    cfg.params_config_path = CROSSVAL_OUTPUT_DIR "/config_9_active.json";

    // ---- 2. Generate virtual patient (RK4 h=1.0 baseline) ----
    std::cout << "Generating virtual patient (RK4 h=1.0)...\n";
    auto vp = generate_virtual_patient(cfg, 0);
    const auto& rk4_t = vp.true_data.cgm_time;
    const auto& rk4_G = vp.true_data.cgm_value;

    std::cout << "Patient generated: " << rk4_t.size() << " CGM samples, "
              << vp.active_desc.size() << " active params\n";

    // ---- 3. Steady state ----
    State ss;
    double u_basal = SteadyStateSolver::find(ss, vp.true_params, vp.true_params.Gb);
    Inputs u;
    u.insulin_rate = u_basal;

    // ---- 4. Build meal/bolus events from config (same expansion as virtual_patient.cpp) ----
    struct Event { int t; double cho_mg; double bolus_pmol_kg; };
    std::vector<Event> events;
    int n_days = cfg.duration_min / 1440 + 1;
    for (int day = 0; day < n_days; ++day) {
        int offset = day * 1440;
        for (const auto& dm : cfg.daily_meals) {
            int t = offset + dm.time_min;
            if (t > cfg.duration_min) break;
            events.push_back({t, dm.cho_mg, dm.bolus_dose_pmol_kg});
        }
    }

    // ---- 5. Odeint reference integration (Cash-Karp 54 adaptive, tol=1e-9) ----
    std::cout << "Integrating with odeint (Cash-Karp 54, tol=1e-9)...\n";

    OdeState ode_x = to_ode(ss);
    UvaPadovaOde rhs_functor{u, vp.true_params};
    auto stepper = odeint::make_controlled<odeint::runge_kutta_cash_karp54<OdeState>>(
        1.0e-9, 1.0e-9);

    std::vector<std::pair<double, double>> odeint_cgm;
    int event_idx = 0;

    for (int t = 0; t <= cfg.duration_min; ++t) {
        // Apply events at this minute
        while (event_idx < static_cast<int>(events.size()) && events[event_idx].t == t) {
            State s = from_ode(ode_x);
            s.Qsto1 += events[event_idx].cho_mg;
            s.mealD = events[event_idx].cho_mg;
            s.Isc1 += events[event_idx].bolus_pmol_kg;
            ode_x = to_ode(s);
            event_idx++;
        }

        // Record CGM at sample intervals
        if (t % cfg.cgm_interval == 0) {
            odeint_cgm.push_back({static_cast<double>(t), from_ode(ode_x).Gsc});
        }

        // Integrate one minute forward
        if (t < cfg.duration_min) {
            odeint::integrate_adaptive(stepper, rhs_functor, ode_x,
                                       static_cast<double>(t),
                                       static_cast<double>(t + 1),
                                       0.1);
        }
    }

    // ---- 6. Compare ----
    size_t n = std::min(rk4_t.size(), odeint_cgm.size());
    double max_diff = 0.0, max_diff_day = 0.0, max_diff_night = 0.0;
    double sum_abs_diff = 0.0;

    std::ofstream csv("rk4_vs_odeint_seed42.csv");
    csv << "t_min,G_rk4,G_odeint,diff\n";

    for (size_t i = 0; i < n; ++i) {
        double diff = rk4_G[i] - odeint_cgm[i].second;
        double abs_diff = std::abs(diff);
        sum_abs_diff += abs_diff;

        csv << rk4_t[i] << "," << rk4_G[i] << ","
            << odeint_cgm[i].second << "," << diff << "\n";

        if (abs_diff > max_diff) max_diff = abs_diff;

        int hour = static_cast<int>(rk4_t[i]) / 60 % 24;
        if (hour >= 7 && hour < 23) {
            if (abs_diff > max_diff_day) max_diff_day = abs_diff;
        } else {
            if (abs_diff > max_diff_night) max_diff_night = abs_diff;
        }
    }
    csv.close();

    // ---- 7. Print results ----
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n=== RK4 vs Odeint comparison ===\n";
    std::cout << "  Config: 9 active params, seed=42, zero noise, "
              << cfg.duration_min / 1440 << " days\n";
    std::cout << "  max |G_rk4 - G_odeint|: " << max_diff << " mg/dL\n";
    std::cout << "  max daytime (7-23):     " << max_diff_day << " mg/dL\n";
    std::cout << "  max nighttime (23-7):   " << max_diff_night << " mg/dL\n";
    std::cout << "  mean |diff|:            " << (sum_abs_diff / n) << " mg/dL\n";
    std::cout << "  N samples:              " << n << "\n";

    if (max_diff > 0.1) {
        std::cout << "\n  >>> RK4 ERROR EXCEEDS 0.1 mg/dL — "
                  << "likely cause of Gate B failure\n";
    } else if (max_diff > 0.01) {
        std::cout << "\n  >>> RK4 error is moderate (0.01–0.1 mg/dL) — "
                  << "may contribute to Gate B\n";
    } else {
        std::cout << "\n  >>> RK4 error is negligible (<0.01 mg/dL) — "
                  << "look elsewhere for Gate B cause\n";
    }

    // Print first few diffs to see the pattern
    std::cout << "\nFirst 10 CGM samples:\n";
    std::cout << "  t\tG_rk4\tG_odeint\tdiff\n";
    for (size_t i = 0; i < std::min(n, size_t(10)); ++i) {
        std::cout << "  " << rk4_t[i] << "\t" << rk4_G[i] << "\t"
                  << odeint_cgm[i].second << "\t"
                  << (rk4_G[i] - odeint_cgm[i].second) << "\n";
    }

    return 0;
}
