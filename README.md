# UVA-Padova Glucose Control System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Status: Experimental](https://img.shields.io/badge/status-experimental-orange.svg)](docs/STATUS.md) [![Build](https://github.com/1paulpo1/uva-padova-glucose-control-system/actions/workflows/ci.yml/badge.svg)](https://github.com/1paulpo1/uva-padova-glucose-control-system/actions)

> **Research prototype** for personalized glucose prediction from the UVA-Padova model (Dalla Man 2007/2013). The project studies **identifiability by a single CGM channel** — the hard part of “dinner + insulin + CGM” at home. Offline identification (CMA-ES → LM) + Gates A/B/C is validated; online CKF and MPC are **experimental / future work** (see [STATUS](docs/STATUS.md)).

*Spec (RU, full math): [docs/main.md](docs/main.md) · Hybrid gut extension draft: [docs/new.md](docs/new.md) · Terminology fix: [docs/TERMINOLOGY.md](docs/TERMINOLOGY.md)*

---

## 30-second demo (no real data needed)

```bash
# 1) Build
cmake --preset release  # or: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build --output-on-failure

# 2) Synthetic recovery — 3 virtual patients, 1 day, ~60s
./build/synthetic_study/cross_validate --patients 3 --seed 42 --evals 200 --noise-preset cgm --json /tmp/synth.json
python3 synthetic_study/analyze_results.py --json /tmp/synth.json --out /tmp/figs --no-plots
cat /tmp/synth.json | python3 -m json.tool | head -n 50

# 3) Real-world validation — needs T1DEXI/OhioT1DM parquet (not in git)
export UVA_PADOVA_PARQUET=~/Downloads/metabonet_public.parquet
python3 real_world_validation/run_validation.py --study Loop --max-patients 5 --evals 200 --verbose
```

---

## What’s solid vs what’s not

| Area | Code | Validated |
|---|---|---|
| 14-state T1D model (`types.hpp:56`, `odesystem.hpp:20`) | ✅ | ✅ unit tests |
| Structural identifiability (`real_world_validation/identifiability/`) | ✅ | ✅ Julia + CSVs |
| Offline ID CMA-ES→LM, SoftL1, sandwich (`cost_function.hpp:97`, `optimizer_impl.hpp:50`) | ✅ | ✅ 300-patient sweeps (archived locally, see below) |
| Gates A/B/C (`diagnostics.cpp`) | ✅ | ✅ |
| CKF / RTS / meal-priors (`cubature_kalman_filter.hpp:60`, `online_adaptation.hpp`) | ✅ compiles | ⚠️ experimental, no real-data loop test |
| MPC | — | 🔜 Future Work |

> **No medical advice.** Not a device, not for dosing decisions.

---

## Project structure

```
include/uva_padova/          # core model + identification + online (header-only online)
src/                         # types, odesystem, integrals, diagnostics, optimizer
config/                      # parameter descriptors (5-active default, 9-active benchmark, scenarios)
real_world_validation/       # real CGM (T1DEXI/OhioT1DM) — offline ID + Gates A/B/C
  parsers/                   # parquet → CSV (pyarrow)
  identifiability/           # StructuralIdentifiability.jl studies
synthetic_study/             # virtual patients — recovery/observability study
  virtual_patient.{hpp,cpp}  # perturbation + error models
  cross_validate.cpp         # sweep runner
  online_evaluate.cpp        # CKF horizon study (experimental)
tests/                       # unit / integration / recovery
docs/main.md                 # full spec (RU)
docs/STATUS.md               # honest scope
docs/TERMINOLOGY.md          # validation vs synthetic naming fix
```

**Terminology fix (2026-09):** old `validation/` → `real_world_validation/` (real CGM), old `crossval/` → `synthetic_study/` (virtual patients). See [docs/TERMINOLOGY.md](docs/TERMINOLOGY.md).

---

## Achievements (from archived sweeps — not in git, kept locally)

Large sweeps (5–18M JSON per scenario, hours to recompute) are `.gitignore`’d but archived at `~/Documents/uva-padova-archive-local-backup/synthetic_study/` (and `/tmp/uva-padova-archive-backup/`). Use them to render the portfolio section without recomputing:

- **300 patients × 3 days, 5 restarts, preset C (irregular meals)** — median `|θ_err|` per param, Gate A pass rate, prior-dominated ranking. Render with `python3 synthetic_study/analyze_results.py --json ~/Documents/uva-padova-archive-local-backup/synthetic_study/crossval_results_1d.json --out /tmp/portfolio_figs`
- Key finding: 4-active config (`VG, kabs, kp1, kp2`) recovers within 0.15 log-space at 80% Gate A pass; 9-active degrades — matches structural scaling group.
- **Showcase**: `examples/complex_irregular_patient/` — one deliberately hard virtual patient (irregular meals + correction boluses, 16 params, 25 restarts). Run `./examples/complex_irregular_patient/run.sh` (a few minutes); outputs stay local (git-ignored). Its `true_vs_recovered.png` panels report per-param `|err|` with axes never tighter than ±0.25 log-units, so closeness to `y=x` reads against a fixed scale. Honest expectation: pipeline succeeds end-to-end but Gates A/B typically fail on this 16-param/1-day case — the example documents the failure mode, not a victory lap.

Run artifacts (JSON/CSV/PNG) are never committed — only scripts, configs, and docs.

---

## Reproducibility

- **Dependencies:** Eigen3, Ceres, libcmaes, nlohmann/json, GTest, Boost (Ljung-Box). No hard-coded `/opt/homebrew` — uses `CMAKE_PREFIX_PATH` / `find_package` (fix 2026-09).
- **Python:** `pyarrow`, `pandas`, `numpy`, `matplotlib`, `scikit-learn` (optional for PCA/t-SNE). `python3 -m pip install -r real_world_validation/requirements.txt` (to be added) or `conda`.
- **Data:** real parquet **not in git**. Set `UVA_PADOVA_PARQUET` to your T1DEXI/OhioT1DM download. Synthetic needs no data.
- **CI:** `.github/workflows/ci.yml` builds `ubuntu-latest` + `macos-14`, runs `ctest`, smoke synthetic run.

---

## Roadmap

- [ ] MPC with safety constraints + uncertainty-aware cost (Future Work)
- [ ] Heteroskedastic weights `w_k` in `ObservationCost` (FIXME #9)
- [ ] Real-data closed-loop online validation + on-device profiling
- [ ] `git filter-repo` history purge of large JSON before `v0.1.0` tag (local backups kept)

---

## Citation

If you use this code, please cite Dalla Man et al. 2007/2014/2018 and link to this repo. A `CITATION.cff` will be added.

## License

MIT — see [LICENSE](LICENSE).
