# Contributing

This is a research prototype. Issues and PRs are welcome.

## Quick dev loop

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build --output-on-failure
./build/synthetic_study/cross_validate --patients 3 --seed 42 --evals 200 --noise-preset cgm --json /tmp/s.json
```

## Conventions

- C++20, `clang-format` (not enforced yet), `-Wall -Wextra -Wpedantic` on core libs.
- Python: `black` + `ruff` preferred for `real_world_validation/` and `synthetic_study/`.
- Do not commit large JSON/CSV results — they are `.gitignore`’d. Keep a local archive (`~/Documents/uva-padova-archive-local-backup/`).
- Real patient parquet is not in git. Set `UVA_PADOVA_PARQUET`.

## Terminology

See `docs/TERMINOLOGY.md` — `real_world_validation/` = real CGM, `synthetic_study/` = virtual patients.
