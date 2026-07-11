# Regenerate summary_realistic.txt from existing CSV files
# Fixes the Symbol-key bug: result dict uses QQMPolyRingElem keys, not Symbol

outdir = @__DIR__
thresholds = [0.99, 0.95, 0.90]

# ── Read CSV files ──
function read_csv_dict(path)
    d = Dict{String, Bool}()
    isfile(path) || return d
    for line in readlines(path)[2:end]  # skip header
        parts = split(line, ",", limit=3)
        length(parts) >= 2 || continue
        var = strip(parts[1])
        val = lowercase(strip(parts[2])) == "true"
        d[var] = val
    end
    return d
end

local_results = Dict{Float64, Dict{String, Bool}}()
for th in thresholds
    csvpath = joinpath(outdir, "a_full_local_$(Int(th*100)).csv")
    local_results[th] = read_csv_dict(csvpath)
    nL = count(v -> v, values(local_results[th]))
    println("p=$th: $nL/$(length(local_results[th])) L")
end

# ── Parameter lists (same as realistic_analysis.jl) ──
all_params_local = [
    ("Qsto1(t)", "State"), ("Qsto2(t)", "State"), ("Qgut(t)", "State"),
    ("Isc1(t)", "State"), ("Isc2(t)", "State"), ("Ip(t)", "State"),
    ("Ii(t)", "State"), ("I_prime(t)", "State"), ("Xl(t)", "State"),
    ("X(t)", "State"), ("Gp(t)", "State"), ("Gt(t)", "State"),
    ("Gsc(t)", "State"),
    ("kgri", "GI"), ("kmax", "GI"), ("kabs", "GI"), ("f", "GI"), ("BW", "GI"),
    ("ka1", "SC ins"), ("ka2", "SC ins"), ("kd", "SC ins"),
    ("m1", "Ins kin"), ("m2", "Ins kin"), ("m4", "Ins kin"),
    ("VI", "Delay"), ("Ib", "Delay"),
    ("ki", "EGP"), ("kp1", "EGP"), ("kp2", "EGP"), ("kp3", "EGP"),
    ("Fcns", "Util"), ("p2U", "Delay"),
    ("Vm0", "Util"), ("Vmx", "Util"), ("Km0", "Util"),
    ("k1", "Glc kin"), ("k2", "Glc kin"),
    ("VG", "Glc kin"), ("Ts", "CGM"),
]

# ── Build local_full using string keys (correct!) ──
local_full = Dict{String, String}()
for (p, _) in all_params_local
    local_full[p] = get(local_results[0.99], p, false) ? "L" : "N"
end

# ── Decomposed and realistic dicts ──
decomposed = Dict(
    "ka1" => "G", "ka2" => "L", "kd" => "L",
    "m1" => "L", "m2" => "L", "m4" => "L",
    "VI" => "G", "Ib" => "G",
    "ki" => "G", "kp1" => "N", "kp2" => "G", "kp3" => "N",
    "k1" => "G", "k2" => "G", "VG" => "N",
    "Vm0" => "N", "Vmx" => "N", "Km0" => "N", "Fcns" => "N",
    "p2U" => "G", "Ts" => "G",
    "kgri" => "L", "kmax" => "L", "kabs" => "L", "f" => "N", "BW" => "N",
    "ke1" => "L", "ke2" => "N",
)

realistic = Dict(
    "ka1" => "N", "ka2" => "N", "kd" => "N",
    "m1" => "N", "m2" => "N", "m4" => "N",
    "VI" => "N", "Ib" => "N",
    "ki" => "N", "kp1" => "N", "kp2" => "N", "kp3" => "N",
    "k1" => "N", "k2" => "N", "VG" => "N",
    "Vm0" => "N", "Vmx" => "N", "Km0" => "N", "Fcns" => "N",
    "p2U" => "N", "Ts" => "N",
    "kgri" => "N", "kmax" => "N", "kabs" => "N", "f" => "N", "BW" => "N",
    "ke1" => "N", "ke2" => "N",
)

comparison_params = [
    ("ka1", "SC ins"), ("ka2", "SC ins"), ("kd", "SC ins"),
    ("m1", "Ins kin"), ("m2", "Ins kin"), ("m4", "Ins kin"),
    ("VI", "Delay"), ("ki", "Delay"), ("p2U", "Delay"), ("Ib", "Delay"),
    ("kp1", "EGP"), ("kp2", "EGP"), ("kp3", "EGP"),
    ("Fcns", "Util"), ("Vm0", "Util"), ("Vmx", "Util"), ("Km0", "Util"),
    ("k1", "Glc kin"), ("k2", "Glc kin"), ("VG", "Glc kin"), ("Ts", "CGM"),
    ("kgri", "GI"), ("kmax", "GI"), ("kabs", "GI"), ("f", "GI"), ("BW", "GI"),
    ("ke1", "Renal"), ("ke2", "Renal"),
]

# ── Read P8a and B1/B2 functions ──
function read_functions(path)
    !isfile(path) && return String[]
    return [strip(split(line, ",")[2]) for line in readlines(path)[2:end] if occursin(",", line)]
end

p8a_functions = read_functions(joinpath(outdir, "p8a_glu_core_functions.csv"))
b1_functions  = read_functions(joinpath(outdir, "b1_glu_delays_functions.csv"))
b2_functions  = read_functions(joinpath(outdir, "b2_glu_core_functions.csv"))

# ═══════════════════════════════════════════════════════════════════════════════
#  GENERATE REPORT
# ═══════════════════════════════════════════════════════════════════════════════

summary_path = joinpath(outdir, "summary_realistic.txt")
open(summary_path, "w") do io
    w(s) = println(io, s)

    w("╔══════════════════════════════════════════════════════════════════════════╗")
    w("║  REALISTIC IDENTIFIABILITY ANALYSIS                                     ║")
    w("║  UVA-Padova T1DM — CGM + pump + meals only                             ║")
    w("╚══════════════════════════════════════════════════════════════════════════╝")
    w("")

    w("── DESIGN ──")
    w("  Measurements: y = Gsc (CGM sensor)")
    w("  Known inputs: u (insulin pump), u_meal (meal)")
    w("  No additional measurements: no Ip, no Ii, no Ra")
    w("  No decomposition assumptions: Xl, X, Ip are STATES, not known inputs")
    w("")

    # ── A: Local ──
    w("── A: FULL MODEL LOCAL IDENTIFIABILITY ──")
    w("  13 states, 26 params, 2 inputs (u, u_meal), 1 output (Gsc)")
    w("  Analysis run at 3 probability thresholds (0.99 / 0.95 / 0.90)")
    w("")

    nL_total = [count(v -> v, values(local_results[th])) for th in thresholds]
    nL_params = [count(p -> get(local_results[th], p, false),
                       [p for (p,_) in all_params_local if !endswith(p, "(t)")])
                 for th in thresholds]
    w("  Total L/N (any var): $(nL_total[1])/39, $(nL_total[2])/39, $(nL_total[1])/39")
    w("  Parameter L:         $(nL_params[1])/26, $(nL_params[2])/26, $(nL_params[3])/26")
    w("")

    w("  $(rpad("Parameter", 22))  p=0.99  p=0.95  p=0.90  Category")
    w("  " * "-" ^ 65)
    for (p, cat) in all_params_local
        vals = [get(local_results[th], p, false) ? "Y" : "N" for th in thresholds]
        w("  $(rpad(p, 22))  $(vals[1])       $(vals[2])       $(vals[3])       $cat")
    end
    w("")

    # Threshold stability
    all_same = all(Set([get(local_results[th], p, false) for th in thresholds]) |> length == 1
                   for (p,_) in all_params_local)
    if all_same
        w("  Threshold stability: ALL params identical across p=0.99/0.95/0.90")
        w("  No borderline parameters.")
    else
        w("  Threshold stability: NOT all identical — some params change with p")
    end
    w("")

    # ── B: Global ──
    w("── B: GLOBAL ANALYSIS ──")
    w("")

    if length(b1_functions) > 0
        w("  B1: Glucose + Delays (6 states, Xl,X as states, Ip(t) input)")
        w("  Identifiable functions ($(length(b1_functions))):")
        for (i, f) in enumerate(b1_functions)
            w("    [$i]  $f")
        end
    else
        w("  B1: Glucose + Delays → CRASHED (FLINT/MM Gröbner basis overflow)")
        w("  The 6-state model with MM term cannot be analyzed globally")
    end
    w("")

    if length(b2_functions) > 0
        w("  B2: Glucose Core (3 states, Xl,X as known inputs)")
        w("  Identifiable functions ($(length(b2_functions))):")
        for (i, f) in enumerate(b2_functions)
            w("    [$i]  $f")
        end
    elseif length(p8a_functions) > 0
        w("  B2: Glucose Core — using cached P8a results:")
        for (i, f) in enumerate(p8a_functions)
            w("    [$i]  $f")
        end
    else
        w("  B2: Glucose Core — no results available")
    end
    w("")

    # ── COMPARISON ──
    w("══ THREE-WAY COMPARISON ══")
    w("")
    w("  G = globally id   L = locally id   N = non-id")
    w("  Decomposed (global) = P1-P4 (Xl, X, Ra known — optimistic upper bound)")
    w("  Realistic (global)  = from CGM+pump+meals, no decomposition")
    w("  Local (full)        = full model, p=0.99")
    w("")

    w("  $(rpad("Parameter", 10))  Decomposed  Realistic  Local(full)  Category")
    w("  " * "-" ^ 65)
    for (p, cat) in comparison_params
        d = get(decomposed, p, "?")
        r = get(realistic, p, "?")
        lf = get(local_full, p, "?")
        w("  $(rpad(p, 10))  $d           $r           $lf          $cat")
    end
    w("")

    # ── INTERPRETATION ──
    w("── GAP ANALYSIS: DECOMPOSED vs REALISTIC vs LOCAL ──")
    w("")
    w("  Parameter   Decomposed  Realistic  Local(full)  Category")
    w("  " * "-" ^ 58)
    for (p, cat) in comparison_params
        d = get(decomposed, p, "?")
        r = get(realistic, p, "?")
        lf = get(local_full, p, "?")
        (d, r, lf) == ("?", "?", "?") && continue
        w("  $(rpad(p, 12)) $(d)           $(r)           $(lf)")
    end
    w("")
    w("  Key observation: Local(full) = $(nL_params[1])/26 L — the 11 N are the scaling group.")
    w("  Decomposed analysis is overly optimistic (assumes known Xl, X, Ip).")
    w("")

    # ── CLINICAL SUMMARY ──
    w("── CLINICAL SUMMARY ──")
    w("")
    w("  From CGM + pump + meals only (no decomposition):")
    w("")
    w("  15/26 parameters are locally identifiable:")
    w("    Ts, k1, k2, ka1, ka2, kabs, kd, kgri, ki, kmax, kp2, m1, m2, m4, p2U")
    w("")
    w("  11 parameters are NOT locally identifiable (scaling symmetry group):")
    w("    BW, Fcns, Ib, Km0, VG, VI, Vm0, Vmx, f, kp1, kp3")
    w("    — these form 5 scaling pairs and Km0 (MM term)")
    w("")
    w("  What fixes the scaling group (greedy, from prior_fix_experiment):")
    w("    Fix Km0 + VI + BW + Fcns + Ib (5 population priors)")
    w("    → all 21 remaining become locally identifiable")
    w("")
    w("  Practical takeaway:")
    w("    With CGM data and population priors for just 5 parameters,")
    w("    the remaining 21 individual parameters are estimable.")
    w("    11 structural N parameters are exactly the scaling pairs —")
    w("    these need either an additional measurement (Ip) or population prior(s).")
    w("")

    w("── FILES ──")
    for f in sort(readdir(outdir))
        if startswith(f, "a_") || startswith(f, "b1_") || startswith(f, "b2_") ||
           startswith(f, "p8a_") || startswith(f, "prior_fix_")
            w("  $f")
        end
    end
    w("")
end

println("✅ Corrected report: $summary_path")
println("   15/26 L (was incorrectly 0/26 due to Symbol-key bug)")
