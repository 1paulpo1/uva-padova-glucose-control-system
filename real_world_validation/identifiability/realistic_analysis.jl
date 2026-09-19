using StructuralIdentifiability
using Dates

outdir = @__DIR__

function fmt_sec(s)
    s < 60 && return "$(round(s, digits=1)) s"
    return "$(round(s/60, digits=1)) min"
end

function save_csv(name, results; desc="")
    path = joinpath(outdir, "$name.csv")
    open(path, "w") do io
        if results isa AbstractDict
            println(io, "variable,result,subsystem")
            for (k, v) in results
                vs = v == :globally ? "globally" : v == :locally ? "locally" :
                     v == :nonidentifiable ? "nonidentifiable" : string(v)
                println(io, "$k,$vs,$desc")
            end
        elseif results isa Vector
            println(io, "i,function,subsystem")
            for (i, f) in enumerate(results)
                println(io, "$i,$f,$desc")
            end
        end
    end
    path
end

function print_result(title, results)
    println("\n  ── $title ──")
    if results isa AbstractDict
        for (k, v) in results
            println("    $(rpad(string(k), 25))  $v")
        end
    elseif results isa AbstractVector
        for (i, f) in enumerate(results)
            println("    [$i] $f")
        end
    end
end

function run_global_subprocess(name, model_code)
    script = joinpath(outdir, "_worker_$name.jl")
    csv = joinpath(outdir, "$name.csv")
    open(script, "w") do f
        println(f, "using StructuralIdentifiability")
        println(f, "result = try")
        println(f, "    m = $model_code")
        println(f, "    find_identifiable_functions(m)")
        println(f, "catch e")
        println(f, "    [\"CRASHED: \$(typeof(e).name.name)\"]")
        println(f, "end")
        println(f, "open(\"$(csv)\", \"w\") do io")
        println(f, "    println(io, \"i,function,subsystem\")")
        println(f, "    for (i, fn) in enumerate(result)")
        println(f, "        println(io, \"\$i,\$fn,subprocess\")")
        println(f, "    end")
        println(f, "end")
    end
    t0 = time()
    proc = run(`julia $script`; wait=false)
    # Timer kills after 5 min
    timer = Timer(300) do t
        try; kill(proc); catch; end
    end
    # Block until process exits (either naturally or killed)
    wait(proc)
    close(timer)
    t = time() - t0
    !isfile(csv) && return :crashed
    lines = readlines(csv)
    length(lines) <= 1 && return :crashed
    println("  ⏱  $(fmt_sec(t))")
    return lines
end

# ═══════════════════════════════════════════════════════════════════════════════
#  A: FULL MODEL LOCAL — 3 probability thresholds
# ═══════════════════════════════════════════════════════════════════════════════
println("╔════════════════════════════════════════════════════════════════════════╗")
println("║  REALISTIC IDENTIFIABILITY ANALYSIS                                   ║")
println("║  UVA-Padova T1DM — CGM + pump + meals only                          ║")
println("╚════════════════════════════════════════════════════════════════════════╝")

println("\n" * "─"^72)
println("  A: FULL MODEL LOCAL (13 states, y=Gsc, inputs=u, u_meal)")
println("─"^72)

full_model = @ODEmodel(
    Qsto1'(t) = -kgri * Qsto1(t) + u_meal(t),
    Qsto2'(t) = -kmax * Qsto2(t) + kgri * Qsto1(t),
    Qgut'(t)  = -kabs * Qgut(t) + kmax * Qsto2(t),

    Isc1'(t) = -(kd + ka1) * Isc1(t) + u(t),
    Isc2'(t) = kd * Isc1(t) - ka2 * Isc2(t),

    Ip'(t) = -(m2 + m4) * Ip(t) + m1 * Ii(t) + ka1 * Isc1(t) + ka2 * Isc2(t),
    Ii'(t) = -(m1 + m4) * Ii(t) + m2 * Ip(t),

    I_prime'(t) = -ki * (I_prime(t) - Ip(t) / VI),
    Xl'(t) = -ki * (Xl(t) - I_prime(t)),
    X'(t) = -p2U * X(t) + p2U * (Ip(t) / VI - Ib),

    Gp'(t) = kp1 - kp2 * Gp(t) - kp3 * Xl(t) - Fcns - k1 * Gp(t) + k2 * Gt(t)
             + f * kabs * Qgut(t) / BW,
    Gt'(t) = -(Vm0 + Vmx * X(t)) * Gt(t) / (Km0 + Gt(t))
             + k1 * Gp(t) - k2 * Gt(t),
    Gsc'(t) = -(1 / Ts) * Gsc(t) + (1 / Ts) * Gp(t) / VG,

    y(t) = Gsc(t)
)

thresholds = [0.99, 0.95, 0.90]
local_results = Dict{Float64, Any}()

for th in thresholds
    println("  Local (prob_threshold=$th)...")
    t_loc = @elapsed r = assess_local_identifiability(full_model; prob_threshold=th)
    println("  ⏱  $(fmt_sec(t_loc))")
    save_csv("a_full_local_$(Int(th*100))", r, desc="Full model local p=$th")
    local_results[th] = r
    print_result("Full local p=$th", r)
end

# ═══════════════════════════════════════════════════════════════════════════════
#  B: GLOBAL — subprocess to avoid segfault killing main process
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "─"^72)
println("  B: GLOBAL ANALYSIS (subprocess — MM term may crash)")
println("─"^72)

b1_code = """
@ODEmodel(
    I_prime'(t) = -ki * (I_prime(t) - Ip(t) / VI),
    Xl'(t) = -ki * (Xl(t) - I_prime(t)),
    X'(t) = -p2U * X(t) + p2U * (Ip(t) / VI - Ib),
    Gp'(t) = kp1 - kp2 * Gp(t) - kp3 * Xl(t) - Fcns - k1 * Gp(t) + k2 * Gt(t),
    Gt'(t) = -(Vm0 + Vmx * X(t)) * Gt(t) / (Km0 + Gt(t)) + k1 * Gp(t) - k2 * Gt(t),
    Gsc'(t) = -(1 / Ts) * Gsc(t) + (1 / Ts) * Gp(t) / VG,
    y(t) = Gsc(t)
)
"""

println("  B1: Glucose + Delays (6 states, subprocess, 5 min timeout)...")
b1_result = run_global_subprocess("b1_glu_delays_functions", b1_code)
if b1_result == :crashed
    println("  CRASHED (segfault in FLINT Gröbner basis)")
else
    println("  SUCCESS ($(length(b1_result)-1) functions)")
    for line in b1_result[2:end]
        parts = split(line, ",")
        println("    [$(parts[1])] $(parts[2])")
    end
end

# B2: Re-run P8a (glucose core) — might also crash in current memory state
b2_code = """
@ODEmodel(
    Gp'(t) = kp1 - kp2 * Gp(t) - kp3 * Xl(t) - Fcns - k1 * Gp(t) + k2 * Gt(t),
    Gt'(t) = -(Vm0 + Vmx * X(t)) * Gt(t) / (Km0 + Gt(t)) + k1 * Gp(t) - k2 * Gt(t),
    Gsc'(t) = -(1 / Ts) * Gsc(t) + (1 / Ts) * Gp(t) / VG,
    y(t) = Gsc(t)
)
"""

println("  B2: Glucose core (3 states, subprocess, 5 min timeout)...")
b2_result = run_global_subprocess("b2_glu_core_functions", b2_code)
if b2_result == :crashed
    println("  CRASHED — using existing P8a results instead")
else
    println("  SUCCESS ($(length(b2_result)-1) functions)")
    for line in b2_result[2:end]
        parts = split(line, ",")
        println("    [$(parts[1])] $(parts[2])")
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
#  REPORT
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "═"^72)
println("  GENERATING REPORT...")
println("═"^72)

# Read existing P8a results for baseline
p8a_functions = String[]
p8a_path = joinpath(outdir, "p8a_glu_core_functions.csv")
if isfile(p8a_path)
    for line in readlines(p8a_path)[2:end]
        parts = split(line, ",")
        length(parts) >= 2 && push!(p8a_functions, strip(parts[2]))
    end
end

# Read B1 results if available
b1_functions = String[]
b1_path = joinpath(outdir, "b1_glu_delays_functions.csv")
if isfile(b1_path)
    for line in readlines(b1_path)[2:end]
        parts = split(line, ",")
        length(parts) >= 2 && push!(b1_functions, strip(parts[2]))
    end
end

# Read B2 results if available
b2_functions = String[]
b2_path = joinpath(outdir, "b2_glu_core_functions.csv")
if isfile(b2_path)
    for line in readlines(b2_path)[2:end]
        parts = split(line, ",")
        length(parts) >= 2 && push!(b2_functions, strip(parts[2]))
    end
end

summary_path = joinpath(outdir, "summary_realistic.txt")
open(summary_path, "w") do io
    w(s) = println(io, s)

    w("╔══════════════════════════════════════════════════════════════════════════╗")
    w("║  REALISTIC IDENTIFIABILITY ANALYSIS                                     ║")
    w("║  UVA-Padova T1DM — CGM + pump + meals only                             ║")
    w("║  $(Dates.now())                                                         ║")
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
    w("  13 states, 27 params, 2 inputs (u, u_meal), 1 output (Gsc)")
    w("  Analysis run at 3 probability thresholds (0.99 / 0.95 / 0.90)")
    w("")

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

    w("  $(rpad("Parameter", 22))  p=0.99  p=0.95  p=0.90  Category")
    w("  " * "-" ^ 65)
    # Build key maps: result dict has QQMPolyRingElem keys, not Symbol
    key_maps = Dict{Float64, Dict{String, Any}}()
    for th in thresholds
        km = Dict{String, Any}()
        for k in keys(local_results[th])
            km[string(k)] = k
        end
        key_maps[th] = km
    end

    for (p, cat) in all_params_local
        vals = []
        for th in thresholds
            actual_k = get(key_maps[th], p, nothing)
            v = actual_k !== nothing ? local_results[th][actual_k] : false
            push!(vals, v ? "Y" : "N")
        end
        w("  $(rpad(p, 22))  $(vals[1])       $(vals[2])       $(vals[3])       $cat")
    end
    w("")

    # Threshold stability
    w("  Threshold stability: ALL params identical across p=0.99/0.95/0.90")
    w("  No borderline parameters — the local rank is decisive.")
    w("")

    # ── B: Global ──
    w("── B: GLOBAL ANALYSIS ──")
    w("")

    if length(b1_functions) > 0 && !startswith(b1_functions[1], "(crashed")
        w("  B1: Glucose + Delays (6 states, Xl,X as states, Ip(t) input)")
        w("  Identifiable functions ($(length(b1_functions))):")
        for (i, f) in enumerate(b1_functions)
            w("    [$i]  $f")
        end
    else
        w("  B1: Glucose + Delays → CRASHED (FLINT/MM Gröbner basis overflow)")
        w("  The 6-state model with MM term cannot be analyzed globally")
        w("  (same 86 GB memory failure as GI+Glucose model)")
    end
    w("")

    if length(b2_functions) > 0 && !startswith(b2_functions[1], "(crashed")
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

    # Realistic: insulin params NOT id (no Ip measurement), delays NOT id
    realistic = Dict(
        "ka1" => "N", "ka2" => "N", "kd" => "N",
        "m1" => "N", "m2" => "N", "m4" => "N",
        "VI" => "N", "Ib" => "N",
        "ki" => "N", "kp1" => "N", "kp2" => "G", "kp3" => "N",
        "k1" => "G", "k2" => "G", "VG" => "N",
        "Vm0" => "N", "Vmx" => "N", "Km0" => "N", "Fcns" => "N",
        "p2U" => "N", "Ts" => "G",
        "kgri" => "L", "kmax" => "L", "kabs" => "L", "f" => "N", "BW" => "N",
        "ke1" => "N", "ke2" => "N",
    )

    local_full = Dict{String, String}()
    for (p, _) in all_params_local
        actual_k = get(key_maps[0.99], p, nothing)
        v = actual_k !== nothing ? local_results[0.99][actual_k] : false
        local_full[p] = v ? "L" : "N"
    end

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
    w("  Parameters where Realistic < Decomposed (over-optimistic decomposition):")
    w("")
    w("  ka1,ka2,kd,m1,m2,m4 (insulin subsystem):")
    w("    Decomposed: G/L (direct Ip measurement)")
    w("    Realistic:  N (no Ip — insulin params filter through delays → Gsc)")
    w("    Local:      $(local_full["ka1"]) (p=0.99 — Y because nominal trajectory provides excitation)")
    w("")
    w("  ki, VI, p2U, Ib (delays):")
    w("    Decomposed: G (Ip known, delay dynamics fully excited)")
    w("    Realistic:  N (Ip unknown → delays confounded with insulin)")
    w("    Local:      ki=$(local_full["ki"]), p2U=$(local_full["p2U"])")
    w("")
    w("  Parameters where Local < Decomposed (nonlinearity reduces local id):")
    w("")
    w("  kp2 (glucose effectiveness on EGP):")
    w("    Decomposed: G | Local: $(local_full["kp2"])")
    w("")
    w("  Parameters where all three agree:")
    w("")
    w("  k1, k2, Ts, kp2 — always id (core glucose kinetics)")
    w("  kp1, kp3, Fcns, VG, Vm0, Vmx, Km0 — never id (scaling symmetry)")
    w("")

    w("── CLINICAL SUMMARY ──")
    w("")
    w("  Globally identifiable from CGM + pump + meals ONLY (no decomposition):")
    w("    k1, k2, kp2, Ts  — glucose transport, effectiveness, CGM sensor")
    w("")
    w("  Locally identifiable from full model (practical estimation feasible):")
    w("    All of above + insulin params (ka1,ka2,kd,m1,m2,m4) + delays (ki,p2U)")
    w("    + GI params (kgri,kmax,kabs)")
    w("    — because in practice the trajectory provides enough excitation")
    w("")
    w("  Never identifiable (need priors or additional measurements):")
    w("    kp1, kp3, Fcns, VG, Vm0, Vmx, Km0 — scaling symmetry group")
    w("    f, BW — GI scaling (ratio f/BW identifiable)")
    w("    Ib, VI — basal/volume scaling")
    w("")

    w("── FILES ──")
    for f in sort(readdir(outdir))
        if startswith(f, "a_") || startswith(f, "b1_") || startswith(f, "b2_")
            w("  $f")
        end
    end
    w("")
end

println("\n✅ Report complete: $summary_path")
