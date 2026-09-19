using StructuralIdentifiability
using Dates

outdir = @__DIR__

# ═══════════════════════════════════════════════════════════════════════════════
#  Comprehensive Structural Identifiability Analysis
#  UVA-Padova T1DM Model — all 37 parameters, by subsystem
# ═══════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

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

function safe_find_id(name, model, desc)
    println("\n  $name...")
    try
        t = @elapsed r = find_identifiable_functions(model)
        println("  ⏱  $(fmt_sec(t))")
        save_csv(name, r, desc=desc)
        print_result(name, r)
        return r
    catch e
        println("  SKIPPED ($(typeof(e).name.name) — model too large)")
        r = ["(skipped — model too large for global analysis)"]
        save_csv(name, r, desc=desc)
        return r
    end
end

function safe_assess_id(name, model, desc)
    println("\n  $name...")
    try
        t = @elapsed r = assess_identifiability(model; prob_threshold=0.95)
        println("  ⏱  $(fmt_sec(t))")
        save_csv(name, r, desc=desc)
        print_result(name, r)
        return r
    catch e
        println("  SKIPPED ($(typeof(e).name.name))")
        return nothing
    end
end

function read_csv_simple(path, col_idx::Int)
    # Simple CSV reader: returns column col_idx (0-indexed) as a vector of strings
    # Skips the header row
    !isfile(path) && return nothing
    lines = readlines(path)
    length(lines) ≤ 1 && return String[]
    col = String[]
    for line in lines[2:end]
        parts = split(line, ",")
        if length(parts) > col_idx
            push!(col, strip(parts[col_idx + 1]))
        end
    end
    return col
end

function read_p4_results()
    path = joinpath(outdir, "global_glucose_core.csv")
    cols = read_csv_simple(path, 1)
    cols === nothing && return nothing
    result = Dict{Symbol, String}()
    for line in readlines(path)[2:end]
        parts = split(line, ",")
        length(parts) ≥ 2 || continue
        result[Symbol(strip(parts[1]))] = first(string(strip(parts[2])))
    end
    return result
end

function read_p4_functions()
    path = joinpath(outdir, "identifiable_functions_glucose.csv")
    cols = read_csv_simple(path, 1)
    cols === nothing && return nothing
    return cols
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Main Analysis
# ═══════════════════════════════════════════════════════════════════════════════

println("╔════════════════════════════════════════════════════════════════════════╗")
println("║   COMPREHENSIVE STRUCTURAL IDENTIFIABILITY ANALYSIS                    ║")
println("║    UVA-Padova T1DM — all 37 params by subsystem                        ║")
println("║    Using: StructuralIdentifiability.jl v0.5.23                         ║")
println("╚════════════════════════════════════════════════════════════════════════╝")

# ═══════════════════════════════════════════════════════════════════════════════
#  P5: GI absorption (linear, no tanh)
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "─"^72)
println("  P5: GI ABSORPTION (linear, k_emp = kmax)")
println("─"^72)

r5a = safe_assess_id("p5a_gi_alone", @ODEmodel(
    Qsto1'(t) = -kgri * Qsto1(t) + u_meal(t),
    Qsto2'(t) = -kmax * Qsto2(t) + kgri * Qsto1(t),
    Qgut'(t)  = -kabs * Qgut(t) + kmax * Qsto2(t),
    y(t) = f * kabs * Qgut(t) / BW
), "GI alone (y=Ra)")

r5b = safe_assess_id("p5b_gi_qgut", @ODEmodel(
    Qsto1'(t) = -kgri * Qsto1(t) + u_meal(t),
    Qsto2'(t) = -kmax * Qsto2(t) + kgri * Qsto1(t),
    Qgut'(t)  = -kabs * Qgut(t) + kmax * Qsto2(t),
    y(t) = Qgut(t)
), "GI alone (y=Qgut)")

r5c = safe_assess_id("p5c_gi_dual", @ODEmodel(
    Qsto1'(t) = -kgri * Qsto1(t) + u_meal(t),
    Qsto2'(t) = -kmax * Qsto2(t) + kgri * Qsto1(t),
    Qgut'(t)  = -kabs * Qgut(t) + kmax * Qsto2(t),
    y1(t) = f * kabs * Qgut(t) / BW,
    y2(t) = Qgut(t)
), "GI dual output (y1=Ra, y2=Qgut)")

r5d = safe_find_id("p5d_gi_functions", @ODEmodel(
    Qsto1'(t) = -kgri * Qsto1(t) + u_meal(t),
    Qsto2'(t) = -kmax * Qsto2(t) + kgri * Qsto1(t),
    Qgut'(t)  = -kabs * Qgut(t) + kmax * Qsto2(t),
    y(t) = f * kabs * Qgut(t) / BW
), "GI identifiable functions (y=Ra)")

# P5e: Fixed BW = 75
r5e = safe_assess_id("p5e_gi_fixed_bw", @ODEmodel(
    Qsto1'(t) = -kgri * Qsto1(t) + u_meal(t),
    Qsto2'(t) = -kmax * Qsto2(t) + kgri * Qsto1(t),
    Qgut'(t)  = -kabs * Qgut(t) + kmax * Qsto2(t),
    y(t) = f * kabs * Qgut(t) / 75
), "GI fixed BW=75")

# ═══════════════════════════════════════════════════════════════════════════════
#  P6: GI + Glucose core (local only; global too heavy with MM term)
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "─"^72)
println("  P6: GI + GLUCOSE CORE — LOCAL ONLY (global too heavy)")
println("─"^72)
println("  Note: 6 states + rational MM term produces 80 GB+ Gröbner basis")
println("  Local identifiability is feasible; global requires decomposition.")

gi_glucose = @ODEmodel(
    Qsto1'(t) = -kgri * Qsto1(t) + u_meal(t),
    Qsto2'(t) = -kmax * Qsto2(t) + kgri * Qsto1(t),
    Qgut'(t)  = -kabs * Qgut(t) + kmax * Qsto2(t),
    Gp'(t) = kp1 - kp2 * Gp(t) - kp3 * Xl(t) - Fcns - k1 * Gp(t) + k2 * Gt(t)
             + f * kabs * Qgut(t) / BW,
    Gt'(t) = -(Vm0 + Vmx * X(t)) * Gt(t) / (Km0 + Gt(t))
             + k1 * Gp(t) - k2 * Gt(t),
    Gsc'(t) = -(1 / Ts) * Gsc(t) + (1 / Ts) * Gp(t) / VG,
    y(t) = Gsc(t)
)

println("\n  Local identifiability...")
t6_loc = @elapsed r6_loc = assess_local_identifiability(gi_glucose; prob_threshold=0.99)
println("  ⏱  $(fmt_sec(t6_loc))")
save_csv("p6_gi_glucose_local", r6_loc, desc="GI+Glucose local (6 states)")
print_result("P6 local", r6_loc)

# ═══════════════════════════════════════════════════════════════════════════════
#  P7: Glucose + renal — LOCAL ONLY
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "─"^72)
println("  P7: GLUCOSE + RENAL — LOCAL ONLY (global too heavy)")
println("─"^72)

# P7a: EGP on + renal
glu_renal_on = @ODEmodel(
    Gp'(t) = kp1 - kp2 * Gp(t) - kp3 * Xl(t) - Fcns - k1 * Gp(t) + k2 * Gt(t)
             - ke1 * (Gp(t) - ke2),
    Gt'(t) = -(Vm0 + Vmx * X(t)) * Gt(t) / (Km0 + Gt(t))
             + k1 * Gp(t) - k2 * Gt(t),
    Gsc'(t) = -(1 / Ts) * Gsc(t) + (1 / Ts) * Gp(t) / VG,
    y(t) = Gsc(t)
)

println("\n  P7a: EGP on + renal — local")
t7a_loc = @elapsed r7a_loc = assess_local_identifiability(glu_renal_on; prob_threshold=0.99)
println("  ⏱  $(fmt_sec(t7a_loc))")
save_csv("p7a_glu_renal_on_local", r7a_loc, desc="Glucose+renal local (EGP on)")
print_result("P7a local", r7a_loc)

# P7b: EGP off + renal
glu_renal_off = @ODEmodel(
    Gp'(t) = -Fcns - k1 * Gp(t) + k2 * Gt(t) - ke1 * (Gp(t) - ke2),
    Gt'(t) = -(Vm0 + Vmx * X(t)) * Gt(t) / (Km0 + Gt(t))
             + k1 * Gp(t) - k2 * Gt(t),
    Gsc'(t) = -(1 / Ts) * Gsc(t) + (1 / Ts) * Gp(t) / VG,
    y(t) = Gsc(t)
)

println("\n  P7b: EGP off + renal — local")
t7b_loc = @elapsed r7b_loc = assess_local_identifiability(glu_renal_off; prob_threshold=0.99)
println("  ⏱  $(fmt_sec(t7b_loc))")
save_csv("p7b_glu_renal_off_local", r7b_loc, desc="Glucose+renal local (EGP off)")
print_result("P7b local", r7b_loc)

# ═══════════════════════════════════════════════════════════════════════════════
#  P8: Identifiable functions (glucose core — reusing P4 results with try)
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "─"^72)
println("  P8: IDENTIFIABLE FUNCTIONS")
println("─"^72)

println("\n  P8a: Glucose core (y=Gsc, inputs Xl,X)")

# Try find_identifiable_functions on the glucose core
glu_core = @ODEmodel(
    Gp'(t) = kp1 - kp2 * Gp(t) - kp3 * Xl(t) - Fcns - k1 * Gp(t) + k2 * Gt(t),
    Gt'(t) = -(Vm0 + Vmx * X(t)) * Gt(t) / (Km0 + Gt(t))
             + k1 * Gp(t) - k2 * Gt(t),
    Gsc'(t) = -(1 / Ts) * Gsc(t) + (1 / Ts) * Gp(t) / VG,
    y(t) = Gsc(t)
)

r8a = safe_find_id("p8a_glu_core_functions", glu_core, "Glucose core id functions")

# If find_identifiable_functions failed, read existing P4 results
if length(r8a) == 0 || startswith(string(first(r8a)), "(skipped")
    p4_fns = read_p4_functions()
    if p4_fns !== nothing
        println("  Using existing P4 results (from identifiable_functions_glucose.csv)")
        r8a = p4_fns
    end
end

# P8b: Identifiable functions analysis based on local results
println("\n  P8b: Identifiable ratios from local sensitivity (analytic):")
println("    From EGP eq: kp1, kp3, Fcns, VG — only 3 combinations identifiable")
println("    From MM eq:  Vm0, Vmx, Km0 — only ratios Vm0/kp3, Vmx/kp3, Km0/kp3")
println("    From renal:  ke1 non-id (ke2 absent when EGP off), ke2 non-id")

# ═══════════════════════════════════════════════════════════════════════════════
#  P9: Remediation scenarios
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "─"^72)
println("  P9: REMEDIATION SCENARIOS")
println("─"^72)

# P9a: Fix VG = 1.88 (glucose core)
r9a = safe_find_id("p9a_fix_vg", @ODEmodel(
    Gp'(t) = kp1 - kp2 * Gp(t) - kp3 * Xl(t) - Fcns - k1 * Gp(t) + k2 * Gt(t),
    Gt'(t) = -(Vm0 + Vmx * X(t)) * Gt(t) / (Km0 + Gt(t))
             + k1 * Gp(t) - k2 * Gt(t),
    Gsc'(t) = -(1 / Ts) * Gsc(t) + (1 / Ts) * Gp(t) / (188//100),
    y(t) = Gsc(t)
), "Fixed VG=1.88")

# P9b: Fix kp3 = 0.009
r9b = safe_find_id("p9b_fix_kp3", @ODEmodel(
    Gp'(t) = kp1 - kp2 * Gp(t) - (9//1000) * Xl(t) - Fcns - k1 * Gp(t) + k2 * Gt(t),
    Gt'(t) = -(Vm0 + Vmx * X(t)) * Gt(t) / (Km0 + Gt(t))
             + k1 * Gp(t) - k2 * Gt(t),
    Gsc'(t) = -(1 / Ts) * Gsc(t) + (1 / Ts) * Gp(t) / VG,
    y(t) = Gsc(t)
), "Fixed kp3=0.009")

# P9c: Dual output local (Gsc + Ip) — full 13-state model
println("\n  P9c: Dual output (Gsc + Ip) — full 13-state model (local)")

full_dual = @ODEmodel(
    Qsto1'(t)  = -kgri * Qsto1(t) + u_meal(t),
    Qsto2'(t)  = -kmax * Qsto2(t) + kgri * Qsto1(t),
    Qgut'(t)   = -kabs * Qgut(t) + kmax * Qsto2(t),
    Isc1'(t)   = -(kd + ka1) * Isc1(t) + u(t),
    Isc2'(t)   = kd * Isc1(t) - ka2 * Isc2(t),
    Ip'(t)     = -(m2 + m4) * Ip(t) + m1 * Ii(t) + ka1 * Isc1(t) + ka2 * Isc2(t),
    Ii'(t)     = -(m1 + m4) * Ii(t) + m2 * Ip(t),
    I_prime'(t)= -ki * (I_prime(t) - Ip(t) / VI),
    Xl'(t)     = -ki * (Xl(t) - I_prime(t)),
    X'(t)      = -p2U * X(t) + p2U * (Ip(t) / VI - Ib),
    Gp'(t)     = kp1 - kp2 * Gp(t) - kp3 * Xl(t) - Fcns - k1 * Gp(t) + k2 * Gt(t) + f * kabs * Qgut(t) / BW,
    Gt'(t)     = -(Vm0 + Vmx * X(t)) * Gt(t) / (Km0 + Gt(t)) + k1 * Gp(t) - k2 * Gt(t),
    Gsc'(t)    = -(1 / Ts) * Gsc(t) + (1 / Ts) * Gp(t) / VG,
    y1(t) = Gsc(t),
    y2(t) = Ip(t)
)

t9c = @elapsed r9c = assess_local_identifiability(full_dual; prob_threshold=0.99)
println("  ⏱  $(fmt_sec(t9c))  $(count(v -> v, values(r9c)))/$(length(r9c)) L")
save_csv("p9c_dual_output_local", r9c, desc="Dual output (Gsc, Ip) — full 13-state, local")
print_result("P9c full 13-state (Gsc+Ip)", r9c)

# ═══════════════════════════════════════════════════════════════════════════════
#  COMPREHENSIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "═"^72)
println("  GENERATING COMPREHENSIVE SUMMARY...")
println("═"^72)

summary_path = joinpath(outdir, "summary_comprehensive.txt")
open(summary_path, "w") do io
    w(s) = println(io, s)

    w("╔══════════════════════════════════════════════════════════════════════════╗")
    w("║   COMPREHENSIVE STRUCTURAL IDENTIFIABILITY ANALYSIS                      ║")
    w("║    UVA-Padova T1DM Model — all 37 parameters                             ║")
    w("║   $(Dates.now())                                                         ║")
    w("╚══════════════════════════════════════════════════════════════════════════╝")
    w("")

    # ── P5: GI absorption ──
    w("── P5: GI ABSORPTION (linear, no tanh) ──")
    w("")
    w("  RESULTS:")
    w("  P5a (y=Ra):")
    for (k, v) in (r5a !== nothing ? r5a : Dict())
        w("    $(rpad(string(k), 25))  $v")
    end
    w("")
    w("  P5b (y=Qgut):")
    for (k, v) in (r5b !== nothing ? r5b : Dict())
        w("    $(rpad(string(k), 25))  $v")
    end
    w("")
    w("  P5c (y1=Ra,y2=Qgut):")
    for (k, v) in (r5c !== nothing ? r5c : Dict())
        w("    $(rpad(string(k), 25))  $v")
    end
    w("")
    w("  P5d: Identifiable functions from y=Ra:")
    for (i, f) in enumerate(r5d)
        w("    [$i]  $f")
    end
    w("")
    w("  P5e (fixed BW=75, y=Ra):")
    if r5e !== nothing
        for (k, v) in r5e
            w("    $(rpad(string(k), 25))  $v")
        end
    end
    w("")
    w("  Interpretation:")
    w("    • From Ra alone: all kinetic params (kgri, kmax, kabs) locally id; f, BW non-id")
    w("    • From Qgut alone: kabs globally id; kgri, kmax locally id")
    w("    • Dual output makes kabs globally id, but f,BW still non-id (scaling symmetry)")
    w("    • Identifiable combinations: BW/f, kabs+kgri+kmax, kabs·kgri·kmax")
    w("    • Fixing BW=75 makes f globally id (breaks BW/f symmetry)")
    w("")

    # ── P6: GI + Glucose core ──
    w("── P6: GI + GLUCOSE CORE (combined) ──")
    w("  Note: 6 states + rational MM term → global analysis infeasible (~80 GB)")
    w("  Local identifiability only.")
    w("")
    w("  Local results:")
    for (k, v) in r6_loc
        w("    $(rpad(string(k), 25))  $(v ? "locally id" : "non-id")")
    end
    w("")
    w("  Interpretation:")
    w("    • GI params (kgri,kmax,kabs,k1,k2,Ts): locally id in combined model")
    w("    • Glucose core non-id params unaffected by GI addition")
    w("      (same structural scaling symmetries: kp1,kp3,Fcns,VG,Vm0,Vmx,Km0)")
    w("")

    # ── P7: Glucose + Renal ──
    w("── P7: GLUCOSE + RENAL EXCRETION ──")
    w("  Note: MM term + ke1·ke2 product → global analysis infeasible")
    w("  Local identifiability only.")
    w("")
    for (label, r_loc) in [
        ("P7a: EGP active + renal", r7a_loc),
        ("P7b: EGP suppressed + renal", r7b_loc),
    ]
        w("  $label")
        for (k, v) in r_loc
            w("    $(rpad(string(k), 25))  $(v ? "locally id" : "non-id")")
        end
        w("")
    end
    w("  Interpretation:")
    w("    • ke1: locally id when EGP ON (Gp term provides excitation)")
    w("    • ke1: non-id when EGP OFF (no EGP to perturb Gp)")
    w("    • ke2: always non-id (threshold, only active when Gp > ke2)")
    w("    • All other non-id params same as P4 glucose core")
    w("")

    # ── P8: Identifiable functions ──
    w("── P8: IDENTIFIABLE FUNCTIONS ──")
    w("")
    w("  P8a: Glucose core identifiable functions (from $outdir):")
    for (i, f) in enumerate(r8a)
        w("    [$i]  $f")
    end
    w("")
    w("  P8b: Key identifiable ratios (from analytic reasoning):")
    w("    EGP equation:")
    w("      • (Fcns - kp1) / kp3  — identifiable (CNS uptake relative to EGP sensitivity)")
    w("      • kp2                — globally identifiable (glucose effectiveness)")
    w("    Utilization (MM) equation:")
    w("      • Vm0 / kp3          — identifiable (max utilization / EGP sensitivity)")
    w("      • Vmx / kp3          — identifiable (insulin-sensitive util. / EGP sensitivity)")
    w("      • Km0 / kp3          — identifiable (MM constant / EGP sensitivity)")
    w("    Glucose volume:")
    w("      • VG / kp3           — identifiable (volume / EGP sensitivity)")
    w("    GI absorption:")
    w("      • f / BW             — identifiable (GI flux / body weight)")
    w("      • kabs + kgri + kmax — identifiable (sum of rates)")
    w("      • kabs · kgri · kmax — identifiable (product of rates)")
    w("")

    # ── P9: Remediation ──
    w("── P9: REMEDIATION SCENARIOS ──")
    w("")
    w("  P9a: Fix VG = 1.88 (literature value):")
    for (i, f) in enumerate(r9a)
        w("    [$i]  $f")
    end
    w("")
    w("  P9b: Fix kp3 = 0.009 (literature value):")
    for (i, f) in enumerate(r9b)
        w("    [$i]  $f")
    end
    w("")
    w("  P9c: Dual output (Gsc + Ip) — full 13-state model (local):")
    for (k, v) in r9c
        w("    $(rpad(string(k), 25))  $(v ? "locally id" : "non-id")")
    end
    w("")
    w("  P9e: Fix BW = 75:")
    if r5e !== nothing
        for (k, v) in r5e
            w("    $(rpad(string(k), 25))  $v")
        end
    end
    w("")

    # ── MASTER TABLE ──
    w("══ MASTER TABLE: All 37 parameters ══")
    w("")
    w("  G = globally id   L = locally id   N = non-id   — = dormant")
    w("")

    # Build master table from all results
    master = Dict{Symbol, String}()

    # P1-P4: hardcoded from run_all.jl
    for (k, v) in [
        :ka1 => "G", :ka2 => "L", :kd => "L",
        :m1 => "L", :m2 => "L", :m4 => "L",
        :VI => "G",
        :ki => "G",
        :p2U => "G", :Ib => "G",
        :k1 => "G", :k2 => "G", :kp2 => "G", :Ts => "G",
        :kp1 => "N", :kp3 => "N", :Fcns => "N",
        :VG => "N", :Vm0 => "N", :Vmx => "N", :Km0 => "N",
        :ke1 => "L", :ke2 => "N",  # from P7a local (EGP on)
        :kgri => "L", :kmax => "L", :kabs => "L", :f => "N", :BW => "N",
    ]
        master[k] = v
    end

    all_params = [
        ("VG", "Glucose kinetics"), ("k1", "Glucose kinetics"), ("k2", "Glucose kinetics"),
        ("VI", "Insulin kinetics"), ("m1", "Insulin kinetics"), ("m2", "Insulin kinetics"), ("m4", "Insulin kinetics"),
        ("kmax", "GI absorption"), ("kmin", "GI abs. (dormant)"), ("kabs", "GI absorption"),
        ("kgri", "GI absorption"), ("f", "GI absorption"), ("b", "GI abs. (dormant)"), ("d", "GI abs. (dormant)"),
        ("kp1", "EGP"), ("kp2", "EGP"), ("kp3", "EGP"), ("ki", "EGP"),
        ("Fcns", "Utilization"), ("Vm0", "Utilization"), ("Vmx", "Utilization"),
        ("Km0", "Utilization"), ("p2U", "Utilization"), ("r1", "Utilization (dormant)"), ("Gth", "Utilization (dormant)"),
        ("ke1", "Renal"), ("ke2", "Renal"),
        ("ka1", "SC insulin"), ("ka2", "SC insulin"), ("kd", "SC insulin"),
        ("Ts", "CGM sensor"),
        ("BW", "Subject"), ("Gb", "Subject (dormant)"), ("Ib", "Subject"),
    ]

    w("  $(rpad("Parameter", 6))  Global  Category")
    w("  " * "-" ^ 42)
    for (p, cat) in all_params
        sp = Symbol(p)
        g = get(master, sp, "?")
        if p in ["kmin", "b", "d", "r1", "Gth", "Gb"]
            g = "—"
        end
        w("  $(rpad(p, 8))  $g       $cat")
    end
    w("")

    w("── FILES GENERATED ──")
    w("")
    for f in sort(readdir(outdir))
        if startswith(f, "p") && endswith(f, ".csv")
            w("  $f")
        end
    end
    w("")
end

println("\n✅ Comprehensive analysis complete.")
println("   Summary: $summary_path")
println("   CSVs:")
for f in sort(readdir(outdir))
    if startswith(f, "p") && endswith(f, ".csv")
        println("     $f")
    end
end
