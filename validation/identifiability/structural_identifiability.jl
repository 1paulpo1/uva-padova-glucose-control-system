using StructuralIdentifiability

# =============================================================================
#  Structural Identifiability Analysis: UVA-Padova T1DM Model (no meals)
#  Package: StructuralIdentifiability.jl v0.5.23
#
#  Scenarios (2 x 2 design):
#    1. EGP active × no renal excretion  (≈ Gp ≤ ke2)
#    2. EGP suppressed × no renal excretion
#    3. EGP active × renal excretion active (≈ Gp > ke2)
#    4. EGP suppressed × renal excretion active
#
#  Assumptions: r1 = 0 (no hypoglycemia risk), D = 0 (no meals)
#  → Fully rational ODE system
# =============================================================================

outdir = @__DIR__

# ---------------------------------------------------------------------------
#  Helper: format results for display and CSV
# ---------------------------------------------------------------------------
function format_result(val)
    if val == true
        return "locally"
    elseif val == false
        return "nonidentifiable"
    else
        return string(val)  # :globally, :locally, :nonidentifiable
    end
end

function print_results(title, results)
    println("\n" ^ 2)
    println("=" ^ 72)
    println("  $title")
    println("=" ^ 72)
    for (k, v) in results
        println("  $(rpad(string(k), 30))  →  $(format_result(v))")
    end
end

function save_csv(filename, results; extra_cols = Dict())
    path = joinpath(outdir, filename)
    open(path, "w") do io
        # Header
        headers = ["variable", "result"]
        for (k, _) in extra_cols
            push!(headers, string(k))
        end
        println(io, join(headers, ","))
        # Rows
        for (k, v) in results
            row = [string(k), format_result(v)]
            for (_, vals) in extra_cols
                push!(row, string(get(vals, k, "")))
            end
            println(io, join(row, ","))
        end
    end
    println("  → Saved $(abspath(path))")
end

# ===========================================================================
#  Model 1A: EGP active, no renal excretion
#  Gp'(t) = kp1 - kp2·Gp - kp3·Xl - Fcns - k1·Gp + k2·Gt
#  no renal term (assuming Gp ≤ ke2)
# ===========================================================================
println("─" ^ 72)
println("  Defining Model 1A: EGP active, no renal excretion")

model_1a = @ODEmodel(
    Gp'(t) = kp1 - kp2 * Gp(t) - kp3 * Xl(t) - Fcns - k1 * Gp(t) + k2 * Gt(t),

    Gt'(t) = -(Vm0 + Vmx * X(t)) * Gt(t) / (Km0 + Gt(t))
             + k1 * Gp(t) - k2 * Gt(t),

    Ip'(t) = -(m2 + m4) * Ip(t) + m1 * Ii(t) + ka1 * Isc1(t) + ka2 * Isc2(t),

    Ii'(t) = -(m1 + m4) * Ii(t) + m2 * Ip(t),

    I_prime'(t) = -ki * (I_prime(t) - Ip(t) / VI),

    Xl'(t) = -ki * (Xl(t) - I_prime(t)),

    X'(t) = -p2U * X(t) + p2U * (Ip(t) / VI - Ib),

    Isc1'(t) = -(kd + ka1) * Isc1(t) + u(t),

    Isc2'(t) = kd * Isc1(t) - ka2 * Isc2(t),

    Gsc'(t) = -(1 / Ts) * Gsc(t) + (1 / Ts) * Gp(t) / VG,

    y(t) = Gsc(t)
)

println("  States:      10 (Gp, Gt, Ip, Ii, I_prime, Xl, X, Isc1, Isc2, Gsc)")
println("  Parameters:  21")
println("  Inputs:       1 (u)")
println("  Outputs:      1 (y = Gsc)")

# ===========================================================================
#  Model 1B: EGP suppressed, no renal excretion
#  Gp'(t) = -Fcns - k1·Gp + k2·Gt  (EGP == 0)
# ===========================================================================
println("\n─" ^ 72)
println("  Defining Model 1B: EGP suppressed, no renal excretion")

model_1b = @ODEmodel(
    Gp'(t) = -Fcns - k1 * Gp(t) + k2 * Gt(t),

    Gt'(t) = -(Vm0 + Vmx * X(t)) * Gt(t) / (Km0 + Gt(t))
             + k1 * Gp(t) - k2 * Gt(t),

    Ip'(t) = -(m2 + m4) * Ip(t) + m1 * Ii(t) + ka1 * Isc1(t) + ka2 * Isc2(t),

    Ii'(t) = -(m1 + m4) * Ii(t) + m2 * Ip(t),

    I_prime'(t) = -ki * (I_prime(t) - Ip(t) / VI),

    Xl'(t) = -ki * (Xl(t) - I_prime(t)),

    X'(t) = -p2U * X(t) + p2U * (Ip(t) / VI - Ib),

    Isc1'(t) = -(kd + ka1) * Isc1(t) + u(t),

    Isc2'(t) = kd * Isc1(t) - ka2 * Isc2(t),

    Gsc'(t) = -(1 / Ts) * Gsc(t) + (1 / Ts) * Gp(t) / VG,

    y(t) = Gsc(t)
)

println("  States:      10")
println("  Parameters:  20 (no kp1)")

# ===========================================================================
#  Model 2A: EGP active, renal excretion active
#  Gp'(t) = ... - ke1·(Gp - ke2)     (assuming Gp > ke2)
# ===========================================================================
println("\n─" ^ 72)
println("  Defining Model 2A: EGP active, renal excretion active")

model_2a = @ODEmodel(
    Gp'(t) = kp1 - kp2 * Gp(t) - kp3 * Xl(t) - Fcns - k1 * Gp(t) + k2 * Gt(t)
             - ke1 * (Gp(t) - ke2),

    Gt'(t) = -(Vm0 + Vmx * X(t)) * Gt(t) / (Km0 + Gt(t))
             + k1 * Gp(t) - k2 * Gt(t),

    Ip'(t) = -(m2 + m4) * Ip(t) + m1 * Ii(t) + ka1 * Isc1(t) + ka2 * Isc2(t),

    Ii'(t) = -(m1 + m4) * Ii(t) + m2 * Ip(t),

    I_prime'(t) = -ki * (I_prime(t) - Ip(t) / VI),

    Xl'(t) = -ki * (Xl(t) - I_prime(t)),

    X'(t) = -p2U * X(t) + p2U * (Ip(t) / VI - Ib),

    Isc1'(t) = -(kd + ka1) * Isc1(t) + u(t),

    Isc2'(t) = kd * Isc1(t) - ka2 * Isc2(t),

    Gsc'(t) = -(1 / Ts) * Gsc(t) + (1 / Ts) * Gp(t) / VG,

    y(t) = Gsc(t)
)

println("  States:      10")
println("  Parameters:  23 (kp1..kp3, ke1, ke2, ...)")

# ===========================================================================
#  Model 2B: EGP suppressed, renal excretion active
# ===========================================================================
println("\n─" ^ 72)
println("  Defining Model 2B: EGP suppressed, renal excretion active")

model_2b = @ODEmodel(
    Gp'(t) = -Fcns - k1 * Gp(t) + k2 * Gt(t)
             - ke1 * (Gp(t) - ke2),

    Gt'(t) = -(Vm0 + Vmx * X(t)) * Gt(t) / (Km0 + Gt(t))
             + k1 * Gp(t) - k2 * Gt(t),

    Ip'(t) = -(m2 + m4) * Ip(t) + m1 * Ii(t) + ka1 * Isc1(t) + ka2 * Isc2(t),

    Ii'(t) = -(m1 + m4) * Ii(t) + m2 * Ip(t),

    I_prime'(t) = -ki * (I_prime(t) - Ip(t) / VI),

    Xl'(t) = -ki * (Xl(t) - I_prime(t)),

    X'(t) = -p2U * X(t) + p2U * (Ip(t) / VI - Ib),

    Isc1'(t) = -(kd + ka1) * Isc1(t) + u(t),

    Isc2'(t) = kd * Isc1(t) - ka2 * Isc2(t),

    Gsc'(t) = -(1 / Ts) * Gsc(t) + (1 / Ts) * Gp(t) / VG,

    y(t) = Gsc(t)
)

println("  States:      10")
println("  Parameters:  22 (no kp1)")

# ===========================================================================
#  STEP 1: Local identifiability for all 4 models
# ===========================================================================
println("\n" ^ 2)
println("#" ^ 72)
println("#  STEP 1: LOCAL IDENTIFIABILITY (all 4 models)")
println("#" ^ 72)

local_results = Dict()

for (name, model, desc) in [
    ("1a_egp_on_no_renal",   model_1a, "EGP active, no renal excretion"),
    ("1b_egp_off_no_renal",  model_1b, "EGP suppressed, no renal excretion"),
    ("2a_egp_on_renal",      model_2a, "EGP active, renal excretion active"),
    ("2b_egp_off_renal",     model_2b, "EGP suppressed, renal excretion active"),
]
    println("\n─" ^ 72)
    println("  Local identifiability: $desc")
    println("─" ^ 72)
    t_start = time()
    result = assess_local_identifiability(model; prob_threshold = 0.99)
    elapsed = time() - t_start
    println("  ⏱  $(round(elapsed, digits=2)) s")
    println()
    for (k, v) in result
        println("    $(rpad(string(k), 30))  →  $(v ? "locally" : "nonidentifiable")")
    end
    local_results[name] = (result, desc, elapsed)

    # Save CSV
    fname = "local_identifiability_$(name).csv"
    open(joinpath(outdir, fname), "w") do io
        println(io, "variable,locally_identifiable,note")
        for (k, v) in result
            println(io, "$(string(k)),$(v ? "yes" : "no"),$desc")
        end
    end
    println("  → Saved $(joinpath(outdir, fname))")
end

# ===========================================================================
#  STEP 2: Global identifiability for Model 1A (EGP active, no renal)
#  This is the most relevant physiological scenario.
#  Note: 21 parameters → may take 30+ minutes.
# ===========================================================================
println("\n" ^ 2)
println("#" ^ 72)
println("#  STEP 2: GLOBAL IDENTIFIABILITY (Model 1A: EGP active, no renal)")
println("#" ^ 72)

t_start = time()
global_1a = assess_identifiability(model_1a; prob_threshold = 0.99)
elapsed_global = time() - t_start

println("\n  ⏱  $(round(elapsed_global, digits=2)) s")
println()
for (k, v) in global_1a
    println("    $(rpad(string(k), 30))  →  $(format_result(v))")
end

# Save CSV
open(joinpath(outdir, "global_identifiability_1a_egp_on_no_renal.csv"), "w") do io
    println(io, "variable,identifiability")
    for (k, v) in global_1a
        println(io, "$(string(k)),$(format_result(v))")
    end
end
println("  → Saved global_identifiability_1a_egp_on_no_renal.csv")

# ===========================================================================
#  STEP 3: Global identifiability for Model 1B (EGP suppressed, no renal)
#  Fewer parameters → faster.
# ===========================================================================
println("\n" ^ 2)
println("#" ^ 72)
println("#  STEP 3: GLOBAL IDENTIFIABILITY (Model 1B: EGP suppressed, no renal)")
println("#" ^ 72)

t_start = time()
global_1b = assess_identifiability(model_1b; prob_threshold = 0.99)
elapsed_global_1b = time() - t_start

println("\n  ⏱  $(round(elapsed_global_1b, digits=2)) s")
println()
for (k, v) in global_1b
    println("    $(rpad(string(k), 30))  →  $(format_result(v))")
end

open(joinpath(outdir, "global_identifiability_1b_egp_off_no_renal.csv"), "w") do io
    println(io, "variable,identifiability")
    for (k, v) in global_1b
        println(io, "$(string(k)),$(format_result(v))")
    end
end
println("  → Saved global_identifiability_1b_egp_off_no_renal.csv")

# ===========================================================================
#  STEP 4: Global identifiability for Model 2A (EGP active, renal active)
# ===========================================================================
println("\n" ^ 2)
println("#" ^ 72)
println("#  STEP 4: GLOBAL IDENTIFIABILITY (Model 2A: EGP active, renal active)")
println("#" ^ 72)

t_start = time()
global_2a = assess_identifiability(model_2a; prob_threshold = 0.99)
elapsed_global_2a = time() - t_start

println("\n  ⏱  $(round(elapsed_global_2a, digits=2)) s")
println()
for (k, v) in global_2a
    println("    $(rpad(string(k), 30))  →  $(format_result(v))")
end

open(joinpath(outdir, "global_identifiability_2a_egp_on_renal.csv"), "w") do io
    println(io, "variable,identifiability")
    for (k, v) in global_2a
        println(io, "$(string(k)),$(format_result(v))")
    end
end
println("  → Saved global_identifiability_2a_egp_on_renal.csv")

# ===========================================================================
#  STEP 5: Identifiable functions for Model 1A
#  Find which combinations of non-identifiable params ARE identifiable
# ===========================================================================
println("\n" ^ 2)
println("#" ^ 72)
println("#  STEP 5: IDENTIFIABLE FUNCTIONS (Model 1A: EGP active, no renal)")
println("#" ^ 72)

# Collect non-identifiable parameters from global analysis
non_id_params = Symbol[]
for (k, v) in global_1a
    if v == :nonidentifiable
        push!(non_id_params, k)
    end
end

if length(non_id_params) > 0
    println("  Non-identifiable params: $(join(string.(non_id_params), ", "))")
    println("  Searching for identifiable functions...")

    t_start = time()
    id_funcs = find_identifiable_functions(model_1a; prob_threshold = 0.99)
    elapsed_idfunc = time() - t_start

    println("\n  ⏱  $(round(elapsed_idfunc, digits=2)) s")
    println()
    for (func, status) in id_funcs
        println("    $(rpad(string(func), 40))  →  $(format_result(status))")
    end

    open(joinpath(outdir, "identifiable_functions_1a.csv"), "w") do io
        println(io, "function,identifiability")
        for (func, status) in id_funcs
            println(io, "$(string(func)),$(format_result(status))")
        end
    end
    println("  → Saved identifiable_functions_1a.csv")
else
    println("  All parameters are globally identifiable — no need for function search.")
end

# ===========================================================================
#  SUMMARY
# ===========================================================================
println("\n" ^ 2)
println("#" ^ 72)
println("#  SUMMARY")
println("#" ^ 72)

println("\n  ┌─────────────────────────────────────────────────────────────────────┐")
println("  │  LOCAL IDENTIFIABILITY                                              │")
println("  ├─────────────────────────────────────────────────────────────────────┤")
for (name, (result, desc, elapsed)) in local_results
    n_id = count(v -> v == true, values(result))
    n_total = length(result)
    println("  │  $(rpad(name, 30))  $(lpad(n_id, 2))/$(lpad(n_total, 2)) id  $(lpad(round(elapsed, digits=1), 6)) s  │")
end
println("  └─────────────────────────────────────────────────────────────────────┘")

println("\n  ┌─────────────────────────────────────────────────────────────────────┐")
println("  │  GLOBAL IDENTIFIABILITY                                             │")
println("  ├─────────────────────────────────────────────────────────────────────┤")
for (name, result, elapsed) in [
    ("1a_egp_on_no_renal",  global_1a, elapsed_global),
    ("1b_egp_off_no_renal", global_1b, elapsed_global_1b),
    ("2a_egp_on_renal",     global_2a, elapsed_global_2a),
]
    n_glob = count(v -> v == :globally, values(result))
    n_loc = count(v -> v == :locally, values(result))
    n_non = count(v -> v == :nonidentifiable, values(result))
    n_total = length(result)
    println("  │  $(rpad(name, 30))  G:$(lpad(n_glob,2)) L:$(lpad(n_loc,2)) N:$(lpad(n_non,2))  $(lpad(round(elapsed, digits=1), 7)) s  │")
end
println("  └─────────────────────────────────────────────────────────────────────┘")

println()
println("  All results saved to: $(outdir)")
println()
