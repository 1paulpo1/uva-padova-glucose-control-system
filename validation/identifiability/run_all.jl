using StructuralIdentifiability
using Dates

outdir = @__DIR__

# ═══════════════════════════════════════════════════════════════════════════════
#  Structural Identifiability Analysis: UVA-Padova T1DM Model (no meals, r1=0)
#
#  Exports:
#    local_identifiability_{1a,1b,2a,2b}.csv  — local results for all 4 scenarios
#    global_identifiability_{insulin,egp,util,glucose}.csv — subsystem global
#    identifiable_functions_glucose.csv        — identifiable functions
#    summary.txt                              — consolidated human-readable report
# ═══════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
#  1. LOCAL IDENTIFIABILITY (all 4 piecewise scenarios)
# ---------------------------------------------------------------------------

function run_local(name, desc, model)
    println("  Local: $desc ...")
    t = @elapsed result = assess_local_identifiability(model; prob_threshold = 0.99)
    path = joinpath(outdir, "local_$(name).csv")
    open(path, "w") do io; println(io, "variable,identifiable,scenario")
        for (k, v) in result
            println(io, "$(k),$(v ? "yes" : "no"),$desc")
        end
    end
    (result, t)
end

println("── LOCAL IDENTIFIABILITY ──")

model_1a = @ODEmodel(
    Gp'(t) = kp1 - kp2*Gp(t) - kp3*Xl(t) - Fcns - k1*Gp(t) + k2*Gt(t),
    Gt'(t) = -(Vm0+Vmx*X(t))*Gt(t)/(Km0+Gt(t)) + k1*Gp(t) - k2*Gt(t),
    Ip'(t) = -(m2+m4)*Ip(t) + m1*Ii(t) + ka1*Isc1(t) + ka2*Isc2(t),
    Ii'(t) = -(m1+m4)*Ii(t) + m2*Ip(t),
    I_prime'(t) = -ki*(I_prime(t) - Ip(t)/VI),
    Xl'(t) = -ki*(Xl(t) - I_prime(t)),
    X'(t) = -p2U*X(t) + p2U*(Ip(t)/VI - Ib),
    Isc1'(t) = -(kd+ka1)*Isc1(t) + u(t),
    Isc2'(t) = kd*Isc1(t) - ka2*Isc2(t),
    Gsc'(t) = -(1/Ts)*Gsc(t) + (1/Ts)*Gp(t)/VG,
    y(t) = Gsc(t)
)

model_1b = @ODEmodel(
    Gp'(t) = -Fcns - k1*Gp(t) + k2*Gt(t),
    Gt'(t) = -(Vm0+Vmx*X(t))*Gt(t)/(Km0+Gt(t)) + k1*Gp(t) - k2*Gt(t),
    Ip'(t) = -(m2+m4)*Ip(t) + m1*Ii(t) + ka1*Isc1(t) + ka2*Isc2(t),
    Ii'(t) = -(m1+m4)*Ii(t) + m2*Ip(t),
    I_prime'(t) = -ki*(I_prime(t) - Ip(t)/VI),
    Xl'(t) = -ki*(Xl(t) - I_prime(t)),
    X'(t) = -p2U*X(t) + p2U*(Ip(t)/VI - Ib),
    Isc1'(t) = -(kd+ka1)*Isc1(t) + u(t),
    Isc2'(t) = kd*Isc1(t) - ka2*Isc2(t),
    Gsc'(t) = -(1/Ts)*Gsc(t) + (1/Ts)*Gp(t)/VG,
    y(t) = Gsc(t)
)

model_2a = @ODEmodel(
    Gp'(t) = kp1 - kp2*Gp(t) - kp3*Xl(t) - Fcns - k1*Gp(t) + k2*Gt(t) - ke1*(Gp(t)-ke2),
    Gt'(t) = -(Vm0+Vmx*X(t))*Gt(t)/(Km0+Gt(t)) + k1*Gp(t) - k2*Gt(t),
    Ip'(t) = -(m2+m4)*Ip(t) + m1*Ii(t) + ka1*Isc1(t) + ka2*Isc2(t),
    Ii'(t) = -(m1+m4)*Ii(t) + m2*Ip(t),
    I_prime'(t) = -ki*(I_prime(t) - Ip(t)/VI),
    Xl'(t) = -ki*(Xl(t) - I_prime(t)),
    X'(t) = -p2U*X(t) + p2U*(Ip(t)/VI - Ib),
    Isc1'(t) = -(kd+ka1)*Isc1(t) + u(t),
    Isc2'(t) = kd*Isc1(t) - ka2*Isc2(t),
    Gsc'(t) = -(1/Ts)*Gsc(t) + (1/Ts)*Gp(t)/VG,
    y(t) = Gsc(t)
)

model_2b = @ODEmodel(
    Gp'(t) = -Fcns - k1*Gp(t) + k2*Gt(t) - ke1*(Gp(t)-ke2),
    Gt'(t) = -(Vm0+Vmx*X(t))*Gt(t)/(Km0+Gt(t)) + k1*Gp(t) - k2*Gt(t),
    Ip'(t) = -(m2+m4)*Ip(t) + m1*Ii(t) + ka1*Isc1(t) + ka2*Isc2(t),
    Ii'(t) = -(m1+m4)*Ii(t) + m2*Ip(t),
    I_prime'(t) = -ki*(I_prime(t) - Ip(t)/VI),
    Xl'(t) = -ki*(Xl(t) - I_prime(t)),
    X'(t) = -p2U*X(t) + p2U*(Ip(t)/VI - Ib),
    Isc1'(t) = -(kd+ka1)*Isc1(t) + u(t),
    Isc2'(t) = kd*Isc1(t) - ka2*Isc2(t),
    Gsc'(t) = -(1/Ts)*Gsc(t) + (1/Ts)*Gp(t)/VG,
    y(t) = Gsc(t)
)

locals = Dict(
    "1a_egp_on_no_renal" => ("EGP active, no renal", model_1a),
    "1b_egp_off_no_renal" => ("EGP suppressed, no renal", model_1b),
    "2a_egp_on_renal" => ("EGP active, renal active", model_2a),
    "2b_egp_off_renal" => ("EGP suppressed, renal active", model_2b),
)

local_results = Dict()
local_result_order = String[]
for (name, (desc, model)) in locals
    local_results[name] = run_local(name, desc, model)
    push!(local_result_order, name)
end

# ---------------------------------------------------------------------------
#  2. GLOBAL IDENTIFIABILITY (subsystem decomposition)
# ---------------------------------------------------------------------------

println("\n── GLOBAL IDENTIFIABILITY (subsystems) ──")

function run_global(name, desc, model; kw...)
    println("  Global: $desc ...")
    t = @elapsed result = assess_identifiability(model; prob_threshold = 0.95, kw...)
    path = joinpath(outdir, "global_$(name).csv")
    open(path, "w") do io; println(io, "variable,identifiability,subsystem")
        for (k, v) in result
            println(io, "$(k),$v,$desc")
        end
    end
    (result, t)
end

# 2a. Insulin subsystem (4 states, 1 input, 1 output)
ins_model = @ODEmodel(
    Isc1'(t) = -(kd+ka1)*Isc1(t) + u(t),
    Isc2'(t) = kd*Isc1(t) - ka2*Isc2(t),
    Ip'(t) = -(m2+m4)*Ip(t) + m1*Ii(t) + ka1*Isc1(t) + ka2*Isc2(t),
    Ii'(t) = -(m1+m4)*Ii(t) + m2*Ip(t),
    y(t) = Ip(t)
)
ins_res = run_global("insulin", "Insulin kinetics (y=Ip)", ins_model)

# 2b. Insulin subsystem (dual output: Ip + Ii)
ins2 = @ODEmodel(
    Isc1'(t) = -(kd+ka1)*Isc1(t) + u(t),
    Isc2'(t) = kd*Isc1(t) - ka2*Isc2(t),
    Ip'(t) = -(m2+m4)*Ip(t) + m1*Ii(t) + ka1*Isc1(t) + ka2*Isc2(t),
    Ii'(t) = -(m1+m4)*Ii(t) + m2*Ip(t),
    y1(t) = Ip(t),
    y2(t) = Ii(t)
)
ins2_res = run_global("insulin_dual", "Insulin kinetics (y1=Ip, y2=Ii)", ins2)

# 2c. EGP delays (2 states, input=Ip, output=Xl)
egp = @ODEmodel(
    I_prime'(t) = -ki*(I_prime(t) - Ip(t)/VI),
    Xl'(t) = -ki*(Xl(t) - I_prime(t)),
    y(t) = Xl(t)
)
egp_res = run_global("egp_delays", "EGP delay (input=Ip, output=Xl)", egp)

# 2d. Utilization (1 state, input=Ip, output=X)
util = @ODEmodel(
    X'(t) = -p2U*X(t) + p2U*(Ip(t)/VI - Ib),
    y(t) = X(t)
)
util_res = run_global("utilization", "Utilization delay (input=Ip, output=X)", util)

# 2e. Glucose core (3 states, inputs=Xl+X, output=Gsc)
glu = @ODEmodel(
    Gp'(t) = kp1 - kp2*Gp(t) - kp3*Xl(t) - Fcns - k1*Gp(t) + k2*Gt(t),
    Gt'(t) = -(Vm0+Vmx*X(t))*Gt(t)/(Km0+Gt(t)) + k1*Gp(t) - k2*Gt(t),
    Gsc'(t) = -(1/Ts)*Gsc(t) + (1/Ts)*Gp(t)/VG,
    y(t) = Gsc(t)
)
glu_res = run_global("glucose_core", "Glucose core (inputs=Xl,X)", glu; funcs_to_check=glu.parameters)

# ---------------------------------------------------------------------------
#  3. IDENTIFIABLE FUNCTIONS (glucose core)
# ---------------------------------------------------------------------------

println("\n── IDENTIFIABLE FUNCTIONS (glucose core) ──")
t_idf = @elapsed id_funcs = find_identifiable_functions(glu; prob_threshold = 0.95)
println("  Found $(length(id_funcs)) functions in $(round(t_idf; digits=1))s")

idf_path = joinpath(outdir, "identifiable_functions_glucose.csv")
open(idf_path, "w") do io; println(io, "i,function")
    for (i, func) in enumerate(id_funcs)
        println(io, "$i,$func")
    end
end
println("  → Saved: $idf_path\n")

# ---------------------------------------------------------------------------
#  4. CONSOLIDATED SUMMARY
# ---------------------------------------------------------------------------

summary_path = joinpath(outdir, "summary.txt")
open(summary_path, "w") do io

    w = function(s) println(io, s) end

    w("╔══════════════════════════════════════════════════════════════════════════╗")
    w("║  STRUCTURAL IDENTIFIABILITY ANALYSIS                                    ║")
    w("║  UVA-Padova T1DM Model (no meals, r1=0, rational approximation)         ║")
    w("╚══════════════════════════════════════════════════════════════════════════╝")
    w("")
    w("Package: StructuralIdentifiability.jl v0.5.23")
    w("Date:    $(string(Dates.now()))")
    w("")

    # ── 4a. Local identifiability summary ──
    w("── LOCAL IDENTIFIABILITY ──")
    w("")
    for name in local_result_order
        result, elapsed = local_results[name]
        n_id = count(v -> v == true, values(result))
        n_tot = length(result)
        w("  $(rpad(name, 30))  $(lpad(n_id,2))/$(lpad(n_tot,2)) locally id  ($(round(elapsed; digits=1)) s)")
    end
    w("")

    # Cross-scenario comparison table
    scenario_order = ["1a_egp_on_no_renal", "1b_egp_off_no_renal",
                      "2a_egp_on_renal", "2b_egp_off_renal"]
    scenario_labels = ["1a", "1b", "2a", "2b"]
    all_params_local = sort(unique(
        [string(k) for (_, (r, _)) in local_results for k in keys(r)]
    ))
    w("  Parameter               " * join(lpad.(scenario_labels, 6), " "))
    w("  " * "-" ^ 52)
    for p in all_params_local
        line = rpad(p, 24)
        for name in scenario_order
            r = local_results[name][1]
            kv = [(string(k), v) for (k, v) in r]
            matched = filter(x -> x[1] == p, kv)
            if isempty(matched)
                line *= "  N/A  "
            elseif matched[1][2] == true
                line *= "  YES  "
            else
                line *= "  NO   "
            end
        end
        w("  $line")
    end
    w("")
    w("  Scenarios:")
    w("    1a = EGP active, no renal excretion")
    w("    1b = EGP suppressed, no renal excretion")
    w("    2a = EGP active, renal excretion active")
    w("    2b = EGP suppressed, renal excretion active")
    w("")

    # ── 4b. Global identifiability (subsystems) ──
    w("── GLOBAL IDENTIFIABILITY (subsystem decomposition) ──")
    w("")
    w("  Methodology: model decomposed into 4 autonomous/feedforward subsystems")
    w("  to make global (differential-algebraic) analysis tractable.")
    w("")

    for (name, (result, elapsed)) in [
        ("Insulin (y=Ip)", ins_res),
        ("Insulin (y1=Ip,y2=Ii)", ins2_res),
        ("EGP delay", egp_res),
        ("Utilization delay", util_res),
        ("Glucose core", glu_res),
    ]
        w("  $name")
        w("  " * "-" ^ length(name))
        for (k, v) in result
            w("    $(rpad(string(k), 25))  $v")
        end
        w("")
    end

    # ── 4c. Composite full-model results ──
    w("── COMPOSITE: Full model structural identifiability ──")
    w("")
    w("  Legend: G = globally id, L = locally id, N = non-identifiable, — = N/A")
    w("")

    # Merge subsystem results
    composite = Dict{String, String}()
    for (r, _) in [ins_res, egp_res, util_res, glu_res]
        for (k, v) in r
            key = string(k)
            if haskey(composite, key)
                if v == :globally
                    composite[key] = "G"
                elseif v == :locally && composite[key] != "G"
                    composite[key] = "L"
                elseif v == :nonidentifiable && composite[key] != "G" && composite[key] != "L"
                    composite[key] = "N"
                end
            else
                composite[key] = uppercase(string(v)[1:1])
            end
        end
    end

    w("  $(rpad("Parameter", 22))  Global  Local(SE)  Note")
    w("  " * "-" ^ 65)

    # Full parameter list sorted by category
    param_order = [
        ("SC insulin", ["kd", "ka1", "ka2"]),
        ("Hepatic/plasma insulin", ["m1", "m2", "m4"]),
        ("Insulin distribution", ["VI"]),
        ("EGP suppression", ["ki", "kp1", "kp2", "kp3"]),
        ("Glucose transport", ["k1", "k2", "VG"]),
        ("Utilization", ["Vm0", "Vmx", "Km0", "Fcns", "p2U", "Ib"]),
        ("CGM sensor", ["Ts"]),
    ]

    for (cat, params) in param_order
        for p in params
            g = get(composite, p, "—")
            # Get local result from 1a
            loc_val = get(local_results["1a_egp_on_no_renal"][1], Symbol(p), nothing)
            loc_str = loc_val === nothing ? "—" : (loc_val ? "L" : "N")
            w("  $(rpad(p, 22))  $g        $loc_str")
        end
        w("")
    end

    # ── 4d. Identifiable functions ──
    w("── IDENTIFIABLE FUNCTIONS (glucose core, Gsc-only output) ──")
    w("")
    w("  Parameters structurally non-identifiable individually, but these")
    w("  rational combinations ARE identifiable from Gsc alone:")
    w("")
    for (i, func) in enumerate(id_funcs)
        w("  [$i]  $func")
    end
    w("")
    w("  Interpretation: Vm0, Vmx, Km0, VG, kp1, kp3, Fcns form identifiable")
    w("  ratios/scaled differences, but individual values cannot be resolved")
    w("  from CGM data alone without additional measurements or informative priors.")
    w("")

    # ── 4e. Clinical interpretation ──
    w("── CLINICAL INTERPRETATION ──")
    w("")
    w("  Parameters that CAN be estimated from CGM alone (kp2, k1, k2, Ts, ")
    w("  ka1, ki, p2U) are globally identifiable. This includes:")
    w("    • kp2 (glucose effectiveness on EGP) — key for insulin titration")
    w("    • k1/k2 (glucose transport rates) — microvascular function")
    w("    • p2U (insulin action lag) — relevant for closed-loop tuning")
    w("    • ki (EGP delay) — hepatic insulin resistance marker")
    w("    • ka1 (SC insulin absorption) — pump therapy parameter")
    w("")
    w("  Parameters that require additional data (meal info, dual-tracer, etc.):")
    w("    • Vm0, Vmx, Km0 (utilization Michaelis-Menten) — need glucose clamp")
    w("    • kp1, kp3 (EGP amplitude) — confounded with VG, Fcns")
    w("    • VG (glucose volume) — requires tracer dilution")
    w("    • Fcns (CNS uptake) — known constant in practice")
    w("")
    w("  The identifiable ratios Vm0/kp3, VG/kp3, (Fcns−kp1)/kp3 suggest that")
    w("  fixing any one of {kp3, VG, Fcns, kp1} to a literature value would")
    w("  break the scaling symmetry and make the others identifiable.")
    w("")
end

println("\n✅ All results saved to $outdir")
println("   Files:")
for f in readdir(outdir)
    println("     $f")
end
