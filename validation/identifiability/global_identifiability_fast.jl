using StructuralIdentifiability

outdir = @__DIR__

# ═══════════════════════════════════════════════════════════════════════════════
#  Fast global identifiability for Model 1A (EGP active, no renal)
#  Strategy:
#    1. prob_threshold = 0.95  (faster cutoff)
#    2. funcs_to_check = only parameters, skip states (slimmer computation)
# ═══════════════════════════════════════════════════════════════════════════════

model = @ODEmodel(
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

println("═" ^ 72)
println("  Model 1A: EGP active, no renal excretion")
println("  Parameters: $(length(model.parameters)) → $(join(model.parameters, ", "))")
println("═" ^ 72)

# ── Run global identifiability (params only, faster) ──
println("\n  assess_identifiability (funcs_to_check = parameters, prob_threshold = 0.95)")
println("  ─" ^ 36)

t_start = time()
result = assess_identifiability(model;
    funcs_to_check = model.parameters,
    prob_threshold = 0.95)
elapsed = time() - t_start

println("\n  ⏱  $(round(elapsed, digits=1)) s\n")

# Counts
n_glob = count(v -> v == :globally, values(result))
n_loc  = count(v -> v == :locally, values(result))
n_non  = count(v -> v == :nonidentifiable, values(result))
println("  Globally identifiable:      $n_glob")
println("  Locally identifiable:       $n_loc")
println("  Non-identifiable:           $n_non")
println()

# Print detailed results
for (k, v) in result
    label = v == :globally ? "GLOBAL" :
            v == :locally  ? "LOCAL " : "NON-ID"
    println("    $label  →  $(rpad(string(k), 22))  $v")
end

# Save
csv_path = joinpath(outdir, "global_identifiability_1a_params_only.csv")
open(csv_path, "w") do io
    println(io, "parameter,identifiability")
    for (k, v) in result
        println(io, "$(string(k)),$v")
    end
end
println("\n  → Saved: $csv_path")

# ── Identifiable functions ──
println("\n" ^ 2)
println("═" ^ 72)
println("  find_identifiable_functions")
println("═" ^ 72)

t_start = time()
id_funcs = find_identifiable_functions(model; prob_threshold = 0.95)
elapsed_idf = time() - t_start

println("\n  ⏱  $(round(elapsed_idf, digits=1)) s\n")

fcsv_path = joinpath(outdir, "identifiable_functions_1a.csv")
open(fcsv_path, "w") do io
    println(io, "function,identifiability")
    for (func, status) in id_funcs
        println(io, "$(string(func)),$status")
    end
end

for (func, status) in id_funcs
    println("    $(rpad(string(func), 40))  →  $status")
end
println("\n  → Saved: $fcsv_path")

println("\n✅ Done.")
