# POP Sensitivity: test whether hierarchy search changes when
# perturbing numeric prior values (non-generic algebraic collision hypothesis).
# Each parameter is perturbed by a small random factor so no "round numbers" remain.

include("prior_fix_experiment.jl")

const PERT_SEED = 123  # fixed seed for the perturbation itself

# ── Define perturbed POP ──
rng = MersenneTwister(PERT_SEED)
raw = Dict{Symbol,Float64}()
for (k, v) in DEFAULT_POP
    # uniform factor in [0.97, 1.03]
    ε = 1.0 + (rand(rng) - 0.5) * 0.06
    raw[k] = round(v * ε; digits=6)
end

POP_PERT = Dict{Symbol,Float64}(
    :BW   => raw[:BW],
    :f    => raw[:f],
    :VG   => raw[:VG],
    :VI   => raw[:VI],
    :Ts   => raw[:Ts],
    :k1   => raw[:k1],
    :k2   => raw[:k2],
    :Ib   => raw[:Ib],
    :kgri => raw[:kgri],
    :kmax => raw[:kmax],
    :kabs => raw[:kabs],
    :Vm0  => raw[:Vm0],
    :Vmx  => raw[:Vmx],
    :Km0  => raw[:Km0],
    :kp1  => raw[:kp1],
    :kp2  => raw[:kp2],
    :kp3  => raw[:kp3],
    :Fcns => raw[:Fcns],
    :ka1  => raw[:ka1],
    :ka2  => raw[:ka2],
    :kd   => raw[:kd],
    :m1   => raw[:m1],
    :m2   => raw[:m2],
    :m4   => raw[:m4],
    :ki   => raw[:ki],
    :p2U  => raw[:p2U],
)

println("╔" * "═"^70 * "╗")
println("║  POP SENSITIVITY ANALYSIS                                              ║")
println("║  Deterministic hierarchy with original vs perturbed POP values         ║")
println("╚" * "═"^70 * "╝")
println()

println("DEFAULT_POP values:")
for k in sort!(collect(keys(DEFAULT_POP)))
    println("  $k = $(DEFAULT_POP[k])")
end
println()
println("POP_PERT values:")
for k in sort!(collect(keys(POP_PERT)))
    println("  $k = $(POP_PERT[k])")
end
println()

# ── Run 1: original POP ──
println("═══ RUN 1: DEFAULT_POP (original values) ═══")
h_log_orig, h_fixed_orig = hierarchy_search(pop=DEFAULT_POP)

# ── Run 2: perturbed POP ──
println("\n═══ RUN 2: POP_PERT (perturbed values) ═══")
h_log_pert, h_fixed_pert = hierarchy_search(pop=POP_PERT)

# ── Save logs ──
function save_hierarchy_csv(path, log, label)
    open(path, "w") do io
        println(io, "label,iteration,fixed_param,remaining,total_L,newly_L,time_s,N_list")
        for row in log
            it, fp, rem, tot, new, t, Ls, Ns = row
            fp_s = fp === nothing ? "-" : string(fp)
            Ns_s = join(string.(Ns), ";")
            println(io, "$label,$it,$fp_s,$rem,$tot,$new,$t,$Ns_s")
        end
    end
end

save_hierarchy_csv(joinpath(outdir, "pop_sens_orig.csv"), h_log_orig, "original")
save_hierarchy_csv(joinpath(outdir, "pop_sens_pert.csv"), h_log_pert, "perturbed")

# ── Compare ──
last_orig = last(h_log_orig)
last_pert = last(h_log_pert)

println("\n" * "═"^70)
println("  COMPARISON")
println("═"^70)
println()
println("  $(rpad("",30))  Original   Perturbed")
println("  " * "-"^55)
println("  $(rpad("Iterations",30))  $(length(h_log_orig))         $(length(h_log_pert))")
println("  $(rpad("Params fixed",30))  $(length(h_fixed_orig))         $(length(h_fixed_pert))")
println("  $(rpad("Final L/total",30))  $(last_orig[4])/$(last_orig[3])        $(last_pert[4])/$(last_pert[3])")
println("  $(rpad("Final N count",30))  $(length(last_orig[8]))         $(length(last_pert[8]))")
println()

# Per-step comparison
println("  Iteration-by-iteration ΔL:")
println("  $(rpad("Iter",5)) $(rpad("Fixed",10)) $(rpad("Orig L",8)) $(rpad("Pert L",8)) $(rpad("Δ",6))  Match?")
println("  " * "-"^65)

# Create lookup by iter
orig_by_iter = Dict{Int, Tuple}()
for row in h_log_orig
    it, fp, rem, tot, new, t, Ls, Ns = row
    orig_by_iter[it] = (fp, tot, rem)
end

pert_by_iter = Dict{Int, Tuple}()
for row in h_log_pert
    it, fp, rem, tot, new, t, Ls, Ns = row
    pert_by_iter[it] = (fp, tot, rem)
end

max_iter = max(maximum(keys(orig_by_iter)), maximum(keys(pert_by_iter)))
all_same = true
for it in 0:max_iter
    o = get(orig_by_iter, it, nothing)
    p = get(pert_by_iter, it, nothing)
    match = false
    if o !== nothing && p !== nothing
        fp_o, tot_o, rem_o = o
        fp_p, tot_p, rem_p = p
        # Check if the same param was fixed at this iteration
        same_fp = (fp_o === nothing && fp_p === nothing) || (fp_o !== nothing && fp_p !== nothing && fp_o == fp_p)
        same_tot = tot_o == tot_p
        match = same_fp && same_tot
        if !match
            all_same = false
        end
        fp_s = fp_o === nothing ? "—" : string(fp_o)
        Δ = tot_o - tot_p
        Δ_str = Δ >= 0 ? "+$Δ" : string(Δ)
        println("  $(rpad(string(it),5)) $(rpad(fp_s,10)) $(rpad(string(tot_o),8)) $(rpad(string(tot_p),8)) $(rpad(Δ_str,6))  $(match ? "✓" : "✗")")
    elseif o !== nothing
        fp_o, tot_o, rem_o = o
        fp_s = fp_o === nothing ? "—" : string(fp_o)
        println("  $(rpad(string(it),5)) $(rpad(fp_s,10)) $(rpad(string(tot_o),8)) $(rpad("—",8)) $(rpad("—",6))  ? (pert shorter)")
    else
        fp_p, tot_p, rem_p = p
        fp_s = fp_p === nothing ? "—" : string(fp_p)
        println("  $(rpad(string(it),5)) $(rpad("—",10)) $(rpad("—",8)) $(rpad(string(tot_p),8)) $(rpad("—",6))  ? (orig shorter)")
    end
end
println()
println("  Matching trajectory: $(all_same ? "YES — identical ΔL per step" : "NO — different ΔL pattern")")
println()

# ── Check for non-monotonicity ──
function count_dips(log)
    dips = 0
    prev_L = 0
    for row in log
        it, fp, rem, tot, new, t, Ls, Ns = row
        if it > 0 && tot < prev_L
            fp_s = fp === nothing ? "—" : string(fp)
            dips += 1
            println("    Dip at iter $it: fix $fp_s → L dropped from $prev_L to $tot ($(tot - prev_L))")
        end
        prev_L = tot
    end
    return dips
end

println()
println("  ── NON-MONOTONICITY ──")
println("  Original:")
orig_dips = count_dips(h_log_orig)
println("  Perturbed:")
pert_dips = count_dips(h_log_pert)
println()
if orig_dips == 0 && pert_dips == 0
    println("  ✅ No dips in either run — hierarchy is monotonic with deterministic seed")
    println("  → Previous ±1 oscillations were entirely a subprocess/seed artifact")
elseif orig_dips == 0 && pert_dips > 0
    println("  ⚠ Original monotonic, perturbed has dips → numeric collision confirmed")
elseif orig_dips > 0 && pert_dips == 0
    println("  ⚠ Perturbed monotonic, original has dips → numeric collision in original values")
else
    println("  ⚠ Both have dips, but may differ → partial numeric sensitivity")
end

println()
println("Files saved: pop_sens_orig.csv, pop_sens_pert.csv")
println("Done.")
