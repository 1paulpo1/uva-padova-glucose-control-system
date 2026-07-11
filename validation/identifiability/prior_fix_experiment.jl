using StructuralIdentifiability, Dates, Random

const outdir = @__DIR__
const SCRIPT_DIR = @__DIR__

const DEFAULT_POP = Dict{Symbol, Float64}(
    :BW   => 75.0,  :f    => 0.9,   :VG   => 1.5,   :VI   => 16.0,  :Ts   => 10.0,
    :k1   => 0.065, :k2   => 0.079, :Ib   => 10.0,
    :kgri => 0.03,  :kmax => 0.04,  :kabs => 0.2,
    :Vm0  => 2.5,   :Vmx  => 0.05,  :Km0  => 8.0,
    :kp1  => 3.3,   :kp2  => 0.008, :kp3  => 0.05,   :Fcns => 1.0,
    :ka1  => 0.002, :ka2  => 0.0005,:kd   => 0.01,
    :m1   => 0.2,   :m2   => 0.3,   :m4   => 0.15,
    :ki   => 0.005, :p2U  => 0.03,
)

const HIERARCHY_ORDER = [
    :BW,   :f,    :VG,  :VI,  :Ts,
    :k1,   :k2,   :Ib,
    :kgri, :kmax, :kabs,
    :Vm0,  :Vmx,  :Km0,
    :kp1,  :kp2,  :kp3, :Fcns,
    :ka1,  :ka2,  :kd,  :m1, :m2, :m4,
    :ki,   :p2U,
]

# ═══════════════════════════════════════════════════════════════════════════════
#  Model builder — generates @ODEmodel code and evals it
#  Uses file-based approach for robustness: write .jl, include it
# ═══════════════════════════════════════════════════════════════════════════════

function build_model_code!(free_vec::Vector{Symbol}; pop::Dict{Symbol,Float64}=DEFAULT_POP)::Vector{String}
    free_set = Set(free_vec)
    val(p::Symbol) = p in free_set ? string(p) : string(pop[p])
    return [
        "using StructuralIdentifiability",
        "m = @ODEmodel(",
        "    Qsto1'(t)  = -$(val(:kgri)) * Qsto1(t) + u_meal(t),",
        "    Qsto2'(t)  = -$(val(:kmax)) * Qsto2(t) + $(val(:kgri)) * Qsto1(t),",
        "    Qgut'(t)   = -$(val(:kabs)) * Qgut(t) + $(val(:kmax)) * Qsto2(t),",
        "    Isc1'(t)   = -($(val(:kd)) + $(val(:ka1))) * Isc1(t) + u(t),",
        "    Isc2'(t)   = $(val(:kd)) * Isc1(t) - $(val(:ka2)) * Isc2(t),",
        "    Ip'(t)     = -($(val(:m2)) + $(val(:m4))) * Ip(t) + $(val(:m1)) * Ii(t) + $(val(:ka1)) * Isc1(t) + $(val(:ka2)) * Isc2(t),",
        "    Ii'(t)     = -($(val(:m1)) + $(val(:m4))) * Ii(t) + $(val(:m2)) * Ip(t),",
        "    I_prime'(t)= -$(val(:ki)) * (I_prime(t) - Ip(t) / $(val(:VI))),",
        "    Xl'(t)     = -$(val(:ki)) * (Xl(t) - I_prime(t)),",
        "    X'(t)      = -$(val(:p2U)) * X(t) + $(val(:p2U)) * (Ip(t) / $(val(:VI)) - $(val(:Ib))),",
        "    Gp'(t)     = $(val(:kp1)) - $(val(:kp2)) * Gp(t) - $(val(:kp3)) * Xl(t) - $(val(:Fcns)) - $(val(:k1)) * Gp(t) + $(val(:k2)) * Gt(t) + $(val(:f)) * $(val(:kabs)) * Qgut(t) / $(val(:BW)),",
        "    Gt'(t)     = -($(val(:Vm0)) + $(val(:Vmx)) * X(t)) * Gt(t) / ($(val(:Km0)) + Gt(t)) + $(val(:k1)) * Gp(t) - $(val(:k2)) * Gt(t),",
        "    Gsc'(t)    = -(1 / $(val(:Ts))) * Gsc(t) + (1 / $(val(:Ts))) * Gp(t) / $(val(:VG)),",
        "    y(t)       = Gsc(t)",
        ")",
    ]
end

function build_model_file(free_vec::Vector{Symbol}, filepath::String; pop::Dict{Symbol,Float64}=DEFAULT_POP)
    write(filepath, join(build_model_code!(free_vec; pop=pop), "\n"))
end

# ═══════════════════════════════════════════════════════════════════════════════
#  In-process evaluation with fixed random seed for determinism.
#  assess_local_identifiability is probabilistic (Monte Carlo), so using
#  a fixed seed ensures identical results across repeated runs.
#  QQMPolyRingElem keys are converted to Symbol for easy consumption.
# ═══════════════════════════════════════════════════════════════════════════════

function build_model_code_seeded!(free_vec::Vector{Symbol}; pop::Dict{Symbol,Float64}=DEFAULT_POP)::Vector{String}
    code = build_model_code!(free_vec; pop=pop)
    push!(code, "Random.seed!(42)")
    push!(code, "assess_local_identifiability(m; prob_threshold=0.99)")
    return code
end

function run_local_inprocess(free_vec::Vector{Symbol}; pop::Dict{Symbol,Float64}=DEFAULT_POP)::Dict{Symbol,Bool}
    path = tempname() * ".jl"
    write(path, join(build_model_code_seeded!(free_vec; pop=pop), "\n"))
    raw = try
        include(path)
    finally
        rm(path, force=true)
    end
    # include returns the value of the last expression (the Dict{QQMPolyRingElem, Bool})
    d = Dict{Symbol, Bool}()
    for (k, v) in raw
        d[Symbol(string(k))] = v
    end
    return d
end

function run_local(free_vec::Vector{Symbol}; pop::Dict{Symbol,Float64}=DEFAULT_POP)::Dict{Symbol,Bool}
    return run_local_inprocess(free_vec; pop=pop)
end

# ═══════════════════════════════════════════════════════════════════════════════

list_L(d, free) = [p for p in free if get(d, p, false)]
list_N(d, free) = [p for p in free if !get(d, p, false)]
count_L(d, free) = sum(1 for p in free if get(d, p, false))
fmt_sec(s) = s < 60 ? "$(round(s, digits=1)) s" : "$(round(s/60, digits=1)) min"

# ═══════════════════════════════════════════════════════════════════════════════
#  HIERARCHY STRATEGY (sequential, in-process, deterministic)
# ═══════════════════════════════════════════════════════════════════════════════

function hierarchy_search(; pop::Dict{Symbol,Float64}=DEFAULT_POP, order::Vector{Symbol}=HIERARCHY_ORDER)
    println("\n" * "─"^72)
    println("  HIERARCHY STRATEGY")
    println("─"^72)

    all_params = sort!(collect(keys(pop)))
    free = Vector{Symbol}(all_params)
    fixed = Symbol[]
    log = []

    print("  Iter 0 (baseline, all free)... ")
    t0 = @elapsed r0 = run_local(free; pop=pop)
    total0 = count_L(r0, free)
    push!(log, (0, nothing, length(free), total0, 0, t0, list_L(r0, free), list_N(r0, free)))
    println("$total0/$(length(free)) L  $(fmt_sec(t0))")

    for (iter, p) in enumerate(order)
        p in free || continue
        free = filter(q -> q != p, free)
        push!(fixed, p)
        prev = total0

        print("  Iter $iter: fix $p... ")
        flush(stdout)
        t_loc = @elapsed r_loc = run_local(free; pop=pop)
        total0 = count_L(r_loc, free)
        newly = total0 - prev
        push!(log, (iter, p, length(free), total0, newly, t_loc, list_L(r_loc, free), list_N(r_loc, free)))
        println("$total0/$(length(free)) L ($(newly >= 0 ? "+" : "")$newly)  $(fmt_sec(t_loc))")

        total0 == length(free) && (println("  ✅ All L!"); break)
    end
    return log, fixed
end

# ═══════════════════════════════════════════════════════════════════════════════
#  GREEDY STRATEGY (sequential, in-process, deterministic)
#  At each step, test each N candidate individually and pick the one
#  that unlocks the most new L params.
# ═══════════════════════════════════════════════════════════════════════════════

function greedy_search(; pop::Dict{Symbol,Float64}=DEFAULT_POP, run_all::Bool=false)
    println("\n" * "─"^72)
    println("  GREEDY STRATEGY (sequential, deterministic)")
    println("─"^72)

    all_params = sort!(collect(keys(pop)))
    free = Vector{Symbol}(all_params)
    fixed = Symbol[]
    log = []

    print("  Iter 0 (baseline, all free)... ")
    flush(stdout)
    t0 = @elapsed r0 = run_local(free; pop=pop)
    total0 = count_L(r0, free)
    push!(log, (0, nothing, length(free), total0, 0, t0, list_L(r0, free), list_N(r0, free)))
    println("$total0/$(length(free)) L  $(fmt_sec(t0))")

    iter = 0
    while true
        iter += 1
        Ns = list_N(r0, free)

        isempty(Ns) && (println("  ✅ All L!"); break)
        if length(Ns) == 1
            println("  ⛔ Last param ($(Ns[1])) remains N — no more candidates to try")
            push!(log, (iter, Ns[1], 1, 0, 0, 0.0, Symbol[], Ns))
            break
        end

        n_cand = length(Ns)
        println("  Iter $iter: testing $n_cand candidates sequentially...")

        # Test each candidate individually
        cand = []
        for (ci, p) in enumerate(Ns)
            print("    [$ci/$n_cand] test $p... ")
            flush(stdout)
            trial_free = filter(q -> q != p, free)
            t_trial = @elapsed tr = run_local(trial_free; pop=pop)
            nL = count_L(tr, trial_free)
            push!(cand, (p, nL, tr))
            println("$nL/$(length(trial_free)) L  $(fmt_sec(t_trial))")
        end

        sort!(cand, by=x->(-x[2]))
        best_p, best_nL, best_result = cand[1]

        was_L = get(r0, best_p, false)
        newly = best_nL - (total0 - (was_L ? 1 : 0))

        if newly <= 0
            n_str = join(string.(Ns), ", ")
            println("  ⛔ No progress — remaining N: $n_str")
            break
        end

        # Commit
        free = filter(q -> q != best_p, free)
        push!(fixed, best_p)
        total0 = best_nL
        r0 = best_result
        push!(log, (iter, best_p, length(free), total0, newly, 0.0, list_L(r0, free), list_N(r0, free)))
        println("  ✓ Fixed $best_p → $total0/$(length(free)) L (+$newly)")

        total0 == length(free) && (println("  ✅ All L!"); break)
    end
    return log, fixed
end

# ═══════════════════════════════════════════════════════════════════════════════
#  REPORT
# ═══════════════════════════════════════════════════════════════════════════════

function make_fix_dict(log)
    d = Dict{Symbol, Tuple{Int,Int}}()
    for row in log
        it, fp, _, _, new, _, _, _ = row
        fp === nothing && continue
        d[Symbol(string(fp))] = (it, new)
    end
    return d
end

function write_log_csv(path, log)
    open(path, "w") do io
        println(io, "iteration,fixed_param,remaining,total_L,newly_L,time_s,L_list,N_list")
        for row in log
            it, fp, rem, tot, new, t, Ls, Ns = row
            fp_s = fp === nothing ? "-" : string(fp)
            Ls_s = join(string.(Ls), ";")
            Ns_s = join(string.(Ns), ";")
            println(io, "$it,$fp_s,$rem,$tot,$new,$t,$Ls_s,$Ns_s")
        end
    end
end

function generate_report(g_log, g_fixed, h_log, h_fixed; all_params::Vector{Symbol}=sort!(collect(keys(DEFAULT_POP))))
    csv_g   = joinpath(outdir, "prior_fix_log_greedy.csv")
    csv_h   = joinpath(outdir, "prior_fix_log_hierarchy.csv")
    csv_comp = joinpath(outdir, "prior_fix_comparison.csv")
    rpt     = joinpath(outdir, "prior_fix_report.txt")

    g_fd = make_fix_dict(g_log)
    h_fd = make_fix_dict(h_log)
    g_last = last(g_log)
    h_last = last(h_log)

    write_log_csv(csv_g, g_log)
    write_log_csv(csv_h, h_log)

    # Comparison CSV
    g_Lfin = Set(g_last[7])
    h_Lfin = Set(h_last[7])
    open(csv_comp, "w") do io
        println(io, "param,greedy_iter,greedy_newL,hierarchy_iter,hierarchy_newL,g_status,h_status")
        for p in all_params
            g_info = get(g_fd, p, nothing)
            h_info = get(h_fd, p, nothing)
            g_iter = g_info === nothing ? "free" : string(g_info[1])
            g_new  = g_info === nothing ? "0" : string(g_info[2])
            h_iter = h_info === nothing ? "free" : string(h_info[1])
            h_new  = h_info === nothing ? "0" : string(h_info[2])
            g_st = p in g_fixed ? "fixed" : (p in g_Lfin ? "L" : "N")
            h_st = p in h_fixed ? "fixed" : (p in h_Lfin ? "L" : "N")
            println(io, "$p,$g_iter,$g_new,$h_iter,$h_new,$g_st,$h_st")
        end
    end

    # Text report
    open(rpt, "w") do io
        w(s) = println(io, s)
        w("╔══════════════════════════════════════════════════════════════════════════╗")
        w("║  PRIOR FIXING EXPERIMENT                                               ║")
        w("║  UVA-Padova T1DM — sequential parameter fixing with population priors  ║")
        w("║  $(Dates.now())                                                          ║")
        w("╚══════════════════════════════════════════════════════════════════════════╝")
        w("")
        w("── DESIGN ──")
        w("  Model: full 13-state, 27-param, y=Gsc, inputs=u, u_meal")
        w("  Method: Local identifiability (p=0.99) after each parameter fix")
        w("  Evaluations: in-process with fixed random seed (deterministic)")
        w("")
        w("  Greedy:    at each step fix the N param that unlocks the most new Ls")
        w("  Hierarchy: fix in physiological order")
        w("")

        w("══ SUMMARY ══")
        w("")
        g_Nf = g_last[8]; h_Nf = h_last[8]
        w("  Greedy:   $(length(g_fixed)) fixed  → $(length(g_last[7]))/$(g_last[3]) L  ($(length(g_Nf)) N)")
        w("  Hierarchy: $(length(h_fixed)) fixed → $(length(h_last[7]))/$(h_last[3]) L  ($(length(h_Nf)) N)")
        w("")

        # Greedy log
        w("── GREEDY FULL LOG ──")
        w("  $(rpad("Iter",5)) $(rpad("Fixed",10)) $(rpad("Rem",4)) $(rpad("L",4)) +New  N_list")
        w("  " * "-"^80)
        for row in g_log
            it, fp, rem, tot, new, t, Ls, Ns = row
            fp_s = fp === nothing ? "—" : string(fp)
            w("  $(rpad(string(it),5)) $(rpad(fp_s,10)) $(rpad(string(rem),4)) $(rpad(string(tot),4)) +$(rpad(string(new),3))  $(join(string.(Ns), ", "))")
        end
        w("")

        # Hierarchy log
        w("── HIERARCHY FULL LOG ──")
        w("  $(rpad("Iter",5)) $(rpad("Fixed",10)) $(rpad("Rem",4)) $(rpad("L",4)) +New  Time     N_list")
        w("  " * "-"^80)
        for row in h_log
            it, fp, rem, tot, new, t, Ls, Ns = row
            fp_s = fp === nothing ? "—" : string(fp)
            t_s = fmt_sec(t)
            w("  $(rpad(string(it),5)) $(rpad(fp_s,10)) $(rpad(string(rem),4)) $(rpad(string(tot),4)) +$(rpad(string(new),3))  $(rpad(t_s,8)) $(join(string.(Ns), ", "))")
        end
        w("")

        # Side-by-side
        w("── PARAMETER COMPARISON ──")
        w("  $(rpad("Param",8))  $(rpad("Greedy",28))  $(rpad("Hierarchy",28))")
        w("  " * "-"^67)
        for p in all_params
            gi = get(g_fd, p, nothing)
            hi = get(h_fd, p, nothing)
            gs = gi === nothing ? (p in g_Lfin ? "L" : "N") : "fixed@$(gi[1]) (+$(gi[2]))"
            hs = hi === nothing ? (p in h_Lfin ? "L" : "N") : "fixed@$(hi[1]) (+$(hi[2]))"
            w("  $(rpad(string(p),8))  $(rpad(gs,28))  $(rpad(hs,28))")
        end
        w("")

        w("── PRIOR SETS ──")
        w("")
        w("  Greedy order ($(length(g_fixed))):")
        for p in g_fixed
            gi = get(g_fd, p, (0,0))
            w("    $(rpad(string(p),8)) iter $(gi[1])  +$(gi[2]) new L")
        end
        w("")
        w("  Hierarchy order ($(length(h_fixed))):")
        for p in h_fixed
            hi = get(h_fd, p, (0,0))
            w("    $(rpad(string(p),8)) iter $(hi[1])  +$(hi[2]) new L")
        end
        w("")

        w("── FILES ──")
        foreach(f -> w("  $f"), ["prior_fix_log_greedy.csv", "prior_fix_log_hierarchy.csv", "prior_fix_comparison.csv"])
        w("")
    end
    println("\n✅ Report: $rpt")
    return rpt
end

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN (call this explicitly; not executed on include())
# ═══════════════════════════════════════════════════════════════════════════════

function main()
    println("╔" * "═"^70 * "╗")
    println("║  PRIOR FIXING EXPERIMENT                                              ║")
    println("║  UVA-Padova T1DM — sequential parameter fixing with population priors  ║")
    println("╚" * "═"^70 * "╝")

    println("  Sequential in-process evaluation (deterministic, seed=42)")

    println("\n  Using DEFAULT_POP values")
    h_log, h_fixed = hierarchy_search(pop=DEFAULT_POP)
    g_log, g_fixed = greedy_search(pop=DEFAULT_POP)
    generate_report(g_log, g_fixed, h_log, h_fixed)
    println("\nDone.")
end

# Only auto-run when this file is the entry script, not on include()
if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
