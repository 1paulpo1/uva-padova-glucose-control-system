using StructuralIdentifiability
result = try
    m = @ODEmodel(
    I_prime'(t) = -ki * (I_prime(t) - Ip(t) / VI),
    Xl'(t) = -ki * (Xl(t) - I_prime(t)),
    X'(t) = -p2U * X(t) + p2U * (Ip(t) / VI - Ib),
    Gp'(t) = kp1 - kp2 * Gp(t) - kp3 * Xl(t) - Fcns - k1 * Gp(t) + k2 * Gt(t),
    Gt'(t) = -(Vm0 + Vmx * X(t)) * Gt(t) / (Km0 + Gt(t)) + k1 * Gp(t) - k2 * Gt(t),
    Gsc'(t) = -(1 / Ts) * Gsc(t) + (1 / Ts) * Gp(t) / VG,
    y(t) = Gsc(t)
)

    find_identifiable_functions(m)
catch e
    ["CRASHED: $(typeof(e).name.name)"]
end
open("C:\Users\beleg\Documents\uva-padova-glucose-control-system\validation\identifiability\b1_glu_delays_functions.csv", "w") do io
    println(io, "i,function,subsystem")
    for (i, fn) in enumerate(result)
        println(io, "$i,$fn,subprocess")
    end
end
