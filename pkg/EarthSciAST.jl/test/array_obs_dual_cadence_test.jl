# A ForwardDiff call that differentiates the state alone takes the observeds
# that read no state from the Float64 buffer (build.jl,
# `_obsext_cadence_from_f64!`).
#
# The const- and time-cadence materialized observeds (array_obs_materialize_test.jl)
# are cached per value type, so a Jacobian at a new `t` used to recompute them in
# Dual arithmetic, although they carry no derivative with respect to the state
# and the Float64 buffer already holds their values. This pins that such a call:
#   * gives the values of the Dual evaluation bit for bit, and its derivatives
#     (`==`: that evaluation can carry a -0.0 where this gives 0.0), against the
#     same build made to take the Dual path, and against the interpreter oracle;
#   * fills the Float64 buffer first when no Float64 call has;
#   * does not evaluate the time-cadence observeds in Dual arithmetic;
#   * still evaluates them in Dual when the parameters carry the derivative;
#   * allocates nothing in steady state.
using Test
using EarthSciAST
using ForwardDiff
include("testutils.jl")
const ESM_ADC = EarthSciAST

@testset "a state Jacobian takes stateless observeds from the Float64 buffer" begin
    N = 8
    isets = Dict("x" => ESM_ADC.IndexSet("interval"; size = N))
    rng = Dict("i" => ESM_ADC.IndexSetRef("x"))
    agg(body) = ESM_ADC.OpExpr(
        "faq", ESM_ADC.ASTExpr[];
        output_idx = Any["i"], ranges = copy(rng), expr_body = body
    )
    vars = Dict(
        "u" => ESM_ADC.ModelVariable(ESM_ADC.UnknownVariable; shape = ["x"]),
        "c" => ESM_ADC.ModelVariable(ESM_ADC.UnknownVariable; shape = ["x"]),
        "w" => ESM_ADC.ModelVariable(ESM_ADC.UnknownVariable; shape = ["x"]),
        "d" => ESM_ADC.ModelVariable(ESM_ADC.UnknownVariable; shape = ["x"]),
        "k" => ESM_ADC.ModelVariable(ESM_ADC.ParameterVariable; default = 0.25),
    )
    eqs = [
        # Const cadence: `k` only. Evaluated in Dual arithmetic, the minus gives
        # -0.0 derivatives.
        ESM_ADC.Equation(
            _v("c"), agg(_op("*", _op("-", _v("k")), _op("sin", _v("i"))))
        ),
        # Time cadence: `t`, the forcing `F` and `c`, no state.
        ESM_ADC.Equation(
            _v("w"), agg(
                _op("+", _op("*", _v("t"), _idx("F", _v("i"))), _idx("c", _v("i")))
            )
        ),
        # State dependent.
        ESM_ADC.Equation(
            _v("d"), agg(
                _op(
                    "+", _op("*", _idx("c", _v("i")), _idx("u", _v("i"))),
                    _op("*", _idx("w", _v("i")), _idx("u", _v("i")), _idx("u", _v("i")))
                )
            )
        ),
        ESM_ADC.Equation(
            agg(_Didx("u", _v("i"))),
            agg(_op("-", _idx("d", _v("i")), _op("*", _idx("w", _v("i")), _idx("u", _v("i")))))
        ),
    ]
    model = ESM_ADC.Model(vars, eqs)
    ics = Dict("u[$j]" => 0.5 + 0.1j for j in 1:N)
    F = [1.0 + 0.5j for j in 1:N]
    kw = (; index_sets = isets, initial_conditions = ics, param_arrays = Dict("F" => F))
    fac = ESM_ADC._build_evaluator_impl(model; kw...)
    ref = ESM_ADC._build_evaluator_impl(model; kw...)
    inl = ESM_ADC._build_evaluator_impl(model; compiler = :interpreter, kw...)
    @test fac[6].n_mat_const_levels == 1
    @test fac[6].n_mat_time_levels == 1
    f!, u0, p, _, _, _ = fac
    g! = ref[1]
    fi! = inl[1]

    # One Dual call over every state, seeded as a Jacobian is.
    D = ForwardDiff.Dual{Nothing, Float64, N}
    seed(i) = ForwardDiff.Partials{N, Float64}(ntuple(j -> Float64(i == j), N))
    ud = D[D(u0[i], seed(i)) for i in 1:N]
    # The same parameters as Duals with zero derivatives: the value type is the
    # same, but a `p` that carries a Dual takes the Dual evaluation of every
    # observed, which is what a state Jacobian ran before.
    zp(p) = NamedTuple{keys(p)}(map(v -> D(v, zero(ForwardDiff.Partials{N, Float64})), values(p)))
    val(y) = ForwardDiff.value.(y)
    der(y) = ForwardDiff.partials.(y)
    dual_call(h!, pp, tt) = (y = zeros(D, N); h!(y, ud, pp, tt); y)
    function agrees(pp, tt)
        y = dual_call(f!, pp, tt)
        yr = dual_call(g!, zp(pp), tt)
        yi = dual_call(fi!, pp, tt)
        return all(val(y) .=== val(yr)) && der(y) == der(yr) &&
            val(y) == val(yi) && der(y) == der(yi)
    end

    du = similar(u0)
    p2 = typeof(p)(map(x -> 3.0 * x, values(p)))
    # After a Float64 call at the same (p, t): the Float64 buffer is fresh.
    f!(du, u0, p, 1.5)
    @test agrees(p, 1.5)
    @test agrees(p, 1.5)
    # A new t with no Float64 call: the Float64 buffer is filled first.
    @test agrees(p, 2.5)
    # The parameters move: the const tier too.
    @test agrees(p2, 2.5)
    @test agrees(p, 2.5)

    # The time-cadence observeds are not evaluated in Dual arithmetic. A write
    # to `F` WITHOUT the refresh notification is invisible to every cached tier
    # by design; here it tells the two apart: a Dual call at the t of a
    # Float64 call sees the `w` that call computed, an evaluation would see the
    # new `F`.
    f!(du, u0, p, 3.5)
    du35 = copy(du)
    F2 = F[2]
    F[2] = 99.0
    y = dual_call(f!, p, 3.5)
    @test val(y) == du35
    @test val(dual_call(fi!, p, 3.5))[2] != du35[2]
    F[2] = F2
    ESM_ADC.notify_forcing_refresh!(F)
    @test agrees(p, 3.5)

    # Parameters that carry the derivative: these observeds have one, and are
    # evaluated in Dual arithmetic as before.
    D1 = ForwardDiff.Dual{Nothing, Float64, 1}
    pk = NamedTuple{keys(p)}((D1(p.k, ForwardDiff.Partials((1.0,))),))
    y1 = zeros(D1, N)
    yi1 = zeros(D1, N)
    f!(y1, u0, pk, 1.5)
    fi!(yi1, u0, pk, 1.5)
    @test val(y1) == val(yi1)
    @test der(y1) == der(yi1)
    @test any(d -> d[1] != 0, der(y1))

    # Steady state allocates nothing, at the t of the last call and at a new one.
    if VERSION >= v"1.12"
        y = zeros(D, N)
        f!(y, ud, p, 5.5)
        f!(y, ud, p, 6.5)
        @test (@allocated f!(y, ud, p, 6.5)) == 0
        @test (@allocated f!(y, ud, p, 7.5)) == 0
    end
end
