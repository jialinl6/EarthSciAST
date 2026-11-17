# The parallel compile at build (tree_walk/codegen_kernel.jl,
# `_cg_parallel_compile`).
#
# Every generated function an RHS reaches is a static call — a section's own
# function through RGF's `generated_callfunc`, a by-value split's sub-functions
# as `_CGSubFn`s — which Julia compiles together with its caller, one after
# another, at the RHS's first call. With more than one thread a build compiles
# them on several tasks before anything calls the RHS (unless
# ESS_CODEGEN_PARALLEL_COMPILE=0). This pins, on a split whose sub-functions
# take more than 32 arguments:
#   * how the switch reads;
#   * what the build compiles, read off the RHS's Float64 call: the section's
#     own function and its by-value sub-functions, and only the section's
#     function when the split keeps inner definitions or there is none;
#   * that the default build compiles them all before the first call where a
#     build with ESS_CODEGEN_PARALLEL_COMPILE=0 compiles nothing, and that no
#     value changes — at Float64 and under ForwardDiff, and with the section
#     run chunked;
#   * the same for a body that takes the by-value transport by its size (above
#     `_CG_BY_VALUE_MIN_NODES`, lowered here) rather than by the override.
# The compile itself runs in a `julia -t 4` subprocess (the CHILD branch).
using Test
using Logging
using EarthSciAST
using ForwardDiff
include("testutils.jl")
const ESM = EarthSciAST

# Two equations over a 1-D grid, each reading `K` distinct parameter-only
# expressions. Each is a loop-invariant local of the section's prologue, and a
# chunk sub-function receives every one its loops read as an argument of its
# own, so a split of this section has sub-functions of about `K` arguments.
function _pc_model(N, K)
    vars = Dict{String, ESM.ModelVariable}(
        "u" => ESM.ModelVariable(ESM.UnknownVariable),
        "w" => ESM.ModelVariable(ESM.UnknownVariable)
    )
    for k in 1:K
        vars["a$k"] = ESM.ModelVariable(ESM.ParameterVariable; default = 0.1 + k / 97)
    end
    tu = ASTExpr[]
    tw = ASTExpr[]
    for k in 1:K
        push!(tu, _op("*", _op("sin", _v("a$k")), _op("+", _idx("u", _v("i")), _n(k / 11))))
        push!(
            tw, _op(
                "*", _op("cos", _v("a$k")),
                _op("-", _idx("w", _v("i")), _idx("u", _v("i")))
            )
        )
    end
    eqs = [
        ESM.Equation(_faq1d(_D_idx("u", _v("i")), "i", 1, N), _faq1d(_op("+", tu...), "i", 1, N)),
        ESM.Equation(_faq1d(_D_idx("w", _v("i")), "i", 1, N), _faq1d(_op("+", tw...), "i", 1, N)),
    ]
    ics = Dict{String, Float64}()
    for i in 1:N
        ics["u[$i]"] = 0.1i
        ics["w[$i]"] = 1.0 + 0.01i
    end
    return ESM.Model(vars, eqs), ics
end

# Build under the given `ESS_*` settings (`nothing` unsets one); RHS + tallies.
function _pc_build(model, ics, env::Pair...)
    return withenv(env...) do
        ESM._reset_cascade_tally!()
        f!, u0, p, _t, _vm, _diag = ESM._build_evaluator_impl(model; initial_conditions = ics)
        (f!, u0, p, copy(ESM._CASCADE_TALLY))
    end
end

function _pc_allocs(f!, u, p)
    du = similar(u)
    f!(du, u, p, 0.3)
    f!(du, u, p, 0.3)
    return @allocated f!(du, u, p, 0.3)
end

_pc_du(f!, u, p, t) = (d = similar(u); fill!(d, 0.0); f!(d, u, p, t); d)
_pc_probe(u0, k) = Float64[u0[i] + 0.3 * sin(1.3i + 0.7k) for i in eachindex(u0)]
_pc_same(a, b) = size(a) == size(b) && all(a .=== b)
_pc_decl(t) = sum(v for (k, v) in t if startswith(String(k), "codegen_decline"); init = 0)
_pc_jac(f!, u, p) = ForwardDiff.jacobian(uu -> _pc_du(f!, uu, p, 0.4), u)
_pc_section(f!) = getfield(f!, :kernel_section)

# The RHS's Float64 call, the one a build compiles.
_pc_rhs_sig(f!, u, p) = Tuple{typeof(f!), typeof(u), typeof(u), typeof(p), Float64}

# The section's sub-functions: the tuple a by-value split appends to its tabs.
_pc_subfns(f!) = fieldtypes(typeof(_pc_section(f!).cgtabs))[end].parameters

# A sub-function's call signature is its `_CGSubFn` type and its arguments.
_pc_is_subfn(s) = s.parameters[1] <: ESM._CGSubFn
_pc_nargs(s) = length(s.parameters) - 1

# Whether the call `sig` has native code yet: the MethodInstance of exactly that
# signature has a latest CodeInstance with an entry point.
function _pc_compiled(sig)
    for mi in Base.specializations(Base._which(sig).method)
        mi isa Core.MethodInstance && mi.specTypes === sig || continue
        return isdefined(mi, :cache) && mi.cache.invoke != C_NULL
    end
    return false
end

# A cap this small puts each equation in a chunk sub-function of its own and
# spills helpers out of each.
const _PC_CAP = "ESS_CODEGEN_FN_NODE_CAP" => "120"
# The by-value transport, which a body of this size takes only when forced.
const _PC_VALUE = "ESS_CODEGEN_SPLIT_TRANSPORT" => "value"

# With `_CG_BY_VALUE_MIN_NODES` lowered to `lim`, no override.
function _pc_with_threshold(f, lim)
    lim0 = ESM._CG_BY_VALUE_MIN_NODES[]
    try
        ESM._CG_BY_VALUE_MIN_NODES[] = lim
        return withenv(f, "ESS_CODEGEN_SPLIT_TRANSPORT" => nothing)
    finally
        ESM._CG_BY_VALUE_MIN_NODES[] = lim0
    end
end

if get(ENV, "ESS_PC_CHILD", "") == "1"
    # =========================================================================
    # CHILD: the parallel compile under `-t 4` (spawned below), with
    # ESS_THREADS_MIN_WORK=0 so the section also runs chunked.
    # =========================================================================
    @testset "parallel compile at build (child, $(Threads.nthreads()) threads)" begin
        @test Threads.nthreads() >= 2
        model, ics = _pc_model(32, 40)

        # With ESS_CODEGEN_PARALLEL_COMPILE=0, a build compiles nothing:
        # nothing has called the RHS yet.
        fo, u0, po, _ = _pc_build(
            model, ics, _PC_CAP, _PC_VALUE, "ESS_CODEGEN_PARALLEL_COMPILE" => "0"
        )
        subs, roots = ESM._cg_compile_sigs(_pc_rhs_sig(fo, u0, po))
        @test !isempty(roots)
        @test !isempty(subs)
        @test all(_pc_is_subfn, subs)
        @test !any(_pc_compiled, roots)
        @test !any(_pc_compiled, subs)

        # By default the build compiles them before returning. The two builds
        # emit the same code, so their generated functions are the same ones.
        fp, vp, pp, tp = _pc_build(
            model, ics, _PC_CAP, _PC_VALUE, "ESS_CODEGEN_PARALLEL_COMPILE" => nothing
        )
        @test get(tp, :codegen_kernel, 0) >= 1
        @test get(tp, :cg_split_by_value, 0) >= 1
        @test _pc_decl(tp) == 0
        @test vp == u0
        @test ESM._cg_compile_sigs(_pc_rhs_sig(fp, vp, pp)) == (subs, roots)
        @test all(_pc_compiled, roots)
        @test all(_pc_compiled, subs)

        # No value changes, chunked across the threads.
        fr, vr, pr, _ = _pc_build(model, ics)
        @test vr == u0
        for k in 1:4, t in (0.0, 0.7, 3.25)
            u = k == 1 ? copy(u0) : _pc_probe(u0, k)
            ref = _pc_du(fr, u, pr, t)
            @test _pc_same(_pc_du(fp, u, pp, t), ref)
            @test _pc_same(_pc_du(fo, u, po, t), ref)
        end
        @test getfield(_pc_section(fp), :tcache).state == 1      # ran chunked
        Jr = _pc_jac(fr, u0, pr)
        @test _pc_same(_pc_jac(fp, u0, pp), Jr)
        @test _pc_same(_pc_jac(fo, u0, po), Jr)
        @test _pc_allocs(fp, vp, pp) <= _pc_allocs(fo, u0, po)

        # A body above `_CG_BY_VALUE_MIN_NODES` takes the by-value transport by
        # its size, with no override, and the build compiles its sub-functions
        # the same way. Another size, so none of these is compiled yet.
        model2, ics2 = _pc_model(24, 36)
        _pc_with_threshold(1) do
            go, w0, qo, to = _pc_build(
                model2, ics2, _PC_CAP, "ESS_CODEGEN_PARALLEL_COMPILE" => "0"
            )
            @test get(to, :cg_split_by_value, 0) >= 1
            subs2, roots2 = ESM._cg_compile_sigs(_pc_rhs_sig(go, w0, qo))
            @test !isempty(roots2)
            @test !isempty(subs2)
            @test all(_pc_is_subfn, subs2)
            @test isempty(intersect(subs2, subs))
            @test !any(_pc_compiled, roots2)
            @test !any(_pc_compiled, subs2)

            gp, wp, qp, tq = _pc_build(model2, ics2, _PC_CAP)
            @test get(tq, :cg_split_by_value, 0) >= 1
            @test _pc_decl(tq) == 0
            @test wp == w0
            @test ESM._cg_compile_sigs(_pc_rhs_sig(gp, wp, qp)) == (subs2, roots2)
            @test all(_pc_compiled, roots2)
            @test all(_pc_compiled, subs2)

            gr, _, qr, _ = _pc_build(model2, ics2)
            for k in 1:3, t in (0.0, 0.7)
                u = k == 1 ? copy(w0) : _pc_probe(w0, k)
                @test _pc_same(_pc_du(gp, u, qp, t), _pc_du(gr, u, qr, t))
            end
            @test _pc_same(_pc_jac(gp, w0, qp), _pc_jac(gr, w0, qr))
        end

        # A call that reaches no generated function has nothing to compile in
        # parallel, and one that fails is logged at debug level; neither
        # throws, and the first call then compiles as it would have.
        @test ESM._cg_parallel_compile(sin, Tuple{Float64}) === false
        @test_logs(
            (:debug, r"Parallel compile failed"), min_level = Logging.Debug,
            @test ESM._cg_parallel_compile(fp, Union{Tuple{Int}, Tuple{Float64}}) === false
        )
    end
else
    # The parallel compile in this process: under the default test run it has
    # one thread, and the default build then compiles nothing ahead. The
    # compile itself runs in the subprocess at the end.
    @testset "parallel compile at build" begin
        # On unless ESS_CODEGEN_PARALLEL_COMPILE=0.
        for (v, on) in ((nothing, true), ("1", true), ("0", false))
            @test withenv(ESM._cg_parallel_compile_enabled, "ESS_CODEGEN_PARALLEL_COMPILE" => v) === on
        end

        model, ics = _pc_model(32, 40)
        fo, u0, po, to = _pc_build(
            model, ics, _PC_CAP, _PC_VALUE, "ESS_CODEGEN_PARALLEL_COMPILE" => "0"
        )
        @test get(to, :cg_split_by_value, 0) >= 1

        # What a build compiles ahead, read off the RHS's Float64 call: the
        # section's own function and the sub-functions it calls — the build's
        # own, each once, among them ones of more than 32 arguments. A
        # sub-function's own calls (the helpers spilled out of a chunk) are not
        # read: compiling it compiles them.
        subs, roots = ESM._cg_compile_sigs(_pc_rhs_sig(fo, u0, po))
        @test length(roots) == 1
        @test !isempty(subs)
        @test allunique(subs)
        @test all(_pc_is_subfn, subs)
        @test all(s -> s.parameters[1] in _pc_subfns(fo), subs)
        @test length(subs) < length(_pc_subfns(fo))
        @test any(s -> _pc_nargs(s) > 32, subs)
        # A build that did not split has its section's function and nothing else.
        fr, _, pr, _ = _pc_build(model, ics)
        rsubs, rroots = ESM._cg_compile_sigs(_pc_rhs_sig(fr, u0, pr))
        @test isempty(rsubs)
        @test length(rroots) == 1
        # Nor does a split that keeps its sub-functions as inner definitions,
        # which compile as part of the section's function. (Julia < 1.12 always
        # carries them by value.)
        fd, _, pd, td = _pc_build(
            model, ics, _PC_CAP, "ESS_CODEGEN_SPLIT_TRANSPORT" => "definition"
        )
        dsubs, droots = ESM._cg_compile_sigs(_pc_rhs_sig(fd, u0, pd))
        @test length(droots) == 1
        @test isempty(dsubs) == ESM._cg_split_supported()
        @test (get(td, :cg_split_by_value, 0) == 0) == ESM._cg_split_supported()

        # The same for a body that takes the by-value transport by its size.
        _pc_with_threshold(1) do
            fz, _, pz, tz = _pc_build(model, ics, _PC_CAP, "ESS_CODEGEN_PARALLEL_COMPILE" => "0")
            @test get(tz, :cg_split_by_value, 0) >= 1
            @test ESM._cg_compile_sigs(_pc_rhs_sig(fz, u0, pz)) == (subs, roots)
        end

        fp, vp, pp, tp = _pc_build(
            model, ics, _PC_CAP, _PC_VALUE, "ESS_CODEGEN_PARALLEL_COMPILE" => nothing
        )
        @test get(tp, :codegen_kernel, 0) >= 1
        @test _pc_decl(tp) == 0
        @test vp == u0
        Threads.nthreads() == 1 &&
            @test ESM._cg_parallel_compile(fp, Tuple{typeof(vp), typeof(vp), typeof(pp), Float64}) === false
        for k in 1:3, t in (0.0, 0.7)
            u = k == 1 ? copy(u0) : _pc_probe(u0, k)
            ref = _pc_du(fr, u, pr, t)
            @test _pc_same(_pc_du(fp, u, pp, t), ref)
            @test _pc_same(_pc_du(fo, u, po, t), ref)
        end
        @test _pc_same(_pc_jac(fp, u0, pp), _pc_jac(fr, u0, pr))

        @testset "threaded subprocess (julia -t 4)" begin
            env = copy(ENV)
            for k in collect(keys(env))
                startswith(k, "ESS_") && delete!(env, k)
            end
            env["ESS_PC_CHILD"] = "1"
            env["ESS_THREADS_MIN_WORK"] = "0"
            env["JULIA_PROJECT"] = Base.active_project()
            cmd = setenv(`$(Base.julia_cmd()) --startup-file=no -t 4 $(@__FILE__)`, env)
            proc = run(pipeline(ignorestatus(cmd); stdout = stdout, stderr = stderr))
            @test success(proc)
        end
    end
end
