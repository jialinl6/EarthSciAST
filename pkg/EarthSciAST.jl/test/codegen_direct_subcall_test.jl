# By-value split sub-function calls (tree_walk/codegen_kernel.jl,
# `_cg_direct_subcall`), and the parallel compile at build they make possible
# (`_cg_parallel_compile`).
#
# A by-value split hands each emitted sub-function to the body that calls it as
# a RuntimeGeneratedFunction in a tuple, `_cgfns`. Called as `_cgfns[k](…)`, it
# goes through the RuntimeGeneratedFunction's functor, which splats its
# arguments into `generated_callfunc`, and inference does not expand a splat of
# more than 32 elements: a sub-function taking more was a dynamic call, which
# allocated on every call and compiled at its own first call. The build calls
# `generated_callfunc(_cgfns[k], …)` instead, what the functor forwards to.
# This pins, on a split whose sub-functions take more than 32 arguments:
#   * the RHS is bit-identical with the direct call on and off
#     (ESS_CODEGEN_DIRECT_SUBCALL=0) and to the un-split build, at Float64 and
#     under ForwardDiff;
#   * with the direct call no `_cgfns[k](…)` call is left, and a warm RHS call
#     allocates no more than the un-split build does.
# The direct call makes every generated function an RHS reaches a static call,
# which Julia compiles together with its caller, one after another, so with
# ESS_CODEGEN_PARALLEL_COMPILE=1 a build compiles them on several tasks before
# anything calls the RHS. This pins what that compiles (the section's own
# function and its sub-functions, read off the RHS's Float64 call), that a
# build compiles them all before the first call where a build without the
# switch compiles nothing, and that no value changes: in this process, and in
# a `julia -t 4` subprocess (the CHILD branch at the end) where the compile is
# parallel and the section runs chunked.
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
function _ds_model(N, K)
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
function _ds_build(model, ics, env::Pair...)
    return withenv(env...) do
        ESM._reset_cascade_tally!()
        f!, u0, p, _t, _vm, _diag = ESM._build_evaluator_impl(model; initial_conditions = ics)
        (f!, u0, p, copy(ESM._CASCADE_TALLY))
    end
end

# Every RuntimeGeneratedFunction reachable from a built RHS, once each.
function _ds_rgfs(root)
    RGF = ESM.RuntimeGeneratedFunctions
    out = Any[]
    seen = IdDict{Any, Nothing}()
    ids = Set{Any}()
    stack = Any[root]
    while !isempty(stack)
        x = pop!(stack)
        T = typeof(x)
        (isbitstype(T) || x isa AbstractString || x isa Symbol || x isa Module) && continue
        if ismutable(x)
            haskey(seen, x) && continue
            seen[x] = nothing
        end
        if x isa RGF.RuntimeGeneratedFunction
            T.parameters[4] in ids && continue
            push!(ids, T.parameters[4])
            push!(out, x)
        elseif x isa Array
            eltype(x) <: Number && continue
            for i in eachindex(x)
                isassigned(x, i) && push!(stack, x[i])
            end
        elseif !(x isa AbstractDict)
            for i in 1:nfields(x)
                isdefined(x, i) && push!(stack, getfield(x, i))
            end
        end
    end
    return out
end

_ds_arities(root) = Int[length(typeof(r).parameters[1]) for r in _ds_rgfs(root)]

# Calls in the generated code of `root` whose callee satisfies `pred`.
function _ds_ncalls(root, pred)
    n = 0
    function walk(e)
        e isa Expr || return
        e.head === :call && !isempty(e.args) && pred(e.args[1]) && (n += 1)
        foreach(walk, e.args)
        return
    end
    for r in _ds_rgfs(root)
        walk(ESM.RuntimeGeneratedFunctions.get_expression(r))
    end
    return n
end
_ds_functor(f) = f isa Expr && f.head === :ref && f.args[1] === :_cgfns
_ds_direct(f) = f isa GlobalRef && f.name === :generated_callfunc

function _ds_allocs(f!, u, p)
    du = similar(u)
    f!(du, u, p, 0.3)
    f!(du, u, p, 0.3)
    return @allocated f!(du, u, p, 0.3)
end

_ds_du(f!, u, p, t) = (d = similar(u); fill!(d, 0.0); f!(d, u, p, t); d)
_ds_probe(u0, k) = Float64[u0[i] + 0.3 * sin(1.3i + 0.7k) for i in eachindex(u0)]
_ds_same(a, b) = size(a) == size(b) && all(a .=== b)
_ds_decl(t) = sum(v for (k, v) in t if startswith(String(k), "codegen_decline"); init = 0)
_ds_jac(f!, u, p) = ForwardDiff.jacobian(uu -> _ds_du(f!, uu, p, 0.4), u)
_ds_section(f!) = getfield(f!, :kernel_section)

# The RHS's Float64 call, the one a build with the switch on compiles.
_ds_rhs_sig(f!, u, p) = Tuple{typeof(f!), typeof(u), typeof(u), typeof(p), Float64}

# The section's sub-functions: the tuple a by-value split appends to its tabs.
_ds_subfns(f!) = fieldtypes(typeof(_ds_section(f!).cgtabs))[end].parameters

# Whether the generated-function call `sig` has native code yet: its
# MethodInstance's latest CodeInstance has an entry point.
function _ds_compiled(sig)
    for m in methods(ESM.RuntimeGeneratedFunctions.generated_callfunc)
        for mi in Base.specializations(m)
            mi isa Core.MethodInstance && mi.specTypes === sig || continue
            return isdefined(mi, :cache) && mi.cache.invoke != C_NULL
        end
    end
    return false
end

# A cap this small puts each equation in a chunk sub-function of its own and
# spills helpers out of each.
const _DS_CAP = "120"

if get(ENV, "ESS_DS_CHILD", "") == "1"
    # =========================================================================
    # CHILD: the parallel compile under `-t 4` (spawned below), with
    # ESS_THREADS_MIN_WORK=0 so the section also runs chunked.
    # =========================================================================
    @testset "parallel compile at build (child, $(Threads.nthreads()) threads)" begin
        @test Threads.nthreads() >= 2
        model, ics = _ds_model(32, 40)
        cap = "ESS_CODEGEN_FN_NODE_CAP" => _DS_CAP

        # Without the switch, a build compiles nothing: nothing has called
        # the RHS yet.
        fo, u0, po, _ = _ds_build(model, ics, cap)
        subs, roots = ESM._cg_compile_sigs(_ds_rhs_sig(fo, u0, po))
        @test !isempty(roots)
        @test !isempty(subs)
        @test !any(_ds_compiled, roots)
        @test !any(_ds_compiled, subs)

        # With it, the build compiles them before returning. The two builds
        # emit the same code, so their generated functions are the same ones.
        fp, vp, pp, tp = _ds_build(
            model, ics, cap, "ESS_CODEGEN_PARALLEL_COMPILE" => "1"
        )
        @test get(tp, :codegen_kernel, 0) >= 1
        @test _ds_decl(tp) == 0
        @test vp == u0
        @test ESM._cg_compile_sigs(_ds_rhs_sig(fp, vp, pp)) == (subs, roots)
        @test all(_ds_compiled, roots)
        @test all(_ds_compiled, subs)

        # No value changes, chunked across the threads.
        fr, vr, pr, _ = _ds_build(model, ics)
        @test vr == u0
        for k in 1:4, t in (0.0, 0.7, 3.25)
            u = k == 1 ? copy(u0) : _ds_probe(u0, k)
            ref = _ds_du(fr, u, pr, t)
            @test _ds_same(_ds_du(fp, u, pp, t), ref)
            @test _ds_same(_ds_du(fo, u, po, t), ref)
        end
        @test getfield(_ds_section(fp), :tcache).state == 1      # ran chunked
        Jr = _ds_jac(fr, u0, pr)
        @test _ds_same(_ds_jac(fp, u0, pp), Jr)
        @test _ds_same(_ds_jac(fo, u0, po), Jr)
        @test _ds_allocs(fp, vp, pp) <= _ds_allocs(fo, u0, po)

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
    @testset "by-value sub-functions are called directly" begin
        model, ics = _ds_model(32, 40)
        fr, u0, pr, rt = _ds_build(model, ics)
        @test get(rt, :codegen_kernel, 0) >= 1
        @test _ds_decl(rt) == 0
        @test length(_ds_rgfs(fr)) == 1                # the reference does not split
        aref = _ds_allocs(fr, u0, pr)
        Jr = ForwardDiff.jacobian(uu -> _ds_du(fr, uu, pr, 0.4), u0)

        fd, vd, pd, td = _ds_build(model, ics, "ESS_CODEGEN_FN_NODE_CAP" => _DS_CAP)
        ff, vf, pf, tf = _ds_build(
            model, ics, "ESS_CODEGEN_FN_NODE_CAP" => _DS_CAP,
            "ESS_CODEGEN_DIRECT_SUBCALL" => "0"
        )
        for (f, v, t) in ((fd, vd, td), (ff, vf, tf))
            @test get(t, :codegen_kernel, 0) >= 1
            @test _ds_decl(t) == 0
            @test v == u0
        end
        # The split is the one this file is about: sub-functions past the 32
        # arguments inference expands in a splat.
        @test maximum(_ds_arities(fd)) > 32
        @test _ds_arities(fd) == _ds_arities(ff)

        # Every call site is rewritten by default, and none with the switch off.
        @test _ds_ncalls(fd, _ds_functor) == 0
        @test _ds_ncalls(fd, _ds_direct) == _ds_ncalls(ff, _ds_functor) > 0
        @test _ds_ncalls(ff, _ds_direct) == 0

        # Bit-identical: both call forms and the un-split build.
        for k in 1:5, t in (0.0, 0.7, 3.25)
            u = k == 1 ? copy(u0) : _ds_probe(u0, k)
            ref = _ds_du(fr, u, pr, t)
            @test _ds_same(_ds_du(fd, u, pd, t), ref)
            @test _ds_same(_ds_du(ff, u, pf, t), ref)
        end
        @test _ds_same(ForwardDiff.jacobian(uu -> _ds_du(fd, uu, pd, 0.4), u0), Jr)
        @test _ds_same(ForwardDiff.jacobian(uu -> _ds_du(ff, uu, pf, 0.4), u0), Jr)

        # The direct call does not allocate. (Through the functor, this split
        # allocates on every call of a sub-function of more than 32 arguments.)
        @test _ds_allocs(fd, vd, pd) <= aref
    end


    # ESS_CODEGEN_PARALLEL_COMPILE=1 in this process: under the default test run
    # it has one thread, and the build then compiles nothing ahead. The compile
    # itself runs in the subprocess at the end.
    @testset "parallel compile at build" begin
        model, ics = _ds_model(32, 40)
        cap = "ESS_CODEGEN_FN_NODE_CAP" => _DS_CAP
        fo, u0, po, _ = _ds_build(model, ics, cap)

        # What a build compiles ahead, read off the RHS's Float64 call: the
        # section's own function and the sub-functions it calls — the build's
        # own, each once, among them the ones of more than 32 arguments that the
        # functor form kept out of the section's code.
        subs, roots = ESM._cg_compile_sigs(_ds_rhs_sig(fo, u0, po))
        @test length(roots) == 1
        @test !isempty(subs)
        @test allunique(subs)
        @test all(s -> s.parameters[2] in _ds_subfns(fo), subs)
        @test any(s -> length(s.parameters) - 2 > 32, subs)
        # A build that did not split has its section's function and nothing else.
        fr, _, pr, _ = _ds_build(model, ics)
        rsubs, rroots = ESM._cg_compile_sigs(_ds_rhs_sig(fr, u0, pr))
        @test isempty(rsubs)
        @test length(rroots) == 1

        fp, vp, pp, tp = _ds_build(model, ics, cap, "ESS_CODEGEN_PARALLEL_COMPILE" => "1")
        @test get(tp, :codegen_kernel, 0) >= 1
        @test _ds_decl(tp) == 0
        @test vp == u0
        Threads.nthreads() == 1 &&
            @test ESM._cg_parallel_compile(fp, Tuple{typeof(vp), typeof(vp), typeof(pp), Float64}) === false
        for k in 1:3, t in (0.0, 0.7)
            u = k == 1 ? copy(u0) : _ds_probe(u0, k)
            @test _ds_same(_ds_du(fp, u, pp, t), _ds_du(fo, u, po, t))
        end
        @test _ds_same(_ds_jac(fp, u0, pp), _ds_jac(fo, u0, po))

        @testset "threaded subprocess (julia -t 4)" begin
            env = copy(ENV)
            for k in collect(keys(env))
                startswith(k, "ESS_") && delete!(env, k)
            end
            env["ESS_DS_CHILD"] = "1"
            env["ESS_THREADS_MIN_WORK"] = "0"
            env["JULIA_PROJECT"] = Base.active_project()
            cmd = setenv(`$(Base.julia_cmd()) --startup-file=no -t 4 $(@__FILE__)`, env)
            proc = run(pipeline(ignorestatus(cmd); stdout = stdout, stderr = stderr))
            @test success(proc)
        end
    end
end
