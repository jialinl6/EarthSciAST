# By-value split sub-function calls (tree_walk/codegen_kernel.jl,
# `_cg_direct_subcall`).
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
using Test
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

# A cap this small puts each equation in a chunk sub-function of its own and
# spills helpers out of each.
const _DS_CAP = "120"

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
