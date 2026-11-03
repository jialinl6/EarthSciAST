# Intra-kernel body split (ess-iip-split, tree_walk/codegen_kernel.jl).
#
# A kernel whose emitted cell body exceeds `_codegen_fn_node_cap` is partitioned
# so no single generated function is oversized — the body is split across
# `@noinline` helper functions threaded by RETURN VALUE — while the kernel STAYS
# FULLY COMPILED and NEVER declines to the per-cell interpreter (runtime speed is
# critical; the duo LMARS momentum kernels are ~1.5e5 nodes and would otherwise
# OOM the Julia compiler as one function). This pins:
#   * a forced-split build (tiny ESS_CODEGEN_FN_NODE_CAP) still CODEGENS the
#     kernel (`:codegen_kernel` fires) and does NOT decline it to the interpreter;
#   * its du is BIT-identical to the un-split build — the same build with the
#     cap left at its default, which the kernel fits inside — at Float64 and
#     under ForwardDiff (the split is value-exact, order-preserving).
# The split has TWO transports, and this pins both, on every Julia. Which one a
# build takes is `_cg_split_by_value`:
#
#   * BY VALUE, the default — each sub-function compiled as its own
#     RuntimeGeneratedFunction and handed to the body in a tuple appended to
#     `tabs`, called as `_cgfns[k](...)` at a literal index;
#   * inner `@noinline` definitions inside the generated body, called by name,
#     with `ESS_CODEGEN_SPLIT_TRANSPORT=inner` on Julia >= 1.12. RGF rewrites
#     an inner definition into an opaque closure, which is UNTYPED below 1.12
#     (it boxes per cell and segfaults when nested), so there the setting is
#     ignored; on 1.12 it is typed only where the body that builds it calls it.
#
# Both emit the SAME partitioned expression, so this pins that the split
# codegens and is bit-identical under EITHER transport -- there is no Julia on
# which an oversized body has to decline to the per-cell interpreter, which
# under `compiler=:native` would be a refusal.
using Test
using EarthSciAST
using ForwardDiff
include("testutils.jl")
const ESM = EarthSciAST

# A cell body with enough distinct arithmetic that a small fn-node cap must split
# it into several helper functions (a 2-D 9-point-ish combination over gathers).
function _bs_model(N)
    vars = Dict("u" => ESM.ModelVariable(ESM.UnknownVariable; shape = ["i", "j"]))
    g(di, dj) = _idx(
        "u", _op(di < 0 ? "-" : "+", _v("i"), _i(abs(di) == 0 ? 0 : abs(di))),
        _op(dj < 0 ? "-" : "+", _v("j"), _i(abs(dj) == 0 ? 0 : abs(dj)))
    )
    # nonlinear so the tree is deep and not trivially foldable
    terms = ASTExpr[]
    for di in -1:1, dj in -1:1
        push!(terms, _op("*", g(di, dj), _op("+", g(di, dj), _op("*", _n(0.5), g(0, 0)))))
    end
    body = _op("+", terms...)
    lhs = ESM.OpExpr(
        "faq", ESM.ASTExpr[]; output_idx = Any["i", "j"],
        expr_body = _Didx("u", _v("i"), _v("j")), ranges = Dict("i" => [1, N], "j" => [1, N])
    )
    rhs = ESM.OpExpr(
        "faq", ESM.ASTExpr[]; output_idx = Any["i", "j"],
        expr_body = body, ranges = Dict("i" => [1, N], "j" => [1, N])
    )
    return ESM.Model(vars, [ESM.Equation(lhs, rhs)])
end

# Build at an explicit per-function node cap (a retained tuning threshold);
# `fncap = nothing` is the shipped cap, which this kernel fits inside, so that
# build is the un-split reference. `transport` forces the by-value transport.
# Returns RHS + a tally snapshot.
function _bs_build(
        model, ics; fncap = nothing, transport = nothing,
        const_arrays = Dict{String, Vector{Float64}}()
    )
    return withenv(
        "ESS_CODEGEN_FN_NODE_CAP" => (fncap === nothing ? nothing : string(fncap)),
        "ESS_CODEGEN_SPLIT_TRANSPORT" => transport
    ) do
        ESM._reset_cascade_tally!()
        f!, u0, p, _t, vm, _diag = ESM._build_evaluator_impl(
            model; initial_conditions = ics, const_arrays = const_arrays
        )
        (f!, u0, p, copy(ESM._CASCADE_TALLY))
    end
end

# A cell body that reads the state through K distinct connectivity arrays:
# every `u[connk[i]]` lowers to a slot-table read, and each one brings the
# body a loop-invariant table local and geometry locals of its own (bound once
# ahead of the loops). A split of this body spills helpers that reference
# hundreds of them, the shape that made a large stencil RHS allocate per cell.
function _bs_tbl_model(N, K)
    vars = Dict(
        "y" => ESM.ModelVariable(ESM.UnknownVariable),
        "u" => ESM.ModelVariable(ESM.UnknownVariable)
    )
    terms = ASTExpr[]
    for k in 1:K
        g = _idx("u", _idx("conn$k", _v("i")))
        push!(terms, _op("*", _n(0.25 + k / 64), _op("*", g, _op("+", g, _idx("y", _v("i"))))))
    end
    eqs = [
        ESM.Equation(_faq1d(_D_idx("y", _v("i")), "i", 1, N), _faq1d(_op("+", terms...), "i", 1, N)),
        ESM.Equation(
            _faq1d(_D_idx("u", _v("i")), "i", 1, N),
            _faq1d(_op("-", _idx("u", _v("i"))), "i", 1, N)
        ),
    ]
    # Distinct fixed permutations (an odd multiplier is a unit mod a power of
    # two), none of them a run and none with a ghost, so every table stays a
    # table.
    @assert ispow2(N)
    conns = Dict("conn$k" => Float64[mod1((2k + 1) * i + k, N) for i in 1:N] for k in 1:K)
    ics = Dict{String, Float64}()
    for i in 1:N
        ics["y[$i]"] = 0.1i
        ics["u[$i]"] = 1.0 + 0.01i
    end
    return ESM.Model(vars, eqs), ics, conns
end

# The argument count of every split helper reachable from a built RHS. RGF
# rewrites an inner helper `function _cghN(…)` into `_cghN = @opaque (…) -> …`
# inside the generated body; a by-value helper is a generated function of its
# own, whose body carries the split's line marker at top level.
function _bs_helper_arities(root)
    RGF = ESM.RuntimeGeneratedFunctions
    out = Int[]
    ishelper(ex) = ex isa Expr && ex.args[2] isa Expr && any(
        a -> a isa Expr && a.head === :macrocall && length(a.args) >= 2 &&
            a.args[2] isa LineNumberNode && a.args[2].file === Symbol("ess-iip-split"),
        ex.args[2].args
    )
    function scan(e)
        e isa Expr || return
        if e.head === :(=) && e.args[1] isa Symbol &&
                startswith(String(e.args[1]), "_cgh") && e.args[2] isa Expr &&
                e.args[2].head === :macrocall
            lam = e.args[2].args[end]
            lam isa Expr && lam.head === :(->) && push!(out, length(lam.args[1].args))
        end
        foreach(scan, e.args)
        return
    end
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
            id = T.parameters[4]
            id in ids && continue
            push!(ids, id)
            ex = RGF.get_expression(x)
            scan(ex)
            ishelper(ex) && push!(out, length(T.parameters[1]))
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

function _bs_allocs(f!, u, p)
    du = similar(u)
    f!(du, u, p, 0.3)
    f!(du, u, p, 0.3)
    return @allocated f!(du, u, p, 0.3)
end

_bs_du(f!, u, p, t) = (d = similar(u); fill!(d, 0.0); f!(d, u, p, t); d)
_bs_probe(n, k) = Float64[1.4 + 0.9 * sin(1.3i + 0.7k) * cos(0.31i * k) + 0.05i for i in 1:n]
_bs_same(a, b) = size(a) == size(b) && all(a .=== b)
_decl(t) = sum(v for (k, v) in t if startswith(String(k), "codegen_decline"); init = 0)

@testset "intra-kernel body split" begin
    model = _bs_model(6)
    ics = Dict("u" => 1.0)

    # Reference: the shipped cap, which this kernel fits inside — so the split
    # is a no-op and the emission is one function per kernel.
    fr, u0, pr, rt = _bs_build(model, ics)
    @test get(rt, :codegen_kernel, 0) >= 1        # kernel codegens (not interpreter)
    @test _decl(rt) == 0                          # nothing declined

    # Forced split: a small fn-node cap makes the body exceed one function. It is
    # partitioned into sub-functions — and STILL codegens, on EVERY Julia.
    fs, v0, ps, st = _bs_build(model, ics; fncap = 40)
    @test get(st, :codegen_kernel, 0) >= 1    # STILL codegen'd (split, not declined)
    @test _decl(st) == 0                      # the split never falls back to the interpreter

    @test u0 == v0

    # Bit-identical du across the split and the reference, Float64 + the AD axis.
    for k in 1:5, t in (0.0, 0.7, 3.25)
        u = k == 1 ? copy(u0) : _bs_probe(length(u0), k)
        @test _bs_same(_bs_du(fs, u, ps, t), _bs_du(fr, u, pr, t))
    end
    Js = ForwardDiff.jacobian(uu -> _bs_du(fs, uu, ps, 0.4), u0)
    Jr = ForwardDiff.jacobian(uu -> _bs_du(fr, uu, pr, 0.4), u0)
    @test _bs_same(Js, Jr)

    # The INNER-DEFINITION transport, forced. On Julia < 1.12 the setting is
    # ignored and this is the same build the block above already made; on 1.12+
    # it is the one that would otherwise never run here, which is the point — a
    # transport no build takes by default is the one that rots.
    fv, w0, pv, vt = _bs_build(model, ics; fncap = 40, transport = "inner")
    @test get(vt, :codegen_kernel, 0) >= 1
    @test _decl(vt) == 0
    @test u0 == w0
    for k in 1:5, t in (0.0, 0.7, 3.25)
        u = k == 1 ? copy(u0) : _bs_probe(length(u0), k)
        @test _bs_same(_bs_du(fv, u, pv, t), _bs_du(fr, u, pr, t))
    end
    @test _bs_same(ForwardDiff.jacobian(uu -> _bs_du(fv, uu, pv, 0.4), u0), Jr)

    # A build that does NOT split is unchanged by the transport: no
    # sub-function is minted (so, by value, nothing is appended to `tabs`).
    fn, x0, pn, nt = _bs_build(model, ics; transport = "inner")
    @test get(nt, :codegen_kernel, 0) >= 1
    @test _decl(nt) == 0
    @test u0 == x0
    @test _bs_same(_bs_du(fn, copy(u0), pn, 0.7), _bs_du(fr, copy(u0), pr, 0.7))
end

# A split helper takes a handful of arguments however many loop-invariant
# locals its body references: they arrive packed in one tuple, not one argument
# each. One argument each, a stencil that reads its state through many slot
# tables spilled helpers of hundreds of arguments, every call boxed every one of
# them, and the RHS allocated per cell (and ran ~300x slower). Pinned on both
# transports: helpers take a handful of arguments, and the split is
# bit-identical to the un-split build; by value it also allocates no more per
# call than the un-split build does.
@testset "split helpers receive loop-invariant locals packed" begin
    model, ics, conns = _bs_tbl_model(64, 48)
    fr, u0, pr, rt = _bs_build(model, ics; const_arrays = conns)
    @test get(rt, :codegen_kernel, 0) >= 1
    @test _decl(rt) == 0
    @test isempty(_bs_helper_arities(fr))          # the reference does not split
    aref = _bs_allocs(fr, u0, pr)
    Jr = ForwardDiff.jacobian(uu -> _bs_du(fr, uu, pr, 0.4), u0)

    @testset "transport = $(something(transport, "default (by value)"))" for transport in (nothing, "inner")
        # A cap this small splits the body into several helpers, one calling
        # another.
        fs, v0, ps, st = _bs_build(model, ics; fncap = 1500, transport, const_arrays = conns)
        @test get(st, :codegen_kernel, 0) >= 1
        @test _decl(st) == 0
        @test u0 == v0
        ar = _bs_helper_arities(fs)
        @test length(ar) >= 2
        # One argument per invariant local is ~240 for this body.
        @test maximum(ar) <= 16
        for k in 1:4, t in (0.0, 0.7)
            u = k == 1 ? copy(u0) : _bs_probe(length(u0), k)
            @test _bs_same(_bs_du(fs, u, ps, t), _bs_du(fr, u, pr, t))
        end
        @test _bs_same(ForwardDiff.jacobian(uu -> _bs_du(fs, uu, ps, 0.4), u0), Jr)
        # Boxing per cell costs hundreds of KB per call here. The inner-definition
        # transport still allocates per cell when one helper calls another,
        # however few arguments it passes (why it is not the default), so this
        # pins the default one.
        if transport === nothing || !ESM._cg_split_supported()
            @test _bs_allocs(fs, v0, ps) <= aref + 1024
        end
    end
end
