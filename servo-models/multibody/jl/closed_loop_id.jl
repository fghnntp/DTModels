# closed_loop_id.jl
# ----------------------------------------------------------
# Closed-loop internal transfer model & identification (Step 3)
# - 识别 2×2 LTI: 输入 [d_hat, q_r] → 输出 [e, u]
# - 共同分母 IIR：外层随机极点搜索，内层最小二乘求各分子
# - 给定 Gp(z) 后，在频域恢复 S、Gfb、Gff 的频响
#
# 兼容 Julia v1.11.6；仅依赖 Base/Stdlib
# ----------------------------------------------------------

module ClosedLoopID

using LinearAlgebra
using Random
using Statistics

# -----------------------------
# Utilities
# -----------------------------

# 原地多项式相乘：c(x) = a(x)*b(x)，系数按降幂 [1, c1, c2, ...]
function poly_mul!(a::Vector{ComplexF64}, b::Vector{ComplexF64})
    na = length(a); nb = length(b)
    c = zeros(ComplexF64, na + nb - 1)
    @inbounds for i in 1:na
        ai = a[i]
        for j in 1:nb
            c[i+j-1] += ai * b[j]
        end
    end
    return c
end

"""
    poly_from_poles(poles::AbstractVector)

由极点集合构造单项式分母系数 a(z) = 1 + a1 z^-1 + ... + an z^-n。
等价于 Python 的 np.poly(poles)；返回向量 a，长度 = n+1，a[1]==1。
"""
function poly_from_poles(poles::AbstractVector)
    coeffs = ComplexF64[1.0 + 0im]        # 单位多项式
    for p in poles
        coeffs = poly_mul!(coeffs, ComplexF64[1.0 + 0im, -complex(p)])
    end
    a = real.(coeffs)                      # 共轭成对 → 实系数
    a ./= a[1]                             # 单项式（a[1] = 1）
    return a
end




"""
    stable_random_poles(n; rmin=0.3, rmax=0.95, rng=Random.default_rng())

在单位圆内随机采样 n 个极点；包含一定概率的共轭成对复极点。
"""
function stable_random_poles(n::Integer; rmin=0.3, rmax=0.95, rng=Random.default_rng())
    poles = ComplexF64[]
    i = 0
    while i < n
        if (n - i) ≥ 2 && rand(rng) < 0.6
            r  = rand(rng)*(rmax - rmin) + rmin
            th = rand(rng)*(0.95π - 0.05π) + 0.05π
            p  = r*exp(im*th)
            push!(poles, p); push!(poles, conj(p))
            i += 2
        else
            r   = rand(rng)*(rmax - rmin) + rmin
            sgn = rand(rng) < 0.5 ? -1.0 : 1.0
            push!(poles, sgn*r + 0im)
            i += 1
        end
    end
    return poles[1:n]
end

"""
    lfilter_den(x, a)

求 y，使得 (1 + a1 z^-1 + ... + an z^-n) y = x  （因果 IIR 分母）。
a[1] = 1；返回和 x 同长度的向量 y。
"""
function lfilter_den(x::AbstractVector, a::AbstractVector)
    N = length(x)
    na = length(a) - 1
    xv = collect(Float64, x)
    y  = zeros(Float64, N)
    @inbounds for k in 1:N
        acc = xv[k]
        for i in 1:na
            if k - i ≥ 1
                acc -= a[i+1] * y[k - i]
            end
        end
        y[k] = acc
    end
    return y
end

"""
    build_regressor(inputs, a, nb)

给定 inputs = [x1, x2, ...]、共同分母 a，以及分子阶次 nb，
构造回归矩阵：
  每个输入 xm 产生 (nb+1) 列：φ_{xm,j} = lfilter_den(z^{-j} xm, a),  j=0..nb
返回矩阵尺寸 (N, n_inputs*(nb+1))
"""
function build_regressor(inputs::Vector{<:AbstractVector}, a::AbstractVector, nb::Integer)
    cols = Vector{Vector{Float64}}()
    for x in inputs
        xv = collect(Float64, x)
        N  = length(xv)
        for j in 0:nb
            xj = j == 0 ? xv : vcat(zeros(j), xv[1:end-j])
            push!(cols, lfilter_den(xj, a))
        end
    end
    return hcat(cols...)
end

"""
    freq_response_num_den(b, a, w)

计算 H(e^{jw}) = B(z^-1)/A(z^-1) 在 w (rad/sample) 上的频响（复数向量）。
a[1]=1；b,a 为 z^-1 形式的系数。
"""
function freq_response_num_den(b::AbstractVector, a::AbstractVector, w::AbstractVector)
    z = @. cis(w)          # e^{j w}
    num = zeros(ComplexF64, length(w))
    den = ones(ComplexF64,  length(w))
    for (i, bi) in enumerate(b)         # i=1→b0
        num .+= bi .* (z .^ (-(i-1)))
    end
    for i in 2:length(a)                # a[1]已在 den 中
        den .+= a[i] .* (z .^ (-(i-1)))
    end
    return num ./ den
end

# -----------------------------
# 2x2 closed-loop identification
# -----------------------------

"""
CL2x2Identifier(nden, nb; n_random=60, seed=0)

- nden : 共同分母阶次
- nb   : 每条 SISO 分子阶次
- n_random : 随机极点样本数量
- seed : 随机种子
"""
Base.@kwdef mutable struct CL2x2Identifier
    nden::Int
    nb::Int
    n_random::Int = 60
    seed::Int = 0
end

"""
    run(id, d_hat, q_r, e, u) -> Dict

识别四个通道（共享分母）：
  Ged:  e/d_hat
  Geqr: e/q_r
  GCd:  u/d_hat
  GCqr: u/q_r

返回键：
  :a, :b_ed, :b_eqr, :b_Cd, :b_Cqr, :obj
"""
function run(id::CL2x2Identifier,
             d_hat::AbstractVector, q_r::AbstractVector,
             e::AbstractVector,     u::AbstractVector)
    d  = collect(Float64, d_hat)
    qr = collect(Float64, q_r)
    ev = collect(Float64, e)
    uv = collect(Float64, u)
    @assert length(d)==length(qr)==length(ev)==length(uv)

    rng = MersenneTwister(id.seed)
    best_obj = Inf
    best = Dict{Symbol,Any}()

    inputs = [d, qr]
    seg = id.nb + 1

    for _ in 1:id.n_random
        poles = stable_random_poles(id.nden; rng=rng)
        a = poly_from_poles(poles)  # monic

        Φ  = build_regressor(inputs, a, id.nb)     # (N, 2*(nb+1))
        Z  = zeros(size(Φ))
        Φb = [Φ  Z;  Z  Φ]                         # block diag for [e; u]
        Y  = vcat(ev, uv)

        θ = Φb \ Y                                  # 最小二乘
        b_ed  = θ[1:seg]
        b_eqr = θ[seg+1:2seg]
        b_Cd  = θ[2seg+1:3seg]
        b_Cqr = θ[3seg+1:4seg]

        # 重构 e_hat, u_hat
        e_hat = Φ * vcat(b_ed,  b_eqr)
        u_hat = Φ * vcat(b_Cd, b_Cqr)
        resid = vcat(ev .- e_hat, uv .- u_hat)
        obj = mean(resid.^2)

        if obj < best_obj
            best_obj = obj
            best = Dict(
                :obj=>obj, :a=>a,
                :b_ed=>b_ed, :b_eqr=>b_eqr, :b_Cd=>b_Cd, :b_Cqr=>b_Cqr
            )
        end
    end

    return best
end

# -----------------------------
# Recover Gfb/Gff from 2x2 + plant
# -----------------------------

"""
    recover_gfb_gff_from_2x2(Ged, Geqr, Gp, w) -> Dict

给定：
  Ged  = (b_ed,  a)
  Geqr = (b_eqr, a)
  Gp   = (b_p,   a_p)
以及频率栅格 w（rad/sample），返回频响：
  :S, :Gfb, :Gff   （复数向量，长度 = length(w)）
"""
function recover_gfb_gff_from_2x2(Ged::Tuple{AbstractVector,AbstractVector},
                                  Geqr::Tuple{AbstractVector,AbstractVector},
                                  Gp::Tuple{AbstractVector,AbstractVector},
                                  w::AbstractVector)
    b_ed, a  = Ged
    b_eqr, a2 = Geqr
    b_p, a_p = Gp

    H_ed  = freq_response_num_den(b_ed,  a,  w)   # e / d_hat
    H_eqr = freq_response_num_den(b_eqr, a2, w)   # e / q_r
    H_p   = freq_response_num_den(b_p,   a_p, w)  # plant

    eps = 1e-16
    S   = .- H_ed ./ (H_p .+ eps)
    Gfb = (1 .- S) ./ (S .* (H_p .+ eps))
    Gff = (1 .- H_eqr ./ (S .+ eps)) ./ (H_p .+ eps)

    return Dict(:S=>S, :Gfb=>Gfb, :Gff=>Gff)
end

end