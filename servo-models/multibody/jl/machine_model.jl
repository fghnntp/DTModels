# machine_model.jl
# ================
# 机电一步法（Step 1）开环辨识 + 扰动重构（Step 2）—— Julia 版
# 兼容 Julia v1.11.6
#
# 主要内容：
# - finite_diff：中央差分一/二阶导
# - PositionGrid1D：位置格点三角形（hat）基函数 + 预测
# - FrictionStribeck1D：Stribeck + 库仑（带继电内态 dn）基函数与预测
# - OpenLoopRegressor1D：构建设计矩阵 W
# - Step1Identifier1D：外层随机搜索 (v_s, dv_band) + 内层最小二乘求线性参数
# - reconstruct_disturbance_1d：按识别参数重构广义扰动 d_hat
#
# 说明：
# - Python 的 dataclass 改为 Julia 的结构体；API/字段名保持一致或等价。
# - 线性最小二乘使用 \ 运算符；等价于 numpy.linalg.lstsq 的最小二乘解。
# - 随机数使用 Random.MersenneTwister(seed) 控制复现性。
# - 向量/矩阵默认为 Float64。
# =====================================================================
module MachineModel

using LinearAlgebra
using Random
using Statistics

# -------------------------------
# 工具：中央差分的一阶与二阶导
# -------------------------------
"""
    finite_diff(x::AbstractVector{<:Real}, dt::Real) -> (dx, ddx)

中央差分近似的一阶/二阶导数。端点使用前/后向差分。
"""
function finite_diff(x::AbstractVector{<:Real}, dt::Real)
    N = length(x)
    @assert N ≥ 3 "finite_diff 需要至少 3 个样本"
    dx  = zeros(Float64, N)
    ddx = zeros(Float64, N)
    # 一阶导
    dx[1]   = (x[2] - x[1]) / dt
    dx[end] = (x[end] - x[end-1]) / dt
    @inbounds for k in 2:N-1
        dx[k] = (x[k+1] - x[k-1]) / (2dt)
    end
    # 二阶导
    ddx[1]   = (x[3] - 2x[2] + x[1]) / (dt^2)
    ddx[end] = (x[end] - 2x[end-1] + x[end-2]) / (dt^2)
    @inbounds for k in 2:N-1
        ddx[k] = (x[k+1] - 2x[k] + x[k-1]) / (dt^2)
    end
    return dx, ddx
end

# --------------------------------------------
# 位置格点：三角形（hat）基函数 + 线性组合
# --------------------------------------------
mutable struct PositionGrid1D
    p::Vector{Float64}   # 格点中心 (Ng)
    m::Vector{Float64}   # 每个格点的幅值系数 (Ng)
    Cg::Float64          # 半支撑宽度（每个 hat 左右支持 Cg，总宽 2*Cg）
end

"""
    weights(grid::PositionGrid1D, q::AbstractVector) -> Wpos::Matrix

返回 (N × Ng) 的权重矩阵；第 i 行是位置 q[i] 对每个格点的三角形权重。
"""
function weights(grid::PositionGrid1D, q::AbstractVector{<:Real})
    qv = collect(Float64, q)
    N  = length(qv)
    Ng = length(grid.p)
    Wpos = zeros(Float64, N, Ng)
    Cg = grid.Cg
    @inbounds for k in 1:Ng
        pk = grid.p[k]
        for i in 1:N
            dist = abs(qv[i] - pk)
            if dist < Cg
                wk = 1.0 - dist / Cg
                # clip 到 [0,1]
                wk = wk < 0 ? 0.0 : (wk > 1 ? 1.0 : wk)
                Wpos[i, k] = wk
            else
                Wpos[i, k] = 0.0
            end
        end
    end
    return Wpos
end

"""
    predict(grid::PositionGrid1D, q) -> up

位置扰动的预测：up = Wpos(q) * m
"""
predict(grid::PositionGrid1D, q::AbstractVector{<:Real}) =
    weights(grid, q) * grid.m

# --------------------------------------------
# 摩擦：Stribeck + 库仑 + 继电内态 dn
# --------------------------------------------
mutable struct FrictionStribeck1D
    v_s::Float64     # Stribeck 速度尺度
    dv_band::Float64 # 内态“速度带宽”
    dn0::Float64     # 初始内态
end

# 计算继电内态 dn 的时间序列
function compute_dn_series(fric::FrictionStribeck1D,
                           dq::AbstractVector{<:Real},
                           ddq::AbstractVector{<:Real})
    dqv  = collect(Float64, dq)
    ddqv = collect(Float64, ddq)
    N = length(dqv)
    @assert length(ddqv) == N "dq 与 ddq 长度需一致"
    dn = zeros(Float64, N)
    dn_prev = fric.dn0
    dvb = max(fric.dv_band, 1e-12)
    @inbounds for k in 1:N
        if k > 1 && sign(ddqv[k]) * sign(dqv[k-1]) == -1
            dn_k = dn_prev
        else
            inc = k == 1 ? 0.0 : (dqv[k] - dqv[k-1]) / dvb
            dn_k = clamp(dn_prev + inc, -1.0, 1.0)
        end
        dn[k] = dn_k
        dn_prev = dn_k
    end
    return dn
end

# 返回两组基函数：bC=dn（库仑极性），bS=exp(-|dq|/v_s)（Stribeck 下陷）
function basis(fric::FrictionStribeck1D,
               dq::AbstractVector{<:Real},
               ddq::AbstractVector{<:Real})
    dn = compute_dn_series(fric, dq, ddq)
    vs = max(fric.v_s, 1e-12)
    bS = @. exp(-abs(dq) / vs)
    return dn, bS
end

# 给定 fC, fS，返回摩擦力预测：uf = fC*bC + fS*bS
function predict(fric::FrictionStribeck1D,
                 dq::AbstractVector{<:Real},
                 ddq::AbstractVector{<:Real},
                 fC::Real, fS::Real)
    bC, bS = basis(fric, dq, ddq)
    return @. fC*bC + fS*bS
end

# --------------------------------------------
# 开环回归矩阵：MΔ*ddq + C*dq + K*q + fC*bC + fS*bS + Σ m_i*w_i(q)
# --------------------------------------------
mutable struct OpenLoopRegressor1D
    grid::Union{Nothing, PositionGrid1D}
    fric::Union{Nothing, FrictionStribeck1D}
end

"""
    build(reg::OpenLoopRegressor1D, q, dq, ddq) -> W::Matrix

按顺序堆列：
[ddq, dq, q, (若有摩擦)bC, bS, (若有位置)每个格点权重列]
"""
function build(reg::OpenLoopRegressor1D,
               q::AbstractVector{<:Real},
               dq::AbstractVector{<:Real},
               ddq::AbstractVector{<:Real})
    qv   = collect(Float64, q)
    dqv  = collect(Float64, dq)
    ddqv = collect(Float64, ddq)
    N = length(qv)
    cols = Vector{Vector{Float64}}()
    push!(cols, ddqv)
    push!(cols, dqv)
    push!(cols, qv)
    if reg.fric !== nothing
        bC, bS = basis(reg.fric, dqv, ddqv)
        push!(cols, bC); push!(cols, bS)
    end
    if reg.grid !== nothing
        Wpos = weights(reg.grid, qv)
        Ng = size(Wpos, 2)
        for k in 1:Ng
            push!(cols, @view Wpos[:, k])
        end
    end
    # hcat 列成矩阵
    return hcat(cols...)
end

# --------------------------------------------
# Step 1：外层搜索 (v_s, dv_band) + 内层最小二乘
# --------------------------------------------
mutable struct Step1Identifier1D
    grid_centers::Vector{Float64}
    grid_halfwidth::Float64
    n_grid::Int
    fric_v_s_bounds::Tuple{Float64,Float64}
    fric_dv_bounds::Tuple{Float64,Float64}
    n_random::Int
    seed::Int
end

function Step1Identifier1D(; grid_centers::AbstractVector{<:Real},
                           grid_halfwidth::Real,
                           n_grid::Integer,
                           fric_v_s_bounds::Tuple{<:Real,<:Real} = (1e-4, 1.0),
                           fric_dv_bounds::Tuple{<:Real,<:Real} = (1e-5, 1e-2),
                           n_random::Integer = 50,
                           seed::Integer = 0)
    return Step1Identifier1D(collect(Float64, grid_centers),
                             float(grid_halfwidth),
                             Int(n_grid),
                             (float(first(fric_v_s_bounds)), float(last(fric_v_s_bounds))),
                             (float(first(fric_dv_bounds)), float(last(fric_dv_bounds))),
                             Int(n_random), Int(seed))
end

"""
    run(id::Step1Identifier1D, q, dq, ddq, u) -> Dict

随机搜索 (v_s, dv)，对每组参数构建 W 并最小二乘解 θ，
以 MSE 作为目标挑最优，返回参数字典：
{M_delta, C, K, fC, fS, m_grid, v_s, dv_band, u_hat, obj, W}
"""
function run(id::Step1Identifier1D,
             q::AbstractVector{<:Real},
             dq::AbstractVector{<:Real},
             ddq::AbstractVector{<:Real},
             u::AbstractVector{<:Real})
    qv   = collect(Float64, q)
    dqv  = collect(Float64, dq)
    ddqv = collect(Float64, ddq)
    uv   = collect(Float64, u)
    @assert length(qv) == length(dqv) == length(ddqv) == length(uv)

    rng = MersenneTwister(id.seed)
    grid = PositionGrid1D(copy(id.grid_centers), zeros(Float64, id.n_grid), id.grid_halfwidth)

    best_obj = Inf
    best = Dict{Symbol,Any}()

    for _ in 1:id.n_random
        v_s = rand(rng) * (id.fric_v_s_bounds[2] - id.fric_v_s_bounds[1]) + id.fric_v_s_bounds[1]
        dv  = rand(rng) * (id.fric_dv_bounds[2]   - id.fric_dv_bounds[1])   + id.fric_dv_bounds[1]
        fric = FrictionStribeck1D(v_s, dv, 0.0)
        W = build(OpenLoopRegressor1D(grid, fric), qv, dqv, ddqv)
        # 最小二乘：W * θ ≈ u
        θ = W \ uv
        u_hat = W * θ
        obj = mean((uv .- u_hat).^2)
        if obj < best_obj
            best_obj = obj
            best = Dict(
                :obj => obj, :v_s => v_s, :dv => dv,
                :theta => θ, :u_hat => u_hat, :W => W
            )
        end
    end

    θ = best[:theta]
    @assert length(θ) ≥ 5 "参数维度不足，检查 n_grid / 设计矩阵构造"

    MΔ, C, K = θ[1], θ[2], θ[3]
    fC, fS   = θ[4], θ[5]
    m = id.n_grid > 0 ? θ[6:5+id.n_grid] : zeros(Float64, 0)

    return Dict(
        "M_delta" => MΔ,
        "C"       => C,
        "K"       => K,
        "fC"      => fC,
        "fS"      => fS,
        "m_grid"  => m,
        "v_s"     => best[:v_s],
        "dv_band" => best[:dv],
        "u_hat"   => best[:u_hat],
        "obj"     => best[:obj],
        "W"       => best[:W]
    )
end

# --------------------------------------------
# Step 2：按识别参数重构广义扰动 d_hat
# --------------------------------------------
"""
    reconstruct_disturbance_1d(q, dq, ddq, pars, grid_centers, grid_halfwidth) -> d_hat

根据 Step1 的参数字典 pars 构造：
d_hat = MΔ*ddq + C*dq + K*q + uf(dq,ddq) + up(q)
"""
function reconstruct_disturbance_1d(q::AbstractVector{<:Real},
                                    dq::AbstractVector{<:Real},
                                    ddq::AbstractVector{<:Real},
                                    pars::Dict{String,Any},
                                    grid_centers::AbstractVector{<:Real},
                                    grid_halfwidth::Real)
    MΔ = float(pars["M_delta"])
    C  = float(pars["C"])
    K  = float(pars["K"])
    fC = float(pars["fC"])
    fS = float(pars["fS"])
    v_s = float(pars["v_s"])
    dv  = float(pars["dv_band"])

    fric = FrictionStribeck1D(v_s, dv, 0.0)
    uf = predict(fric, dq, ddq, fC, fS)

    grid = PositionGrid1D(collect(Float64, grid_centers), collect(Float64, pars["m_grid"]), float(grid_halfwidth))
    up = predict(grid, q)

    u_lin = @. MΔ*ddq + C*dq + K*q
    d_hat = @. u_lin + uf + up
    return d_hat
end

end