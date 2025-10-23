# ===============================================================
# 无 IMU 的多体（刚性连接）系统辨识骨架 — Julia v1.11.6
# 依赖：LinearAlgebra, Statistics, Random, ControlSystems, Polynomials
# 说明：
#   • 复用你已有的 Step1/Step2（见 model.jl 中的：
#       finite_diff, PositionGrid1D, FrictionStribeck1D,
#       OpenLoopRegressor1D, Step1Identifier1D, reconstruct_disturbance_1d）
#   • 本文件提供：多轴数据结构、跨轴/姿态基底 Φx 生成、
#       逐轴 Step1 扩展（先跑你的一步法，再拟合跨轴系数 β_cross）、
#       Step2 扰动重构（含跨轴项）、Step3 共分母 2×2 闭环拟合。
#   • 目标：方案 A（无 IMU）。轴间耦合与姿态依赖通过 Φx 吸收为“广义扰动”。
# ===============================================================


module Model

using Pkg
Pkg.activate(".")


using LinearAlgebra
using Statistics
using Random
using Polynomials


include("machine_model.jl")  # 单轴 Step1/Step2 模块
include("closed_loop_id.jl")  # 单轴 Step3 模块

function generate_axis_model(q::Vector{Float64}, u::Vector{Float64}, Ts::Float64,
                             grid_centers::Vector{Float64},
                             grid_halfwidth::Float64,
                             n_grid::Int,
                             fric_v_s_bounds::Tuple{Float64,Float64},
                             fric_dv_bounds::Tuple{Float64,Float64},
                             n_random_step1::Int,
                             seed_step1::Int,
                             nden::Int,
                             nb::Int,
                             n_random_step3::Int,
                             seed_step_3::Int
                             )
    machine_model = MachineModel.Step1Identifier1D(
        grid_centers = grid_centers,
        grid_halfwidth = grid_halfwidth,
        n_grid = n_grid,
        fric_v_s_bounds = fric_v_s_bounds,
        fric_dv_bounds = fric_dv_bounds,
        n_random = n_random_step1,
        seed = seed_step1,
    )

    dq, ddq = machine_model.finite_diff(q, Ts)

    machine_pars = MachineModel.run(machine_model, q, dq, ddq, u)

    # M_delta = machine_pars["M_delta"]; C = machine_pars["C"]; K = machine_pars["K"]
    # fC = machine_pars["fC"]; fS = machine_pars["fS"]
    # v_s = machine_pars["v_s"]; dv = machine_pars["dv_band"]
    # m_grid = machine_pars["m_grid"]

    d_hat = MachineModel.reconstruct_disturbance_1d(q, dq, ddq, machine_pars, grid_centers, grid_halfwidth)

    servo_model = ClosedLoopID.CL2x2Identifier(
        nden = nden,
        nb = nb,
        n_random = n_random_step3,
        seed = seed_step_3,
    )

    iden = ClosedLoopID.run(servo_model, d_hat, q, zeros(Float64, length(q)), u)
    return iden
end


# -------------------------- 数据结构 -------------------------- #

Base.@kwdef mutable struct AxisData
    q::Vector{Float64}
    qr::Vector{Float64}
    u::Vector{Float64}
    Ts::Float64
    qd::Union{Nothing,Vector{Float64}} = nothing
    qdd::Union{Nothing,Vector{Float64}} = nothing
end

Base.@kwdef mutable struct RigidOptions
    # 位置格点（用于你 Step1 的 PositionGrid1D）
    grid_centers::Vector{Float64} = Float64[]
    grid_halfwidth::Float64 = 0.0
    n_grid::Int = 0  # 与 centers 一致即可

    # 摩擦参数搜索范围（传给 Step1Identifier1D）
    fric_v_s_bounds::Tuple{Float64,Float64} = (1e-4, 1.0)
    fric_dv_bounds::Tuple{Float64,Float64} = (1e-5, 1e-2)
    n_random::Int = 50
    seed::Int = 0

    # 跨轴/姿态基底选项
    include_cross::Bool = true
    include_attitude::Bool = true
    # 位置谐波：对每个轴的 K 与 L 比例尺（若需要额外的位置谐波，可在你 Step1 的网格中表达）
    K_harm::Int = 0   # 如果为 0，则不在 Φx 里再加谐波
    L_map::Dict{Symbol,Float64} = Dict{Symbol,Float64}()
end

# ------------------ 跨轴/姿态基底 Φx 生成 ------------------ #

"""
构造轴 i 的跨轴/姿态特征矩阵 Φx（T×Kx）。
可包含：其他轴的 qd/qdd，姿态（A/B/C）正余弦，简单交叉项等。
注意：这里的列全都会并入“广义扰动”的线性项（β_cross）。
"""
function build_cross_features(i::Symbol, data_all::Dict{Symbol,AxisData},
                              opt::RigidOptions)
    T = length(data_all[i].q)
    cols = Vector{Vector{Float64}}()

    if opt.include_cross
        for (j, aj) in data_all
            j == i && continue
            # 速度/加速度（若无则差分）
            qdj = isnothing(aj.qd) ? vcat((aj.q[2]-aj.q[1])/aj.Ts, diff(aj.q)./aj.Ts) : aj.qd
            qddj= isnothing(aj.qdd) ? vcat((qdj[2]-qdj[1])/aj.Ts, diff(qdj)./aj.Ts) : aj.qdd
            push!(cols, qdj, qddj)
        end
    end

    if opt.include_attitude
        for key in (:A, :B, :C)
            haskey(data_all, key) || continue
            θ = data_all[key].q
            push!(cols, sin.(θ), cos.(θ))
        end
    end

    if opt.K_harm > 0
        ai = data_all[i]
        L = get(opt.L_map, i, 1.0)
        for k in 1:opt.K_harm
            push!(cols, sin.(k .* ai.q ./ L), cos.(k .* ai.q ./ L))
        end
    end

    if isempty(cols)
        return zeros(Float64, T, 0)
    else
        return hcat(cols...)
    end
end

# ---------------- 逐轴 Step1：先跑你的一步法，再拟 β_cross --------------- #

"""
run_step1_rigid!(i, data_all, opt; step1_ctor)
  • 先调用你的 Step1Identifier1D 仅用“本轴回归”（ddq,dq,q,摩擦,位置网格）拟合，得到 pars_base。
  • 计算残差 r = u - u_hat_base，将 r ≈ Φx β_cross 做一轮 LS，得到 β_cross。
  • 返回包含 base 参数与 β_cross 的字典。

要求：外部先 include("model.jl")，并把 Step1Identifier1D/PositionGrid1D/… 放在 Main 下。
你可传入 step1_ctor = (args...)->Main.Step1Identifier1D(args...)
"""
function run_step1_rigid!(i::Symbol, data_all::Dict{Symbol,AxisData}, opt::RigidOptions;
                          step1_ctor = Main.Step1Identifier1D,
                          reconstruct_d1d = Main.reconstruct_disturbance_1d)
    ai = data_all[i]

    # 1) 估速度/加速度（若未提供）
    qd = isnothing(ai.qd) ? vcat((ai.q[2]-ai.q[1])/ai.Ts, diff(ai.q)./ai.Ts) : ai.qd
    qdd= isnothing(ai.qdd) ? vcat((qd[2]-qd[1])/ai.Ts, diff(qd)./ai.Ts) : ai.qdd

    # 2) 跑你的一步法（本轴项）
    id = step1_ctor(
        grid_centers = opt.grid_centers,
        grid_halfwidth = opt.grid_halfwidth,
        n_grid = opt.n_grid,
        fric_v_s_bounds = opt.fric_v_s_bounds,
        fric_dv_bounds = opt.fric_dv_bounds,
        n_random = opt.n_random,
        seed = opt.seed,
    )
    pars_base = Main.run(id, ai.q, qd, qdd, ai.u)  # Dict{String,Any}

    # 3) 由 base 参数重构 u_hat_base，并计算残差
    d_hat_base = reconstruct_d1d(ai.q, qd, qdd, pars_base, opt.grid_centers, opt.grid_halfwidth)
    u_hat_base = d_hat_base  # 你的 Step1 模型是 u ≈ MΔ*ddq + C*dq + K*q + uf + up
    r = ai.u .- u_hat_base

    # 4) 构造 Φx（跨轴/姿态）并拟合 β_cross
    Φx = build_cross_features(i, data_all, opt)
    β_cross = size(Φx, 2) == 0 ? zeros(Float64, 0) : (Φx \ r)

    return Dict(
        :pars_base => pars_base,
        :beta_cross => β_cross,
        :Phi_x => Φx,
        :residual_mse => size(Φx,2) == 0 ? mean(r.^2) : mean((r .- Φx*β_cross).^2)
    )
end

# ---------------- Step 2：重构 d̂（含跨轴项） ---------------- #

"""
reconstruct_dhat_rigid(i, data_all, opt, step1_pack)
  先用 pars_base 重构 base d_hat，再把 Φx*β_cross 加上：
  d_hat = d_hat_base + Φx*β_cross
"""
function reconstruct_dhat_rigid(i::Symbol, data_all::Dict{Symbol,AxisData},
                                opt::RigidOptions, step1_pack::Dict{Symbol,Any};
                                reconstruct_d1d = Main.reconstruct_disturbance_1d)
    ai = data_all[i]
    qd = isnothing(ai.qd) ? vcat((ai.q[2]-ai.q[1])/ai.Ts, diff(ai.q)./ai.Ts) : ai.qd
    qdd= isnothing(ai.qdd) ? vcat((qd[2]-qd[1])/ai.Ts, diff(qd)./ai.Ts) : ai.qdd

    d_hat_base = reconstruct_d1d(ai.q, qd, qdd, step1_pack[:pars_base], opt.grid_centers, opt.grid_halfwidth)
    Φx = step1_pack[:Phi_x]
    β = step1_pack[:beta_cross]
    d_hat = size(Φx,2) == 0 ? d_hat_base : (d_hat_base .+ Φx*β)
    return d_hat
end

# ---------------- Step 3：共分母 2×2 闭环拟合 ---------------- #


# --------------------------------------------
# 获取最后的传递函数模型

end # module Model