module MultiBodyRigidModel

# ===============================================================
# MultiBodyRigidModel — 论文 Sec.3.1（刚性连接，无 IMU）两轴建模
# Julia v1.11.7
#
# 依赖：
#   • MachineModel  （你已实现的 Step1/Step2：开环 + 扰动重构）
#   • ClosedLoopID  （已实现的 Step3：2×2 共享分母闭环识别）
#
# 本模块职责：
#   1) 为直线轴 L 与旋转轴 R 组织数据、差分获得 q̇/q̈（若未给）。
#   2) 逐轴执行 Step1 随机搜索 + 最小二乘（基于 MachineModel.run）。
#   3) （可选）在残差上用“跨轴/姿态”特征 Φx 做一轮 LS：吸收 L↔R 耦合到扰动中。
#   4) Step2 生成 d̂_L, d̂_R。
#   5) Step3 调用 ClosedLoopID.CL2x2Identifier 对每轴拟合 2×2（输入 [d̂, q_r] → 输出 [e, u]）。
#
# 结果：
#   返回每轴一个字典：pars_base（Step1 参数）、beta_cross（跨轴系数）、d_hat（扰动）、
#   以及 Step3 的 {a, b_* , obj}。
# ===============================================================

using LinearAlgebra
using Statistics
using Random

# 显式引用上层模块（请在外部 include("machine_model.jl"); include("closed_loop_id.jl")）
const _MM = Main.MachineModel
const _CL = Main.ClosedLoopID

# -------------------------- 数据与选项 -------------------------- #

Base.@kwdef mutable struct AxisData
    q::Vector{Float64}
    qr::Vector{Float64}
    u::Vector{Float64}
    Ts::Float64
    qd::Union{Nothing,Vector{Float64}} = nothing
    qdd::Union{Nothing,Vector{Float64}} = nothing
end

Base.@kwdef mutable struct Step1Options
    # 位置格点（用于 PositionGrid1D）；若不需要位置基，设 n_grid=0
    grid_centers::Vector{Float64} = Float64[]
    grid_halfwidth::Float64 = 0.0
    n_grid::Int = 0

    # 摩擦搜索范围（传给 Step1Identifier1D）
    fric_v_s_bounds::Tuple{Float64,Float64} = (1e-4, 1.0)
    fric_dv_bounds::Tuple{Float64,Float64} = (1e-5, 1e-2)
    n_random::Int = 60
    seed::Int = 0
end

Base.@kwdef mutable struct CrossFeatOptions
    include_cross::Bool = true             # 是否在残差上拟合跨轴/姿态特征
    include_trig_R::Bool = true            # 在 L 的特征里加入 sin/cos(qR)
    include_trig_L::Bool = false           # 在 R 的特征里加入 sin/cos(qL)（通常不需要）
end

Base.@kwdef mutable struct Step3Options
    nden::Int = 4          # 共享分母阶次
    nb::Int = 2            # 每条分子阶次
    n_random::Int = 80     # 随机极点样本数
    seed::Int = 0
end

Base.@kwdef mutable struct RigidPairOptions
    step1_L::Step1Options = Step1Options()
    step1_R::Step1Options = Step1Options()
    cross::CrossFeatOptions = CrossFeatOptions()
    step3::Step3Options = Step3Options()
end

# ----------------------- 差分与特征构造 ----------------------- #

"""
    ensure_derivatives!(ad::AxisData)
若 qd/qdd 为空，则用中央差分近似（端点用前/后向）填充。
"""
function ensure_derivatives!(ad::AxisData)
    if isnothing(ad.qd) || isnothing(ad.qdd)
        dx, ddx = _MM.finite_diff(ad.q, ad.Ts)
        ad.qd  = dx
        ad.qdd = ddx
    end
    return ad
end

# 构造“跨轴/姿态”特征 Φx（T×K）：用于在 Step1 残差上做 LS
function build_cross_feats_L(dataL::AxisData, dataR::AxisData, opt::CrossFeatOptions)
    T = length(dataL.q)
    cols = Vector{Vector{Float64}}()
    if opt.include_cross
        push!(cols, dataR.qd, dataR.qdd)
        if opt.include_trig_R
            push!(cols, sin.(dataR.q), cos.(dataR.q))
        end
    end
    return isempty(cols) ? zeros(Float64, T, 0) : hcat(cols...)
end

function build_cross_feats_R(dataR::AxisData, dataL::AxisData, opt::CrossFeatOptions)
    T = length(dataR.q)
    cols = Vector{Vector{Float64}}()
    if opt.include_cross
        push!(cols, dataL.qd, dataL.qdd)
        if opt.include_trig_L
            push!(cols, sin.(dataL.q), cos.(dataL.q))
        end
    end
    return isempty(cols) ? zeros(Float64, T, 0) : hcat(cols...)
end

# --------------------------- Step1/2 --------------------------- #

"""
    run_step1_axis!(ad::AxisData, s1opt::Step1Options)
调用你已有的一步法：随机搜索 (v_s,dv) + LS，返回 pars_base::Dict{String,Any}。
"""
function run_step1_axis!(ad::AxisData, s1opt::Step1Options)
    id = _MM.Step1Identifier1D(
        grid_centers = s1opt.grid_centers,
        grid_halfwidth = s1opt.grid_halfwidth,
        n_grid = s1opt.n_grid,
        fric_v_s_bounds = s1opt.fric_v_s_bounds,
        fric_dv_bounds = s1opt.fric_dv_bounds,
        n_random = s1opt.n_random,
        seed = s1opt.seed,
    )
    return _MM.run(id, ad.q, ad.qd, ad.qdd, ad.u)
end

"""
    reconstruct_dhat_axis(ad, pars_base, s1opt)
按 pars_base 重构 d̂_base；返回向量。
"""
function reconstruct_dhat_axis(ad::AxisData, pars_base::Dict{String,Any}, s1opt::Step1Options)
    return _MM.reconstruct_disturbance_1d(ad.q, ad.qd, ad.qdd,
        pars_base, s1opt.grid_centers, s1opt.grid_halfwidth)
end

# 在残差上拟合跨轴特征： r ≈ Φx β_cross
function fit_cross_on_residual(u::Vector{Float64}, uhat_base::Vector{Float64}, Φx::Matrix{Float64})
    r = u .- uhat_base
    β = size(Φx,2) == 0 ? zeros(Float64, 0) : (Φx \ r)
    return β, r, (size(Φx,2) == 0 ? mean(r.^2) : mean((r .- Φx*β).^2))
end

# --------------------------- Step3 --------------------------- #

"""
    run_step3_axis(ad::AxisData, d̂::Vector{Float64}, s3opt::Step3Options)
用共享分母的 2×2 识别（ClosedLoopID.CL2x2Identifier）。
返回 Dict(:a, :b_ed, :b_eqr, :b_Cd, :b_Cqr, :obj)。
"""
function run_step3_axis(ad::AxisData, d̂::Vector{Float64}, s3opt::Step3Options)
    e = ad.qr .- ad.q
    id = _CL.CL2x2Identifier(nden = s3opt.nden, nb = s3opt.nb,
                              n_random = s3opt.n_random, seed = s3opt.seed)
    return _CL.run(id, d̂, ad.qr, e, ad.u)
end

# -------------------------- 顶层入口 -------------------------- #

"""
identify_pair_rigid(dataL::AxisData, dataR::AxisData, opt::RigidPairOptions)
→ 返回 Dict(:L=>packL, :R=>packR)
  • 每个 pack 含：
      :pars_base, :beta_cross, :Phi_x, :dhat_base, :dhat,
      :step1_mse, :step1_mse_after_cross,
      :step3 (内含 :a, :b_ed, :b_eqr, :b_Cd, :b_Cqr, :obj)
"""
function identify_pair_rigid(dataL::AxisData, dataR::AxisData, opt::RigidPairOptions)
    # 统一导数
    ensure_derivatives!(dataL); ensure_derivatives!(dataR)

    # ===== 直线轴 L =====
    parsL = run_step1_axis!(dataL, opt.step1_L)
    dhatL_base = reconstruct_dhat_axis(dataL, parsL, opt.step1_L)
    ΦxL = build_cross_feats_L(dataL, dataR, opt.cross)
    βL, rL, mseL_after = fit_cross_on_residual(dataL.u, dhatL_base, ΦxL)
    dhatL = size(ΦxL,2) == 0 ? dhatL_base : (dhatL_base .+ ΦxL*βL)
    mseL = mean((dataL.u .- dhatL_base).^2)

    stp3L = run_step3_axis(dataL, dhatL, opt.step3)

    packL = Dict(
        :pars_base => parsL,
        :beta_cross => βL,
        :Phi_x => ΦxL,
        :dhat_base => dhatL_base,
        :dhat => dhatL,
        :step1_mse => mseL,
        :step1_mse_after_cross => mseL_after,
        :step3 => stp3L,
    )

    # ===== 旋转轴 R =====
    parsR = run_step1_axis!(dataR, opt.step1_R)
    dhatR_base = reconstruct_dhat_axis(dataR, parsR, opt.step1_R)
    ΦxR = build_cross_feats_R(dataR, dataL, opt.cross)
    βR, rR, mseR_after = fit_cross_on_residual(dataR.u, dhatR_base, ΦxR)
    dhatR = size(ΦxR,2) == 0 ? dhatR_base : (dhatR_base .+ ΦxR*βR)
    mseR = mean((dataR.u .- dhatR_base).^2)

    stp3R = run_step3_axis(dataR, dhatR, opt.step3)

    packR = Dict(
        :pars_base => parsR,
        :beta_cross => βR,
        :Phi_x => ΦxR,
        :dhat_base => dhatR_base,
        :dhat => dhatR,
        :step1_mse => mseR,
        :step1_mse_after_cross => mseR_after,
        :step3 => stp3R,
    )

    return Dict(:L=>packL, :R=>packR)
end

# -------------------------- 便捷工具 -------------------------- #

"""
    extract_2x2_tfz(step3_pack)
把 Step3 结果拆成 2×2 的离散传递函数（z⁻¹ 形式）。
返回 (Ged, Geqr, GCd, GCqr) 四个 (b, a) 对。
"""
function extract_2x2_tfz(step3_pack::Dict{Symbol,Any})
    a    = step3_pack[:a]
    b_ed = step3_pack[:b_ed]
    b_eqr= step3_pack[:b_eqr]
    b_Cd = step3_pack[:b_Cd]
    b_Cqr= step3_pack[:b_Cqr]
    return ( (b_ed, a), (b_eqr, a), (b_Cd, a), (b_Cqr, a) )
end

"""
    pretty_report(pack::Dict)
打印该轴的识别摘要：Step1 残差、跨轴收益、Step3 目标值等。
"""
function pretty_report(pack::Dict{Symbol,Any}; prefix="")
    println(prefix, "Step1 MSE(base) = ", round(pack[:step1_mse], digits=6),
            ",  after cross = ", round(pack[:step1_mse_after_cross], digits=6))
    println(prefix, "Step3 objective = ", round(pack[:step3][:obj], digits=6))
    println(prefix, "beta_cross length = ", length(pack[:beta_cross]))
end

end # module
