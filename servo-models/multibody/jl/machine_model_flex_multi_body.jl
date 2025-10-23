module MultiBodyFlexModel

# ===============================================================
# MultiBodyFlexModel — 论文 Sec.3.1/3.3：直线–旋转“柔性连接”（需要局部振动量）
# Julia v1.11.7
#
# 依赖：
#   • MachineModel  （你的 Step1/Step2：开环 + 扰动重构）
#   • ClosedLoopID  （你的 Step3：2×2 共享分母闭环识别）
#
# 适用场景：
#   直线轴 L 和旋转轴 R 之间存在显著柔性（连接/支撑件引入可观测模态）。
#   需要额外的“局部相对振动”观测量（例如由 IMU 重构的 qδ, q̇δ, q̈δ）。
#
# 工作流程（每轴各自执行）：
#   1) Step1 基线：用 MachineModel 的一步法拟合（质量/阻尼/刚度 + 摩擦 + 位置网格）。
#   2) 残差吸收：在 u 残差上回归 Φx = [跨轴 L↔R, 姿态 trig, 局部振动 qδ系列]，得到 β；
#      最终 d̂ = d̂_base + Φx·β。
#   3) Step3 闭环：调用 ClosedLoopID.CL2x2Identifier，识别 2×2（输入 [d̂, q_r] → 输出 [e, u]），
#      四通道共用分母。
#
# 说明：
#   • 本模块不实现 IMU–编码器的“运动学对齐/旋转矩阵标定”。请将已经重构好的 qδ,q̇δ,q̈δ 传入。
#   • 若暂无 qδ 系列，也可以只用跨轴/姿态项（退化为刚性方案）。
# ===============================================================

using LinearAlgebra
using Statistics
using Random

const _MM = Main.MachineModel
const _CL = Main.ClosedLoopID

# ------------------------------ 数据结构 ------------------------------ #

Base.@kwdef mutable struct AxisData
    q::Vector{Float64}
    qr::Vector{Float64}
    u::Vector{Float64}
    Ts::Float64
    qd::Union{Nothing,Vector{Float64}} = nothing
    qdd::Union{Nothing,Vector{Float64}} = nothing
    # 局部相对振动（由 IMU 重构）：
    qδ::Union{Nothing,Vector{Float64}} = nothing
    qdδ::Union{Nothing,Vector{Float64}} = nothing
    qddδ::Union{Nothing,Vector{Float64}} = nothing
end

Base.@kwdef mutable struct Step1Options
    grid_centers::Vector{Float64} = Float64[]
    grid_halfwidth::Float64 = 0.0
    n_grid::Int = 0
    fric_v_s_bounds::Tuple{Float64,Float64} = (1e-4, 1.0)
    fric_dv_bounds::Tuple{Float64,Float64} = (1e-5, 1e-2)
    n_random::Int = 80
    seed::Int = 0
end

Base.@kwdef mutable struct CrossFeatOptions
    include_cross::Bool = true              # 是否包含跨轴项（另一轴的 q̇/q̈）
    include_trig_R_in_L::Bool = true        # L 的特征里加入 sin/cos(qR)
    include_trig_L_in_R::Bool = false       # R 的特征里加入 sin/cos(qL)
    include_delta_in_L::Bool = true         # 在 L 轴扰动里加入 qδ 系列
    include_delta_in_R::Bool = true         # 在 R 轴扰动里加入 qδ 系列
    include_cross_terms::Bool = false       # 是否加入简单交叉乘积（谨慎开启防止病态）
end

Base.@kwdef mutable struct Step3Options
    nden::Int = 6           # 共享分母阶次（柔性场景建议更高：1~2 控制极点 + 2×模态数）
    nb::Int = 2             # 每个分子阶次
    n_random::Int = 120
    seed::Int = 0
end

Base.@kwdef mutable struct FlexPairOptions
    step1_L::Step1Options = Step1Options()
    step1_R::Step1Options = Step1Options()
    cross::CrossFeatOptions = CrossFeatOptions()
    step3::Step3Options = Step3Options()
end

# ------------------------------ 工具函数 ------------------------------ #

function ensure_derivatives!(ad::AxisData)
    if isnothing(ad.qd) || isnothing(ad.qdd)
        dx, ddx = _MM.finite_diff(ad.q, ad.Ts)
        ad.qd  = dx
        ad.qdd = ddx
    end
    # 若局部振动只有位移，补差分
    if !isnothing(ad.qδ)
        if isnothing(ad.qdδ) || isnothing(ad.qddδ)
            qdδ, qddδ = _MM.finite_diff(ad.qδ, ad.Ts)
            ad.qdδ  = qdδ
            ad.qddδ = qddδ
        end
    end
    return ad
end

# 构造 L 轴的 Φx：含 R 的 q̇/q̈、R 的 trig(qR)、以及局部振动 qδ 系列
function build_feats_L(dataL::AxisData, dataR::AxisData, opt::CrossFeatOptions)
    T = length(dataL.q)
    cols = Vector{Vector{Float64}}()
    if opt.include_cross
        push!(cols, dataR.qd, dataR.qdd)
        if opt.include_trig_R_in_L
            push!(cols, sin.(dataR.q), cos.(dataR.q))
        end
    end
    if opt.include_delta_in_L && !isnothing(dataL.qδ)
        push!(cols, dataL.qδ, dataL.qdδ, dataL.qddδ)
    end
    if opt.include_cross_terms && !isempty(cols)
        # 简单交叉：q̇_R * qδ,  q̈_R * qδ （防止病态，需中心化）
        x1 = opt.include_cross ? dataR.qd : zeros(T)
        x2 = (!isnothing(dataL.qδ) && opt.include_delta_in_L) ? dataL.qδ : zeros(T)
        push!(cols, (x1 .- mean(x1)) .* (x2 .- mean(x2)))
    end
    return isempty(cols) ? zeros(Float64, T, 0) : hcat(cols...)
end

# 构造 R 轴的 Φx：含 L 的 q̇/q̈、（可选）L 的 trig(qL)、以及相同的 qδ 系列
function build_feats_R(dataR::AxisData, dataL::AxisData, opt::CrossFeatOptions)
    T = length(dataR.q)
    cols = Vector{Vector{Float64}}()
    if opt.include_cross
        push!(cols, dataL.qd, dataL.qdd)
        if opt.include_trig_L_in_R
            push!(cols, sin.(dataL.q), cos.(dataL.q))
        end
    end
    if opt.include_delta_in_R && !isnothing(dataR.qδ)
        push!(cols, dataR.qδ, dataR.qdδ, dataR.qddδ)
    end
    if opt.include_cross_terms && !isempty(cols)
        x1 = opt.include_cross ? dataL.qd : zeros(T)
        x2 = (!isnothing(dataR.qδ) && opt.include_delta_in_R) ? dataR.qδ : zeros(T)
        push!(cols, (x1 .- mean(x1)) .* (x2 .- mean(x2)))
    end
    return isempty(cols) ? zeros(Float64, T, 0) : hcat(cols...)
end

# ------------------------------ Step1/2 ------------------------------ #

function run_step1_axis!(ad::AxisData, s1::Step1Options)
    id = _MM.Step1Identifier1D(
        grid_centers = s1.grid_centers,
        grid_halfwidth = s1.grid_halfwidth,
        n_grid = s1.n_grid,
        fric_v_s_bounds = s1.fric_v_s_bounds,
        fric_dv_bounds = s1.fric_dv_bounds,
        n_random = s1.n_random,
        seed = s1.seed,
    )
    return _MM.run(id, ad.q, ad.qd, ad.qdd, ad.u)
end

function reconstruct_dhat_axis(ad::AxisData, pars::Dict{String,Any}, s1::Step1Options)
    _MM.reconstruct_disturbance_1d(ad.q, ad.qd, ad.qdd, pars, s1.grid_centers, s1.grid_halfwidth)
end

# r ≈ Φ β —— 线性最小二乘
fit_beta(Φ::Matrix{Float64}, r::Vector{Float64}) = (size(Φ,2) == 0 ? zeros(Float64,0) : (Φ \ r))

# ------------------------------ Step3 ------------------------------ #

function run_step3_axis(ad::AxisData, d̂::Vector{Float64}, s3::Step3Options)
    e = ad.qr .- ad.q
    id = _CL.CL2x2Identifier(nden=s3.nden, nb=s3.nb, n_random=s3.n_random, seed=s3.seed)
    _CL.run(id, d̂, ad.qr, e, ad.u)
end

# ------------------------------ 顶层入口 ------------------------------ #

"""
identify_pair_flex(dataL, dataR, opt) -> Dict(:L=>..., :R=>...)

每个轴的结果字典包含：
  :pars_base, :Phi_x, :beta, :dhat_base, :dhat, :step1_mse, :step1_mse_after, :step3
"""
function identify_pair_flex(dataL::AxisData, dataR::AxisData, opt::FlexPairOptions)
    ensure_derivatives!(dataL); ensure_derivatives!(dataR)

    # ===== L 轴 =====
    parsL      = run_step1_axis!(dataL, opt.step1_L)
    d̂L_base   = reconstruct_dhat_axis(dataL, parsL, opt.step1_L)
    ΦL         = build_feats_L(dataL, dataR, opt.cross)
    rL         = dataL.u .- d̂L_base
    βL         = fit_beta(ΦL, rL)
    d̂L        = size(ΦL,2)==0 ? d̂L_base : (d̂L_base .+ ΦL*βL)
    mseL_base  = mean((dataL.u .- d̂L_base).^2)
    mseL_after = mean((dataL.u .- d̂L   ).^2)
    stp3L      = run_step3_axis(dataL, d̂L, opt.step3)

    packL = Dict(
        :pars_base => parsL,
        :Phi_x => ΦL,
        :beta => βL,
        :dhat_base => d̂L_base,
        :dhat => d̂L,
        :step1_mse => mseL_base,
        :step1_mse_after => mseL_after,
        :step3 => stp3L,
    )

    # ===== R 轴 =====
    parsR      = run_step1_axis!(dataR, opt.step1_R)
    d̂R_base   = reconstruct_dhat_axis(dataR, parsR, opt.step1_R)
    ΦR         = build_feats_R(dataR, dataL, opt.cross)
    rR         = dataR.u .- d̂R_base
    βR         = fit_beta(ΦR, rR)
    d̂R        = size(ΦR,2)==0 ? d̂R_base : (d̂R_base .+ ΦR*βR)
    mseR_base  = mean((dataR.u .- d̂R_base).^2)
    mseR_after = mean((dataR.u .- d̂R   ).^2)
    stp3R      = run_step3_axis(dataR, d̂R, opt.step3)

    packR = Dict(
        :pars_base => parsR,
        :Phi_x => ΦR,
        :beta => βR,
        :dhat_base => d̂R_base,
        :dhat => d̂R,
        :step1_mse => mseR_base,
        :step1_mse_after => mseR_after,
        :step3 => stp3R,
    )

    return Dict(:L=>packL, :R=>packR)
end

# ------------------------------ 便捷函数 ------------------------------ #

function extract_2x2_tfz(step3_pack::Dict{Symbol,Any})
    a    = step3_pack[:a]
    b_ed = step3_pack[:b_ed]
    b_eqr= step3_pack[:b_eqr]
    b_Cd = step3_pack[:b_Cd]
    b_Cqr= step3_pack[:b_Cqr]
    return ( (b_ed, a), (b_eqr, a), (b_Cd, a), (b_Cqr, a) )
end

function pretty_report(pack::Dict{Symbol,Any}; prefix="")
    println(prefix, "Step1 MSE(base) = ", round(pack[:step1_mse], digits=6),
            ", after features = ", round(pack[:step1_mse_after], digits=6),
            ", beta length = ", length(pack[:beta]))
    println(prefix, "Step3 objective = ", round(pack[:step3][:obj], digits=6))
end

end # module
