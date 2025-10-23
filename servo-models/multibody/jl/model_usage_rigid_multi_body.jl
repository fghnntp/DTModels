# model_usage.jl
# ============================================================
# Step1: 开环机电参数辨识
# Step2: 扰动重构
# Step3: 2×2 闭环辨识 + 反推 S/Gfb/Gff；以及用 Geqr 直接从 q_r 预测 e
#
# 需要同目录下：
#   - machine_model.jl           # Step1/Step2 Julia 版本
#   - closed_loop_id.jl  # Step3 Julia 版本
# ============================================================

##
using Pkg
Pkg.activate(".")
## 
using LinearAlgebra
using Plots
using Printf
using Random
using Statistics

include("machine_model.jl")
include("closed_loop_id.jl")
include("machine_model_rigid_multi_body.jl")  # 包含 MultiBodyRigidModel
include("utilitys.jl")  # 包含 gp_from_mck 等辅助函数


## 0) 生成数据
N  = 2000
Ts = 0.015
d_hat_r_l, q_r_l, e_l, u_l, q_l = generate_signal(N=N, Ts=Ts, seed=42)


# 画原始信号
plot(layout=(4,1), size=(900,600), title="Generated Original System Signals")
plot!(subplot=1, d_hat_r_l, label="d_hat_r")
plot!(subplot=2, q_r_l,     label="q_r")
plot!(subplot=3, e_l,       label="e")
plot!(subplot=4, u_l,       label="u")


d_hat_r_r, q_r_r, e_r, u_r, q_r = generate_signal(N=N, Ts=Ts, seed=1428)
# 画原始信号
plot(layout=(4,1), size=(900,600), title="Generated Original System Signals")
plot!(subplot=1, d_hat_r_r, label="d_hat_r")
plot!(subplot=2, q_r_r,     label="q_r")
plot!(subplot=3, e_r,       label="e")
plot!(subplot=4, u_r,       label="u")

dataL = MultiBodyRigidModel.AxisData(q=q_l, qr=q_r_l, u=u_l, Ts=Ts)
dataR = MultiBodyRigidModel.AxisData(q=q_r, qr=q_r_r, u=u_r, Ts=Ts)


# 选项（可按 L/R 分别设置 Step1 搜索空间；Step3 给定分母阶次等）
opt = MultiBodyRigidModel.RigidPairOptions(
  step1_L = MultiBodyRigidModel.Step1Options(grid_centers = Float64[], grid_halfwidth=0.0, n_grid=0,
                         n_random=80, seed=1),
  step1_R = MultiBodyRigidModel.Step1Options(grid_centers = Float64[], grid_halfwidth=0.0, n_grid=0,
                         n_random=80, seed=2),
  cross   = MultiBodyRigidModel.CrossFeatOptions(include_cross=true, include_trig_R=true),
  step3   = MultiBodyRigidModel.Step3Options(nden=4, nb=2, n_random=120, seed=3)
)

packs = MultiBodyRigidModel.identify_pair_rigid(dataL, dataR, opt)
# 拆出离散 2×2（z⁻¹ 形式）以便频域/仿真
Ged_L, Geqr_L, GCd_L, GCqr_L = MultiBodyRigidModel.extract_2x2_tfz(packs[:L][:step3])
Ged_R, Geqr_R, GCd_R, GCqr_R = MultiBodyRigidModel.extract_2x2_tfz(packs[:R][:step3])

# 快速查看收益
MultiBodyRigidModel.pretty_report(packs[:L], prefix="[L] ")
MultiBodyRigidModel.pretty_report(packs[:R], prefix="[R] ")

idenL = packs[:L][:step3]
idenR = packs[:R][:step3]

e_hat_l = e_from_qr_batch(q_r_l, idenL[:b_eqr], idenL[:a])
e_hat_r = e_from_qr_batch(q_r_r, idenR[:b_eqr], idenR[:a])

plt1 = plot(e_l,     label="e")
plot!(plt1, e_hat_l, label="e_hat", ls=:dash, title="e vs. e_hat", xlabel="Sample",
      grid=:on, legend=:best, size=(900,350))

residuals_l = e_l .- e_hat_l
mse_l = mean(residuals_l.^2)
mae_l = mean(abs.(residuals_l))  # 平均绝对误差
max_error_l = maximum(abs.(residuals_l))  # 最大绝对误差

println("MSE: $mse_l")
println("MAE: $mae_l")
println("Max Error: $max_error_l")

pltr = plot(e_r,     label="e")
plot!(pltr, e_hat_r, label="e_hat", ls=:dash, title="e vs. e_hat", xlabel="Sample",
      grid=:on, legend=:best, size=(900,350))
residuals_r = e_r .- e_hat_r
mse_r = mean(residuals_r.^2)
mae_r = mean(abs.(residuals_r))  # 平均绝对误差
max_error_r = maximum(abs.(residuals_r))  # 最大绝对误
println("MSE: $mse_r")
println("MAE: $mae_r")
println("Max Error: $max_error_r")