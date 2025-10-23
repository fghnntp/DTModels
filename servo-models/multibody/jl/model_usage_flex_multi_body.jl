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
include("machine_model_flex_multi_body.jl")  # 包含 MultiBodyFlexModel
include("utilitys.jl")  # 包含 gp_from_mck 等辅助函数
using .MultiBodyFlexModel

## 0) 生成数据
N = 2000
Ts = 0.015
d_hat_r_l, q_r_l, e_l, u_l, q_l = generate_signal(N=N, Ts=Ts, seed=42)
d_hat_r_r, q_r_r, e_r, u_r, q_r = generate_signal(N=N, Ts=Ts, seed=1428)
qδ_L, _, _, _ = generate_signal(N=N, Ts=Ts, seed=7) .* 0.05  # 局部振动信号示例
qδ_R, _, _, _ = generate_signal(N=N, Ts=Ts, seed=17) .* 0.03

qdδ_L, qddδ_L = MachineModel.finite_diff(qδ_L, Ts)
qdδ_R, qddδ_R = MachineModel.finite_diff(qδ_R, Ts)



# ── 准备数据：把 IMU 重构出的局部振动也喂进来（没有也可留空，退化为“刚性方案”）
dataL = MultiBodyFlexModel.AxisData(q=q_l, qr=q_r_l, u=u_l, Ts=Ts, qδ=qδ_L, qdδ=qdδ_L, qddδ=qddδ_L)
dataR = MultiBodyFlexModel.AxisData(q=q_r, qr=q_r_r, u=u_r, Ts=Ts, qδ=qδ_R, qdδ=qdδ_R, qddδ=qddδ_R)

# 选项：Step1 搜索、特征选择（跨轴/姿态/局部振动）、Step3 分母阶次等
opt = MultiBodyFlexModel.FlexPairOptions(
  step1_L = MultiBodyFlexModel.Step1Options(n_random=100, seed=1),
  step1_R = MultiBodyFlexModel.Step1Options(n_random=100, seed=2),
  cross   = MultiBodyFlexModel.CrossFeatOptions(
              include_cross=true,
              include_trig_R_in_L=true,
              include_delta_in_L=true,
              include_delta_in_R=true,
              include_cross_terms=false
            ),
  step3   = MultiBodyFlexModel.Step3Options(nden=6, nb=2, n_random=150, seed=3)
)

# 识别
packs = MultiBodyFlexModel.identify_pair_flex(dataL, dataR, opt)

# 快速报告
MultiBodyFlexModel.pretty_report(packs[:L], prefix="[L] ")
MultiBodyFlexModel.pretty_report(packs[:R], prefix="[R] ")

# 拆出离散 2×2（z⁻¹ 形式）以便频域/仿真
Ged_L, Geqr_L, GCd_L, GCqr_L = MultiBodyFlexModel.extract_2x2_tfz(packs[:L][:step3])
Ged_R, Geqr_R, GCd_R, GCqr_R = MultiBodyFlexModel.extract_2x2_tfz(packs[:R][:step3])

idenL = packs[:L][:step3]
idenR = packs[:R][:step3]

e_hat_l = e_from_qr_batch(q_r_l, idenL[:b_eqr], idenL[:a])
e_hat_r = e_from_qr_batch(q_r_r, idenR[:b_eqr], idenR[:a])

plt1 = plot(e_l, label="e")
plot!(plt1, e_hat_l, label="e_hat", ls=:dash, title="e vs. e_hat", xlabel="Sample",
      grid=:on, legend=:best, size=(900, 350))

residuals_l = e_l .- e_hat_l
mse_l = mean(residuals_l .^ 2)
mae_l = mean(abs.(residuals_l))  # 平均绝对误差
max_error_l = maximum(abs.(residuals_l))  # 最大绝对误差

println("MSE: $mse_l")
println("MAE: $mae_l")
println("Max Error: $max_error_l")

pltr = plot(e_r, label="e")
plot!(pltr, e_hat_r, label="e_hat", ls=:dash, title="e vs. e_hat", xlabel="Sample",
      grid=:on, legend=:best, size=(900, 350))
residuals_r = e_r .- e_hat_r
mse_r = mean(residuals_r .^ 2)
mae_r = mean(abs.(residuals_r))  # 平均绝对误差
max_error_r = maximum(abs.(residuals_r))  # 最大绝对误
println("MSE: $mse_r")
println("MAE: $mae_r")
println("Max Error: $max_error_r")