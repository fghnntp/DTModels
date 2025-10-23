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

include("utilitys.jl")  # 包含 gp_from_mck 等辅助函数

## 0) 生成数据
N  = 2000
Ts = 0.015
d_hat_r, q_r, e, u, q = generate_signal(N=N, Ts=Ts, seed=42)

# 画原始信号
plot(layout=(4,1), size=(900,600), title="Generated Original System Signals")
plot!(subplot=1, d_hat_r, label="d_hat_r")
plot!(subplot=2, q_r,     label="q_r")
plot!(subplot=3, e,       label="e")
plot!(subplot=4, u,       label="u")

## 1) Step 1：辨识 MΔ, C, K, fC, fS, v_s, dv, m_grid
include("machine_model.jl")

# 用真实输出 q（= q_r - e）来求速度与加速度
dq_sig, ddq_sig = MachineModel.finite_diff(q, Ts)

# 位置网格
Ng = 9
qmin = MachineModel.quantile(q, 0.01);
qmax = MachineModel.quantile(q, 0.99);
grid_centers = collect(range(qmin, qmax; length=Ng))
grid_halfwidth = (qmax - qmin)/(Ng - 1)

iden1 = MachineModel.Step1Identifier1D(grid_centers=grid_centers, grid_halfwidth=grid_halfwidth,
                          n_grid=Ng, fric_v_s_bounds=(1e-4, 1.0),
                          fric_dv_bounds=(1e-5, 1e-2),
                          n_random=30, seed=0)
pars = MachineModel.run(iden1, q, dq_sig, ddq_sig, u)

M_delta = pars["M_delta"]; C = pars["C"]; K = pars["K"]
fC = pars["fC"]; fS = pars["fS"]
v_s = pars["v_s"]; dv = pars["dv_band"]
m_grid = pars["m_grid"]
@printf "Step1: M=%.4g, C=%.4g, K=%.4g, fC=%.4g, fS=%.4g, v_s=%.4g, dv=%.4g\n" M_delta C K fC fS v_s dv


## 2) Step 2：重构扰动
d_hat = MachineModel.reconstruct_disturbance_1d(q, dq_sig, ddq_sig, pars, grid_centers, grid_halfwidth)

## 3) Step 3：2×2 闭环识别 + 反推 S/Gfb/Gff
include("closed_loop_id.jl")

# 3.1 由 Step1 的 M,C,K 形成 Gp(z)
b_p, a_p = gp_from_mck(M_delta, C, K, Ts; method="tustin")

# 3.2 识别 2×2
id2 = ClosedLoopID.CL2x2Identifier(nden=3, nb=3, n_random=100, seed=0)
iden = ClosedLoopID.run(id2, d_hat, q_r, e, u)

# 3.3 频域反解
w = range(0, stop=π, length=1024) |> collect
fr = ClosedLoopID.recover_gfb_gff_from_2x2( (iden[:b_ed], iden[:a]),
                               (iden[:b_eqr], iden[:a]),
                               (b_p, a_p), w )
S   = fr[:S];   Gfb = fr[:Gfb];   Gff = fr[:Gff]
Gp  = freq_response_num_den(b_p, a_p, w)

# 可视化（如需可打开）
# display(plot_frf_grid(w, [("S", S), ("1-S", 1 .- S)]; title_prefix="Sensitivity"))
# display(plot_frf_grid(w, [("Gfb", Gfb)]; title_prefix="Feedback Gfb"))
# display(plot_frf_grid(w, [("Gff", Gff)]; title_prefix="Feedforward Gff"))
# display(plot_frf_grid(w, [("Gp",  Gp)];  title_prefix="Plant Gp"))

# 4) 用 Geqr 批处理直接从 q_r 预测 e
e_hat = e_from_qr_batch(q_r, iden[:b_eqr], iden[:a])

plt1 = plot(e,     label="e")
plotlyjs()
plot!(plt1, e_hat, label="e_hat", ls=:dash, title="e vs. e_hat", xlabel="Sample",
      grid=:on, legend=:best, size=(900,350))

residuals = e .- e_hat
mse = mean(residuals.^2)
mae = mean(abs.(residuals))  # 平均绝对误差
max_error = maximum(abs.(residuals))  # 最大绝对误差

println("MSE: $mse")
println("MAE: $mae") 
println("Max Error: $max_error")


# 5) （可选）在线逐点版本
# e_hat_stream = [step!(ErrorFromQrOnline(iden[:b_eqr], iden[:a]), x) for x in q_r]
# e_hat_stream = collect(e_hat_stream)
