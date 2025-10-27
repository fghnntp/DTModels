using LinearAlgebra
# —— 正交 Procrustes：找 S→T 的刚性旋转 R_TS，使 R_TS*ωS ≈ ωpred_T
function orthogonal_procrustes(A::Matrix{Float64}, B::Matrix{Float64})
    # 找 R 使 R*A ≈ B ; 这里 A=ωS, B=ωpred_T （都是 3×N）
    U, _, Vt = svd(B * A')
    R = U * Vt
    if det(R) < 0
        U[:,end] .= -U[:,end]
        R = U * Vt
    end
    return R
end

# —— 由陀螺得到 qδ 系列，并投影到两条物理方向 nL/nR
function qdelta_from_gyro(ωS::Matrix{Float64}, ωpred_T::Matrix{Float64},
                          nL::Vector{Float64}, nR::Vector{Float64}, Ts::Float64)
    R_TS = orthogonal_procrustes(ωS, ωpred_T)   # S→T
    ωmeas_T = R_TS * ωS                         # 3×N
    ωδ = ωmeas_T .- ωpred_T                     # 局部相对角速度（T系）

    qdδ_L = vec(nL' * ωδ)   # 1×N → N
    qdδ_R = vec(nR' * ωδ)

    # 简单零相位高通（抑制积分漂移）
    function hp(x; α=0.995)
        y = similar(x)
        y[1] = x[1]
        @inbounds for k in 2:length(x)
            y[k] = α*(y[k-1] + x[k] - x[k-1])
        end
        y
    end

    qδ_L = hp(cumsum(qdδ_L) .* Ts)
    qδ_R = hp(cumsum(qdδ_R) .* Ts)

    qddδ_L = vcat((qdδ_L[2]-qdδ_L[1])/Ts, diff(qdδ_L)./Ts)
    qddδ_R = vcat((qdδ_R[2]-qdδ_R[1])/Ts, diff(qdδ_R)./Ts)

    return (qδ_L=qδ_L, qdδ_L=qdδ_L, qddδ_L=qddδ_L,
            qδ_R=qδ_R, qdδ_R=qdδ_R, qddδ_R=qddδ_R)
end


# 例：两轴 q = [qL; qR]，其中 L 是直线轴（只影响平动），R 是绕台面 x 轴旋转
# 则 Jω(q) = [0  1;  0 0;  0 0]  * [q̇L; q̇R]  （示意：只有 R 贡献角速度）
# 注意：实际机型请用真实几何/Denavit-Hartenberg/产品手册计算 Jω
function omega_pred_from_jacobian(qL::Vector{Float64}, qR::Vector{Float64},
                                  qdL::Vector{Float64}, qdR::Vector{Float64})
    N = length(qL)
    ωpred_T = zeros(Float64, 3, N)
    @inbounds for k in 1:N
        # 示意：R 为 A 轴，绕 Tx 旋转 → 角速度 [qdR, 0, 0]
        ωpred_T[:,k] = [qdR[k], 0.0, 0.0]
        # 如果 L 直线轴伴随“刚体俯仰/滚转”（某些头架结构会这样），把对应项加进去
        # ωpred_T[:,k] .+= [α(qL[k])*qdL[k], β(qL[k])*qdL[k], γ(qL[k])*qdL[k]]
    end
    return ωpred_T
end

function align_by_xcorr(a::Vector{Float64}, b::Vector{Float64})
    # 返回使 a 与 b 最相关的 b 的“右移”样本数（b -> circshift(b, lag)）
    N = length(a)
    m = 2N
    A = fft(vcat(a, zeros(N)))
    B = fft(vcat(b, zeros(N)))
    xc = real.(ifft(A .* conj.(B)))
    lag = argmax(xc) - 1
    if lag > N
        lag = lag - m
    end
    return lag
end

# 用在角速度范数上
# 例：
# lag = align_by_xcorr(vec(norm.(eachcol(ωpred_T))), vec(norm.(eachcol(R_guess*ωS))))
# 然后把 ωS 用 circshift 沿时间轴平移 lag 个样本

# 你的数据
Ts  # 采样周期

d_hat_r_l, qL, e_l, u_l, q_l = generate_signal(N=N, Ts=Ts, seed=42)
d_hat_r_r, qR, e_r, u_r, q_r = generate_signal(N=N, Ts=Ts, seed=1428)
d_hat_r_l, uL, e_l, u_l, q_l = generate_signal(N=N, Ts=Ts, seed=142)
d_hat_r_l, uR, e_l, u_l, q_l = generate_signal(N=N, Ts=Ts, seed=242)
d_hat_r_l, ωS1, ωS2, ωS3, q_l = generate_signal(N=N, Ts=Ts, seed=242)

ωS = hcat(ωS1, ωS2, ωS3)  # 3×N 矩阵

ωS = transpose(ωS)  # 转成 3×N

ωS = Matrix(ωS)  # 确保是矩阵类型

qL, qR, uL, uR  # 长度 N 的列向量
ωS  # 3×N IMU 陀螺（S系）

# 4.1 得到关节速度
using .MachineModel  # 复用你的 finite_diff
qdL, _ = MachineModel.finite_diff(qL, Ts)
qdR, _ = MachineModel.finite_diff(qR, Ts)

# 4.2 刚体角速预测（示例：R 是绕 Tx 的转轴）
ωpred_T = omega_pred_from_jacobian(qL, qR, qdL, qdR)  # 或者用 omega_pred_from_screws

# （可选）时间对齐：根据互相关求 lag，然后 circshift! ωS
# lag = align_by_xcorr(vec(norm.(eachcol(ωpred_T))), vec(norm.(eachcol(ωS))))
# ωS = circshift(ωS, (0, lag))

# 4.3 选择两条“物理方向” nL, nR（台面系 T 下的单位向量）
#     nR：旋转轴的转轴；nL：对 L 轴最敏感的挠曲方向（初期可先凭经验）
nR = [1.0, 0.0, 0.0]             # 例如 A 轴绕 Tx
nL = [0.0, 1.0, 0.0]             # 举例：设 L 对绕 Ty 的俯仰最敏感；视机型而定

# 4.4 计算六条序列
qδ = qdelta_from_gyro(ωS, ωpred_T, nL, nR, Ts)

# 4.5 喂给 MultiBodyFlexModel
# include("MultiBodyFlexModel.jl")
# using .MultiBodyFlexModel

# dataL = AxisData(q=qL, qr=qrL, u=uL, Ts=Ts,
#                  qδ = qδ.qδ_L, qdδ = qδ.qdδ_L, qddδ = qδ.qddδ_L)
# dataR = AxisData(q=qR, qr=qrR, u=uR, Ts=Ts,
#                  qδ = qδ.qδ_R, qdδ = qδ.qdδ_R, qddδ = qδ.qddδ_R)

# opt = FlexPairOptions(
#   step1_L = Step1Options(n_random=100, seed=1),
#   step1_R = Step1Options(n_random=100, seed=2),
#   cross   = CrossFeatOptions(include_cross=true,
#                              include_trig_R_in_L=true,
#                              include_delta_in_L=true,
#                              include_delta_in_R=true),
#   step3   = Step3Options(nden=6, nb=2, n_random=150, seed=3)
# )

# packs = identify_pair_flex(dataL, dataR, opt)
# pretty_report(packs[:L], prefix="[L] ")
# pretty_report(packs[:R], prefix="[R] ")
