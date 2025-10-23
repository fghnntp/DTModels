using Printf
using Random

# ============================================================
# 1) 连续二阶 Gp(s)=1/(M s^2 + C s + K) 的离散化（tustin / zoh_approx）
# 2) 频响辅助函数 + 画图
# 3) 合成一组 (d_hat, q_r, e, u) 演示数据
# ===========================================================

# -------------------------------
# Gp(s) 离散化（与 Python 版等价）
# -------------------------------
"""
    gp_from_mck(M_delta, C, K, Ts; method="tustin")

将 Gp(s) = 1/(M s^2 + C s + K) 离散化得到 (b, a)（z^-1 形式，a[1]=1）
method ∈ {"tustin", "zoh_approx"}；使用等效极点映射 + DC 增益配准。
"""
# 健壮版：自动投影到物理可行域 + 欠/过阻尼分支 + DC 配准
function gp_from_mck(M_delta::Real, C::Real, K::Real, Ts::Real; method::AbstractString="tustin")
    # --- 1) 参数投影到物理可行域 ---
    M = max(abs(M_delta), 1e-12)   # 质量 > 0
    Kp = max(abs(K),       1e-12)   # 刚度 > 0
    Cp = abs(C)                     # 阻尼 ≥ 0

    # --- 2) 连续域二阶标准量 ---
    wn   = sqrt(Kp / M)                         # 自然频率
    zeta = Cp / (2 * sqrt(M * Kp))              # 阻尼比（非负）

    # --- 3) 由 (wn, zeta) 得到离散极点 ---
    if lowercase(method) == "tustin"
        # 这里仍然沿用“等效极点映射 + DC 配准”的简洁近似
        if zeta < 1 - 1e-12
            # 欠阻尼：共轭对
            r  = exp(-zeta * wn * Ts)
            wd = wn * sqrt(1 - zeta^2)
            p1 = r * cis(wd * Ts)
            p2 = conj(p1)
        else
            # 过阻尼：实极点
            σ  = zeta*wn
            ωd = wn * sqrt(max(0.0, zeta^2 - 1)) # 实数
            s1 = -σ + ωd
            s2 = -σ - ωd
            p1 = exp(s1 * Ts)
            p2 = exp(s2 * Ts)
        end
    elseif lowercase(method) == "zoh_approx"
        # 近似 ZOH：使用连续极点的指数映射
        if zeta < 1 - 1e-12
            σ  = zeta*wn
            wd = wn * sqrt(1 - zeta^2)
            s1 = -σ + im*wd
            s2 = -σ - im*wd
        else
            σ  = zeta*wn
            ωd = wn * sqrt(max(0.0, zeta^2 - 1))
            s1 = -σ + ωd
            s2 = -σ - ωd
        end
        p1 = exp(s1 * Ts)
        p2 = exp(s2 * Ts)
    else
        error("method 仅支持 \"tustin\" 或 \"zoh_approx\"")
    end

    # --- 4) 由离散极点构造分母 a(z) = 1 + a1 z^-1 + a2 z^-2 ---
    a = [1.0, -real(p1 + p2), real(p1 * p2)]
    a ./= a[1]  # 单项式

    # --- 5) 二阶分子骨架 + DC 增益配准（使 H(e^{j0}) = 1/Kp）---
    b = [1.0, 0.0, 0.0]
    H1 = (b[1] + b[2] + b[3]) / (1 + a[2] + a[3])  # z = 1
    Gdc = 1.0 / Kp
    b = (Gdc / (H1 + 1e-16)) .* b

    return b, a
end

# -------------------------------
# 频响与绘图辅助
# -------------------------------
db(x) = 20 .* log10.(max.(1e-16, abs.(x)))
# unwrap_phase_deg(x) = rad2deg.(unwrap(angle.(x)))
# x: 复频响向量（或相位向量也行，见下方 overload）
function unwrap_phase_deg(x)
    ϕ = angle.(x)                      # [-π, π)
    N = length(ϕ)
    if N == 0
        return Float64[]
    end
    out = similar(ϕ)
    out[1] = ϕ[1]
    @inbounds for k in 2:N
        d = ϕ[k] - ϕ[k-1]
        # 把相位差规约到 (-π, π]，再累计避免跳变
        out[k] = out[k-1] + rem2pi(d, RoundNearest)
    end
    return rad2deg.(out)
end

"""
    freq_response_num_den(b, a, w)

H(e^{jw})=B/A 的频响（复数向量），z^-1 形式系数，w 为 rad/sample。
"""
function freq_response_num_den(b::AbstractVector, a::AbstractVector, w::AbstractVector)
    z = cis.(w)
    num = zeros(ComplexF64, length(w))
    den = ones(ComplexF64,  length(w))
    for (i, bi) in enumerate(b)         # i=1→b0
        num .+= bi .* (z .^ (-(i-1)))
    end
    for i in 2:length(a)                # a[1] 已在 den 中
        den .+= a[i] .* (z .^ (-(i-1)))
    end
    num ./ den
end

"""
    plot_frf_grid(w, curves; title_prefix="", legend_loc=:best)

curves 是 (label::String, H::Vector{Complex}) 的向量
"""
function plot_frf_grid(w, curves; title_prefix::AbstractString="", legend_loc=:best)
    magplt = plot(w, db(first(curves)[2]), label=first(curves)[1], ylabel="Magnitude (dB)",
                  xlabel="", legend=:none, grid=:on, title="$title_prefix Magnitude")
    for (lab,H) in curves[2:end]
        plot!(magplt, w, db(H), label=lab)
    end
    phplt  = plot(w, unwrap_phase_deg(first(curves)[2]), label=first(curves)[1],
                  ylabel="Phase (deg)", xlabel="Frequency (rad/sample)", legend=legend_loc,
                  grid=:on, title="$title_prefix Phase")
    for (lab,H) in curves[2:end]
        plot!(phplt, w, unwrap_phase_deg(H), label=lab)
    end
    plot(magplt, phplt, layout=(2,1), size=(800,600))
end

# -------------------------------
# 合成 (d_hat, q_r, e, u) 示例序列（与 Python 逻辑等价）
# -------------------------------
"""
    generate_signal(; N=2000, Ts=0.001, seed=42)

返回 (d_hat, q_r, e, u)
"""
function generate_signal(; N::Int=2000, Ts::Real=0.001, seed::Int=42)
    rng = MersenneTwister(seed)
    t = (0:N-1) .* Ts

    q_r = 0.02 .* sin.(2π*1.2 .* t) .+
          0.01 .* sin.(2π*3.5 .* t .+ 0.7) .+
          0.005 .* sin.(2π*7.0 .* t .+ 1.1) .+
          0.0015 .* randn(rng, N)

    d_hat_raw = 0.25 .* sin.(2π*5.0 .* t .+ 0.3) .+
                0.18 .* sin.(2π*8.0 .* t .+ 1.4) .+
                0.05 .* randn(rng, N)

    # SAME 模式的滑动平均（零填充边界），避免 0 索引
    function moving_avg(x::AbstractVector{<:Real}, L::Int)
        if L ≤ 1
            return collect(Float64, x)
        end
        N  = length(x)
        k  = L ÷ 2                 # 窗口中心偏移
        y  = zeros(Float64, N)
        cs = zeros(Float64, N + 1) # cs[1]=0，cs[i+1]=∑_{t=1..i} x[t]
        @inbounds for i in 1:N
            cs[i+1] = cs[i] + x[i]
        end
        @inbounds for i in 1:N
            # 以 i 为中心的 SAME 窗口 [i-k, i-k+L-1]
            s = i - k
            e = s + L - 1
            # 零填充边界 → 截断到 [1, N]
            s_clip = max(1, s)
            e_clip = min(N, e)
            # 用 cs 的“右开一位”公式避免 0 索引：sum = cs[e_clip+1] - cs[s_clip]
            win_sum = cs[e_clip + 1] - cs[s_clip]
            # 与 numpy "same" 一致，仍然用 L 归一化（而不是窗内有效点数）
            y[i] = win_sum / L
        end
        return y
    end


    d_hat = moving_avg(d_hat_raw, 9)

    dq  = vcat((q_r[2]-q_r[1])/Ts, (q_r[3:end] .- q_r[1:end-2])./(2Ts), (q_r[end]-q_r[end-1])/Ts)
    ddq = vcat((q_r[3]-2q_r[2]+q_r[1])/(Ts^2),
               (q_r[3:end] .- 2q_r[2:end-1] .+ q_r[1:end-2])./(Ts^2),
               (q_r[end] - 2q_r[end-1] + q_r[end-2])/(Ts^2))

    ff = moving_avg(0.8 .* ddq, 7)

    q_r_slow = moving_avg(q_r, 51)
    e0 = moving_avg(q_r .- q_r_slow, 9)

    de0 = vcat((e0[2]-e0[1])/Ts, (e0[3:end] .- e0[1:end-2])./(2Ts), (e0[end]-e0[end-1])/Ts)
    fb = moving_avg(-0.6 .* e0 .- 0.02 .* de0, 11)

    u = ff .+ fb .+ 0.02 .* randn(rng, N)

    q_lp1 = moving_avg(u .+ d_hat, 13)
    q     = moving_avg(q_lp1, 13)
    e     = q_r .- q

    @printf "Generated sequences:\n"
    @printf "d_hat: (%d,)  q_r: (%d,)  e: (%d,)  u: (%d,)\n" N N N N

    return d_hat, q_r, e, u, q
end

"""
    e_from_qr_batch(q_r, b_eqr, a)

使用已辨识的 Geqr(z)=B/A，从 q_r 计算 e（批处理）
差分方程： y[k] = Σ b[j] x[k-j] - Σ a[i] y[k-i]，a[1]=1（z^-1 形式）。
"""
function e_from_qr_batch(q_r::AbstractVector, b_eqr::AbstractVector, a::AbstractVector)
    x = collect(Float64, q_r)
    nb = length(b_eqr) - 1
    na = length(a) - 1
    y  = zeros(Float64, length(x))
    @inbounds for k in 1:length(x)
        acc = 0.0
        for j in 0:nb
            k-j ≥ 1 && (acc += b_eqr[j+1] * x[k-j])
        end
        for i in 1:na
            k-i ≥ 1 && (acc -= a[i+1] * y[k-i])
        end
        y[k] = acc
    end
    return y
end

mutable struct ErrorFromQrOnline
    b::Vector{Float64}
    a::Vector{Float64}
    nb::Int
    na::Int
    xbuf::Vector{Float64}
    ybuf::Vector{Float64}
    function ErrorFromQrOnline(b_eqr::AbstractVector, a::AbstractVector)
        @assert abs(a[1] - 1.0) < 1e-12 "a[1] 必须为 1（单项式分母）"
        b = collect(Float64, b_eqr); a = collect(Float64, a)
        nb = length(b) - 1; na = length(a) - 1
        new(b, a, nb, na, zeros(nb), zeros(na))
    end

end
function step!(ef::ErrorFromQrOnline, qr_k::Real)
    acc = ef.b[1] * qr_k
    for j in 1:ef.nb
        acc += ef.b[j+1] * (j ≤ length(ef.xbuf) ? ef.xbuf[j] : 0.0)
    end
    for i in 1:ef.na
        acc -= ef.a[i+1] * (i ≤ length(ef.ybuf) ? ef.ybuf[i] : 0.0)
    end
    if ef.nb > 0
        ef.xbuf = vcat(qr_k, ef.xbuf[1:end-1])
    end
    if ef.na > 0
        ef.ybuf = vcat(acc,  ef.ybuf[1:end-1])
    end
    return acc
end