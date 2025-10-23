using CairoMakie, StaticArrays, LinearAlgebra, Random

# ---------- 工具 ----------
function prep_polyline(V::Vector{SVector{2,Float64}})
    N = length(V)
    seg_vecs = [V[i+1] - V[i] for i in 1:N-1]
    seg_len  = [norm(seg_vecs[i]) for i in 1:N-1]
    cum_s    = zeros(Float64, N)
    for i in 2:N
        cum_s[i] = cum_s[i-1] + seg_len[i-1]
    end
    return V, seg_vecs, seg_len, cum_s
end

function project_point_on_polyline(p::SVector{2,Float64},
                                   V, seg_vecs, seg_len, cum_s;
                                   j0::Int=1, Δ::Int=6)
    N = length(V)
    jL = max(1, j0-Δ); jR = min(N-1, j0+Δ)
    best_d = Inf
    best_j, best_t, best_p = j0, 0.0, V[j0]

    @inbounds for j in jL:jR
        v = seg_vecs[j]; L2 = dot(v, v)
        if L2 == 0
            p_hat = V[j]; d = norm(p - p_hat); t = 0.0
        else
            t = clamp(dot(p - V[j], v) / L2, 0.0, 1.0)
            p_hat = V[j] + t * v
            d = norm(p - p_hat)
        end
        if d < best_d
            best_d = d; best_j, best_t, best_p = j, t, p_hat
        end
    end

    j_star, t_star, p_star = best_j, best_t, best_p
    s_star = cum_s[j_star] + t_star * seg_len[j_star]
    return j_star, t_star, p_star, s_star
end

function contour_align(P_nom::Vector{SVector{2,Float64}},
                       P_act::Vector{SVector{2,Float64}}; Δ::Int=6)
    V, seg_vecs, seg_len, cum_s = prep_polyline(P_nom)
    M = length(P_act)
    s_star = Vector{Float64}(undef, M)
    e      = Vector{Float64}(undef, M)
    e_n    = Vector{Float64}(undef, M)
    e_t    = Vector{Float64}(undef, M)
    p_star = Vector{SVector{2,Float64}}(undef, M)
    p_t    = Vector{SVector{2,Float64}}(undef, M)
    t_hatV = Vector{SVector{2,Float64}}(undef, M)

    j0 = 1
    @inbounds for i in 1:M
        j_star, t_star, p_star_i, s_star_i =
            project_point_on_polyline(P_act[i], V, seg_vecs, seg_len, cum_s; j0=j0, Δ=Δ)
        s_star[i] = s_star_i
        p_star[i] = p_star_i

        t_hat = seg_len[j_star] > 0 ? seg_vecs[j_star] / seg_len[j_star] : SVector(1.0, 0.0)
        t_hatV[i] = t_hat

        Δp  = P_act[i] - p_star_i
        et  = dot(Δp, t_hat)
        pt  = p_star_i + et * t_hat
        en  = norm(Δp - et * t_hat)

        e_t[i] = et
        e_n[i] = en
        e[i]   = norm(Δp)
        p_t[i] = pt

        j0 = j_star
    end
    return (; s_star, e, e_n, e_t, p_star, p_t, t_hatV)
end

# ---------- 示例数据 ----------
N = 300
xs = range(0, 10, length=N)
curve(x) = SVector(x, 0.6sin(0.8x) + 0.15x)   # 你的名义 2D 轨迹可替换这里
P_nom = [curve(x) for x in xs]

Random.seed!(2025)
function noisy_point(x; phase=0.02, en_amp=0.06, et_amp=0.04, noise=0.004)
    p  = curve(x + phase)
    ϵ  = 1e-3
    t_hat = normalize(curve(x+ϵ) - curve(x-ϵ))
    n_hat = SVector(-t_hat[2], t_hat[1])
    p + en_amp*n_hat + et_amp*t_hat + noise*SVector(randn(), randn())
end
P_act = [noisy_point(x) for x in xs]

res = contour_align(P_nom, P_act; Δ=8)
(; s_star, e_n, p_star, p_t) = res

set_theme!(Theme(
    fonts = (; regular = "PingFang SC", bold = "PingFang SC")
))
# ---------- 绘图 ----------
set_theme!(Theme(fontsize=14))
fig = Figure(resolution=(1200, 520))

ax1 = Axis(fig[1,1], title="轮廓误差")
lines!(ax1, getindex.(P_nom, 1), getindex.(P_nom, 2), color=:steelblue, linewidth=3, label="Nominal polyline")
scatter!(ax1, getindex.(P_act, 1), getindex.(P_act, 2), color=:crimson, markersize=6, label="Actual TCP")

idxs = 1:15:length(P_act)  # 稀疏画箭头
arrows!(ax1,
        Point2f.(getindex.(p_star[idxs],1), getindex.(p_star[idxs],2)),
        Vec2f.((P_act[i] - p_star[i]) for i in idxs),
        arrowsize=12, linewidth=2.5, color=:red, label="Δp")
arrows!(ax1,
        Point2f.(getindex.(p_star[idxs],1), getindex.(p_star[idxs],2)),
        Vec2f.((p_t[i] - p_star[i]) for i in idxs),
        arrowsize=12, linewidth=2.5, color=:dodgerblue, label="e_t (tangential)")
arrows!(ax1,
        Point2f.(getindex.(p_t[idxs],1), getindex.(p_t[idxs],2)),
        Vec2f.((P_act[i] - p_t[i]) for i in idxs),
        arrowsize=12, linewidth=2.5, color=:seagreen, label="e_n (normal)")

axislegend(ax1, position=:lt)
hidedecorations!(ax1, grid=false)
hidespines!(ax1)

ax2 = Axis(fig[1,2], title="e_n vs arc length s*", xlabel="s* (arc length)", ylabel="e_n")
lines!(ax2, s_star, e_n, color=:seagreen, linewidth=2)
ax2.xgridvisible = true
ax2.ygridvisible = true

save("contour_2d_demo.png", fig)  # 可选导出
fig
