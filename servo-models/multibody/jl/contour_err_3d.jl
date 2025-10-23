# ============================================
# Contour Error in 3D with GLMakie
# - Nominal 3D polyline
# - Actual points with tangential/normal offsets
# - Windowed nearest-point projection
# - Decomposition: Δp, e_t, e_n (3D arrows)
# ============================================

using GLMakie, StaticArrays, LinearAlgebra, Random

# ---------- Utils ----------
xyzcols(V::Vector{SVector{3,Float64}}) = (
    getindex.(V, 1), getindex.(V, 2), getindex.(V, 3)
)

# Precompute nominal polyline segments
function prep_polyline(V::Vector{SVector{3,Float64}})
    N = length(V)
    seg_vecs = [V[i+1] - V[i] for i in 1:N-1]
    seg_len  = [norm(seg_vecs[i]) for i in 1:N-1]
    cum_s    = zeros(Float64, N)
    for i in 2:N
        cum_s[i] = cum_s[i-1] + seg_len[i-1]
    end
    return V, seg_vecs, seg_len, cum_s
end

# Single-point projection on polyline within a window [j0-Δ, j0+Δ]
function project_point_on_polyline(p::SVector{3,Float64},
                                   V, seg_vecs, seg_len, cum_s;
                                   j0::Int=1, Δ::Int=6)
    N  = length(V)
    jL = max(1, j0-Δ); jR = min(N-1, j0+Δ)

    best_d = Inf
    best_j, best_t, best_p = j0, 0.0, V[j0]

    @inbounds for j in jL:jR
        v = seg_vecs[j]
        L2 = dot(v, v)
        if L2 == 0
            p_hat = V[j]
            d     = norm(p - p_hat)
            t     = 0.0
        else
            t     = clamp(dot(p - V[j], v) / L2, 0.0, 1.0)
            p_hat = V[j] + t * v
            d     = norm(p - p_hat)
        end
        if d < best_d
            best_d = d
            best_j, best_t, best_p = j, t, p_hat
        end
    end

    j_star, t_star, p_star = best_j, best_t, best_p
    s_star = cum_s[j_star] + t_star * seg_len[j_star]
    return j_star, t_star, p_star, s_star
end

# Batch alignment + decomposition (Δp, et, en)
function contour_align(P_nom::Vector{SVector{3,Float64}},
                       P_act::Vector{SVector{3,Float64}}; Δ::Int=6)
    V, seg_vecs, seg_len, cum_s = prep_polyline(P_nom)
    M = length(P_act)

    s_star = Vector{Float64}(undef, M)
    e      = Vector{Float64}(undef, M)
    e_n    = Vector{Float64}(undef, M)
    e_t    = Vector{Float64}(undef, M)
    p_star = Vector{SVector{3,Float64}}(undef, M)
    p_t    = Vector{SVector{3,Float64}}(undef, M)
    t_hatV = Vector{SVector{3,Float64}}(undef, M)

    j0 = 1
    @inbounds for i in 1:M
        j_star, t_star, p_star_i, s_star_i =
            project_point_on_polyline(P_act[i], V, seg_vecs, seg_len, cum_s; j0=j0, Δ=Δ)
        s_star[i] = s_star_i
        p_star[i] = p_star_i

        # Unit tangent on the hit segment
        t_hat = seg_len[j_star] > 0 ? seg_vecs[j_star] / seg_len[j_star] : SVector(1.0, 0.0, 0.0)
        t_hatV[i] = t_hat

        Δp  = P_act[i] - p_star_i
        et  = dot(Δp, t_hat)                  # signed tangential component
        pt  = p_star_i + et * t_hat           # point along tangent
        Δp⊥ = Δp - et * t_hat                 # normal component (vector)
        en  = norm(Δp⊥)

        e_t[i] = et
        e_n[i] = en
        e[i]   = norm(Δp)
        p_t[i] = pt

        j0 = j_star
    end
    return (; s_star, e, e_n, e_t, p_star, p_t, t_hatV)
end

# ---------- Example data (3D) ----------
# Nominal 3D curve (smooth "S" in space), represented as a polyline
N  = 350
ts = range(0, 6π, length=N)
curve(t) = SVector( 2.0t,
                    1.6sin(0.65t),
                    1.1cos(0.40t) )
P_nom = [curve(t) for t in ts]

# Actual points: add phase shift + tangential & normal offsets + noise
Random.seed!(2025)
function ortho_frame(t_hat::SVector{3,Float64})
    # Build an orthonormal basis {t_hat, n1_hat, n2_hat}
    ref = abs(t_hat[1]) < 0.9 ? SVector(1.0, 0.0, 0.0) : SVector(0.0, 1.0, 0.0)
    n1  = normalize(cross(t_hat, ref))
    n2  = normalize(cross(t_hat, n1))
    return n1, n2
end

function noisy_point(t; phase=0.020, en_amp=0.10, et_amp=0.06, noise=0.010)
    p   = curve(t + phase)
    ϵ   = 1e-3
    t_hat = normalize(curve(t+ϵ) - curve(t-ϵ))
    n1, n2 = ortho_frame(t_hat)
    # Put normal offset in some combination of n1/n2 for a 3D deviation
    p + et_amp*t_hat + en_amp*(0.7n1 + 0.3n2) + noise*SVector(randn(), randn(), randn())
end

P_act = [noisy_point(t) for t in ts]

# Compute projection & errors
res = contour_align(P_nom, P_act; Δ=8)
(; p_star, p_t) = res

# ---------- Visualization (3D) ----------
fig = Figure(resolution = (1280, 860), fontsize=14)
ax  = Axis3(fig[1,1], title = "Contour Error in 3D (nearest-point projection & decomposition)",
            xlabel="X", ylabel="Y", zlabel="Z")

# Nominal polyline
x, y, z = xyzcols(P_nom)
lines!(ax, x, y, z, color=:steelblue, linewidth=3)

# Actual points
xa, ya, za = xyzcols(P_act)
scatter!(ax, xa, ya, za, markersize=6, color=:crimson)

# Projection points (sparser to avoid clutter)
idxs = 1:15:length(P_act)
xs, ys, zs = xyzcols(p_star[idxs])
scatter!(ax, xs, ys, zs, markersize=8, color=:seagreen)  # p*

xpt, ypt, zpt = xyzcols(p_t[idxs])
scatter!(ax, xpt, ypt, zpt, markersize=7, color=:orange) # pt

# Helper to draw 3D arrows: origins O, vectors V
using GeometryBasics: Point3f, Vec3f  # Point3f == Point3f32, Vec3f == Vec3f32

# 逐元素构造 origins 和 vectors，避免把整批数据展开成大 NTuple
function draw_arrows!(
    ax,
    P0::Vector{SVector{3,Float64}},
    P1::Vector{SVector{3,Float64}};
    color=:black,
    linewidth=2.0,
    arrowsize=12,
)
    # 起点（Float64 -> Float32）
    O = [Point3f(Float32(p0[1]), Float32(p0[2]), Float32(p0[3])) for p0 in P0]

    # 向量（P1 - P0），同样转 Float32
    V = [Vec3f(Float32(p1[1]-p0[1]),
               Float32(p1[2]-p0[2]),
               Float32(p1[3]-p0[3])) for (p0, p1) in zip(P0, P1)]

    arrows!(ax, O, V; color=color, linewidth=linewidth, arrowsize=arrowsize)
end


# Δp: p* -> p_act (red)
draw_arrows!(ax, p_star[idxs], P_act[idxs]; color=:red, linewidth=2.6, arrowsize=14)
# e_t: p* -> pt (blue)
draw_arrows!(ax, p_star[idxs], p_t[idxs]; color=:dodgerblue, linewidth=2.6, arrowsize=14)
# e_n: pt -> p_act (green)
draw_arrows!(ax, p_t[idxs], P_act[idxs]; color=:seagreen, linewidth=2.6, arrowsize=14)

axislegend(ax,
    [
        LineElement(color=:steelblue, linewidth=3),                     # Nominal polyline
        MarkerElement(marker=:circle,  color=:crimson,  markersize=8),  # Actual TCP
        MarkerElement(marker=:circle,  color=:seagreen, markersize=8),  # Projection p*
        MarkerElement(marker=:circle,  color=:orange,   markersize=8),  # pt (tangential end)
        LineElement(color=:red,        linewidth=3),                    # Δp (arrow color)
        LineElement(color=:dodgerblue, linewidth=3),                    # e_t (arrow color)
        LineElement(color=:seagreen,   linewidth=3)                     # e_n (arrow color)
    ],
    [
        "Nominal polyline",
        "Actual TCP",
        "Projection p*",
        "pt (tangential end)",
        "Δp",
        "e_t (tangential)",
        "e_n (normal)"
    ];
    position = :lt
)


# Camera & aesthetics
hidedecorations!(ax, grid = false)
# hidespines!(ax)   # (optional) spines are not shown in 3D

using Makie: cameracontrols, update_cam!, Vec3f


# Save (optional)
save("contour_3d_glmakie.png", fig)

fig
