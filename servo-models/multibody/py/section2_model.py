
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

# 信号假设
# q, dq, ddq, u 可直接通过CNC采样获取
# q: 位置
# dq: 速度
# ddq: 加速度
# u: 动力学输入，电机力矩

def finite_diff(x: np.ndarray, dt: float):
    dx = np.gradient(x, dt, axis=0)
    ddx = np.gradient(dx, dt, axis=0)
    return dx, ddx

@dataclass
class PositionGrid1D:
    p: np.ndarray
    m: np.ndarray
    Cg: float
    def weights(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q).reshape(-1)
        N = q.shape[0]; Ng = self.p.shape[0]
        Wpos = np.zeros((N, Ng))
        for k in range(Ng):
            dist = np.abs(q - self.p[k])
            wk = 1.0 - dist / self.Cg
            wk = np.where(dist < self.Cg, np.clip(wk, 0.0, 1.0), 0.0)
            Wpos[:, k] = wk
        return Wpos
    def predict(self, q: np.ndarray) -> np.ndarray:
        return self.weights(q) @ self.m

@dataclass
class FrictionStribeck1D:
    """
    Stribeck摩擦模型
    f_fric = fC * dn + fS * exp(-|dq|/v_s)
    库仑摩擦模型
    """
    v_s: float = 1.0
    dv_band: float = 1e-3
    dn0: float = 0.0
    def compute_dn_series(self, dq: np.ndarray, ddq: np.ndarray) -> np.ndarray:
        dq = dq.reshape(-1); ddq = ddq.reshape(-1)
        N = dq.shape[0]
        dn = np.zeros(N); dn_prev = self.dn0
        for k in range(N):
            if k > 0 and np.sign(ddq[k]) * np.sign(dq[k-1]) == -1:
                dn_k = dn_prev
            else:
                inc = 0.0 if k == 0 else (dq[k] - dq[k-1]) / max(self.dv_band, 1e-12)
                dn_k = np.clip(dn_prev + inc, -1.0, 1.0)
            dn[k] = dn_k; dn_prev = dn_k
        return dn
    def basis(self, dq: np.ndarray, ddq: np.ndarray):
        dn = self.compute_dn_series(dq, ddq)
        bC = dn
        bS = np.exp(-np.abs(dq) / max(self.v_s, 1e-12))
        return bC, bS
    def predict(self, dq: np.ndarray, ddq: np.ndarray, fC: float, fS: float) -> np.ndarray:
        bC, bS = self.basis(dq, ddq)
        return bC * fC + bS * fS

@dataclass
class OpenLoopRegressor1D:
    """
    质量-阻尼-刚度系统, 线性系统
    u = MΔ*ddq + C*dq + K*q
    """
    grid: Optional[PositionGrid1D] = None
    fric: Optional[FrictionStribeck1D] = None
    def build(self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray) -> np.ndarray:
        q = q.reshape(-1); dq = dq.reshape(-1); ddq = ddq.reshape(-1)
        cols = [ddq, dq, q]
        if self.fric is not None:
            bC, bS = self.fric.basis(dq, ddq)
            cols += [bC, bS]
        if self.grid is not None:
            Wpos = self.grid.weights(q)
            cols += [Wpos[:, k] for k in range(Wpos.shape[1])]
        return np.column_stack(cols)

@dataclass
class Step1Identifier1D:
    grid_centers: np.ndarray
    grid_halfwidth: float
    n_grid: int
    fric_v_s_bounds: Tuple[float, float] = (1e-4, 1.0)
    fric_dv_bounds: Tuple[float, float] = (1e-5, 1e-2)
    n_random: int = 50
    seed: int = 0
    def run(self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray, u: np.ndarray) -> Dict:
        rng = np.random.default_rng(self.seed)
        best = {"obj": np.inf}
        grid = PositionGrid1D(p=self.grid_centers, m=np.zeros(self.n_grid), Cg=self.grid_halfwidth)
        for _ in range(self.n_random):
            v_s = rng.uniform(*self.fric_v_s_bounds)
            dv = rng.uniform(*self.fric_dv_bounds)
            fric = FrictionStribeck1D(v_s=v_s, dv_band=dv, dn0=0.0)
            W = OpenLoopRegressor1D(grid=grid, fric=fric).build(q, dq, ddq)
            theta, *_ = np.linalg.lstsq(W, u, rcond=None)
            u_hat = W @ theta
            obj = np.mean((u - u_hat) ** 2)
            if obj < best["obj"]:
                best = {"obj": obj, "v_s": v_s, "dv": dv, "theta": theta, "u_hat": u_hat, "W": W}
        theta = best["theta"]
        MΔ, C, K = theta[0], theta[1], theta[2]
        fC, fS = theta[3], theta[4]
        m = theta[5:5+self.n_grid] if self.n_grid > 0 else np.zeros(0)
        return {"M_delta": MΔ, "C": C, "K": K, "fC": fC, "fS": fS, "m_grid": m,
                "v_s": best["v_s"], "dv_band": best["dv"], "u_hat": best["u_hat"],
                "obj": best["obj"], "W": best["W"]}

def reconstruct_disturbance_1d(q: np.ndarray, dq: np.ndarray, ddq: np.ndarray,
                               pars: Dict, grid_centers: np.ndarray, grid_halfwidth: float) -> np.ndarray:
    MΔ, C, K = pars["M_delta"], pars["C"], pars["K"]
    fC, fS = pars["fC"], pars["fS"]
    v_s, dv = pars["v_s"], pars["dv_band"]
    fric = FrictionStribeck1D(v_s=v_s, dv_band=dv, dn0=0.0)
    uf = fric.predict(dq, ddq, fC, fS)
    grid = PositionGrid1D(p=grid_centers, m=pars["m_grid"], Cg=grid_halfwidth)
    up = grid.predict(q)
    u_lin = MΔ * ddq + C * dq + K * q
    d_hat = u_lin + uf + up
    return d_hat
