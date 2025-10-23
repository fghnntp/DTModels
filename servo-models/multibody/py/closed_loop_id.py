# closed_loop_id.py
# Closed-loop internal transfer model & identification (Step 3)
# - Identifies the 2x2 LTI map from inputs [d_hat, q_r] to outputs [e, u]
# - Uses common-denominator IIR (discrete-time) with outer pole search + inner LS for numerators
# - Recovers G_fb(z) and G_ff(z) given a plant estimate Gp(z) from Step 1

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List

# -----------------------------
# Utilities
# -----------------------------

def poly_from_poles(poles: np.ndarray) -> np.ndarray:
    """Return monic denominator a(z) = 1 + a1 z^-1 + ... + an z^-n from z-plane poles."""
    coeff_desc = np.poly(poles)        # x^n + c1 x^{n-1} + ... + cn
    a = np.real_if_close(coeff_desc).astype(float)
    a = a / a[0]                       # make monic
    return a                           # [1, a1, ..., an]

def stable_random_poles(n: int, rmin=0.3, rmax=0.95, rng=None) -> np.ndarray:
    """Sample n stable poles inside the unit circle (complex pairs + reals)."""
    rng = np.random.default_rng() if rng is None else rng
    poles = []
    i = 0
    while i < n:
        if n - i >= 2 and rng.random() < 0.6:  # prefer some complex pairs
            r = rng.uniform(rmin, rmax)
            th = rng.uniform(0.05*np.pi, 0.95*np.pi)
            p = r * np.exp(1j*th)
            poles += [p, np.conj(p)]
            i += 2
        else:
            r = rng.uniform(rmin, rmax)
            sgn = -1 if rng.random() < 0.5 else 1
            poles.append(sgn * r)
            i += 1
    return np.array(poles[:n])

def lfilter_den(x: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Filter y via (1 + a1 z^-1 + ... + an z^-n) y = x  (causal IIR denominator)."""
    n = len(a) - 1
    x = np.asarray(x).reshape(-1)
    N = len(x)
    y = np.zeros(N)
    for k in range(N):
        acc = x[k]
        for i in range(1, n+1):
            if k - i >= 0:
                acc -= a[i] * y[k-i]
        y[k] = acc
    return y

def build_regressor(inputs: List[np.ndarray], a: np.ndarray, nb: int) -> np.ndarray:
    """For inputs [x1, x2, ...], build regressor with denominator 'a' and nb numerator taps.
    Columns are [phi_{x1,0} ... phi_{x1,nb} | phi_{x2,0} ... phi_{x2,nb} | ...],
    where phi_{xm,j} = lfilter_den(z^{-j} xm, a).
    """
    cols = []
    for x in inputs:
        x = np.asarray(x).reshape(-1)
        for j in range(nb+1):
            # Delay by j samples (causal), then pass through 1/a(z) filter
            xj = np.concatenate([np.zeros(j), x[:-j] if j>0 else x])
            cols.append(lfilter_den(xj, a))
    return np.column_stack(cols)  # shape (N, n_inputs*(nb+1))

def freq_response_num_den(b: np.ndarray, a: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Frequency response of H(z) = B(z^-1)/A(z^-1) on grid w (rad/sample)."""
    z = np.exp(1j*w)
    num = np.zeros_like(z, dtype=complex)
    den = np.ones_like(z, dtype=complex)
    for i, bi in enumerate(b):
        num += bi * z**(-i)
    for i in range(1, len(a)):
        den += a[i] * z**(-i)
    return num / den

# -----------------------------
# 2x2 closed-loop identification
# -----------------------------

@dataclass
class CL2x2Identifier:
    nden: int                 # common denominator order
    nb: int                   # numerator order per SISO
    n_random: int = 60        # number of pole samples
    seed: int = 0

    def run(self, d_hat: np.ndarray, q_r: np.ndarray, e: np.ndarray, u: np.ndarray) -> Dict:
        """Identify Ged, Geqr, GCd, GCqr with a common denominator.
        Returns dict with keys: 'a','b_ed','b_eqr','b_Cd','b_Cqr','obj'.
        """
        rng = np.random.default_rng(self.seed)
        x_inputs = [np.asarray(d_hat).reshape(-1), np.asarray(q_r).reshape(-1)]
        y_e = np.asarray(e).reshape(-1)
        y_u = np.asarray(u).reshape(-1)

        best = {"obj": np.inf}
        for _ in range(self.n_random):
            poles = stable_random_poles(self.nden, rng=rng)
            a = poly_from_poles(poles)                  # candidate common denominator

            Phi = build_regressor(x_inputs, a, self.nb) # (N, 2*(nb+1)) for inputs [d, qr]

            # Block-diagonal regressor for two outputs: [Phi 0; 0 Phi]
            Phi_blk = np.block([[Phi, np.zeros_like(Phi)],
                                [np.zeros_like(Phi), Phi]])
            Y = np.concatenate([y_e, y_u])

            theta, *_ = np.linalg.lstsq(Phi_blk, Y, rcond=None)

            # Unpack four numerators in order: [b_ed, b_eqr, b_Cd, b_Cqr]
            seg = (self.nb + 1)
            b_ed   = theta[0*seg : 1*seg]
            b_eqr  = theta[1*seg : 2*seg]
            b_Cd   = theta[2*seg : 3*seg]
            b_Cqr  = theta[3*seg : 4*seg]

            # Reconstruct outputs for objective
            e_hat = Phi @ np.concatenate([b_ed,  b_eqr])
            u_hat = Phi @ np.concatenate([b_Cd, b_Cqr])
            resid = np.concatenate([y_e - e_hat, y_u - u_hat])
            obj = np.mean(resid**2)

            if obj < best["obj"]:
                best.update({
                    "obj": obj, "a": a,
                    "b_ed": b_ed, "b_eqr": b_eqr,
                    "b_Cd": b_Cd, "b_Cqr": b_Cqr
                })
        return best

# -----------------------------
# Recover Gfb/Gff from 2x2 + plant
# -----------------------------

def recover_gfb_gff_from_2x2(Ged: Tuple[np.ndarray,np.ndarray],
                             Geqr: Tuple[np.ndarray,np.ndarray],
                             Gp: Tuple[np.ndarray,np.ndarray],
                             w: np.ndarray) -> Dict:
    """Recover frequency responses of S, Gfb, Gff on w (rad/sample).
    Inputs: Ged=(b_ed,a), Geqr=(b_eqr,a), Gp=(b_p,a_p).
    """
    b_ed, a   = Ged
    b_eqr, a2 = Geqr
    b_p, a_p  = Gp

    H_ed = freq_response_num_den(b_ed,  a,  w)   # e / d_hat
    H_eqr= freq_response_num_den(b_eqr, a2, w)   # e / q_r
    H_p  = freq_response_num_den(b_p,   a_p, w)  # plant

    eps = 1e-16
    S   = - H_ed / (H_p + eps)
    Gfb = (1.0 - S) / (S * (H_p + eps))
    Gff = (1.0 - H_eqr / (S + eps)) / (H_p + eps)
    return {"S": S, "Gfb": Gfb, "Gff": Gff}
