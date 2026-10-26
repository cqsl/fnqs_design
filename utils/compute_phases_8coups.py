import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Single-site Fourier transform.
#
# The spins are transformed over EVERY lattice translation, z_r = exp(i k . r).
# The 8-coupling pattern has period 2 in x and y, so it does not conserve k:
# it mixes k with k + G for
#
#       G in {(0,0), (pi,0), (0,pi), (pi,pi)}        (index g below)
#
# and the classical problem is a 4x4 Hermitian eigenproblem over that momentum
# quartet.  A trial state is z_r = sum_G c_G exp(i (k+G) . r), and
#
#   M[g, g'](k) = sum_b jhat_b(G_g - G_g') [ exp(i(k+G_g').b)
#                                          + exp(-i(k+G_g).b) ]
#
#   jhat_b(G) = (1/4) sum_s J[s, d] exp(-i G . delta_s)      (real)
#
# diag(M)[g] is exactly the old scalar J_of_Q evaluated at k + G_g with
# cell-averaged couplings; the off-diagonal entries are the harmonics of the
# coupling pattern.  If all jhat(G != 0) vanish the matrix is diagonal, no
# momenta mix, and the whole thing collapses back to the old scalar dispersion.
#
# The couplings are still defined by a 2x2 pattern (that is the model, not the
# method): sublattice s = 2 * (i % 2) + (j % 2) with offset delta_s,
#       s = 0 -> (0, 0)   s = 1 -> (0, 1)   s = 2 -> (1, 0)   s = 3 -> (1, 1)
# and direction d = 0 / 1 within each family:
#       nearest neighbour  d = 0 -> b = (+1,  0)   d = 1 -> b = ( 0, +1)
#       diagonal           d = 0 -> b = (+1, +1)   d = 1 -> b = (+1, -1)
# Every bond carries the coupling of its origin site, so the 8 diagonal
# couplings are [J2_{0,+}, J2_{0,-}, J2_{1,+}, J2_{1,-}, ...] at flat index
# 2 * s + d, exactly as in the Hamiltonian builder.
#
# Index arithmetic: G_g - G_g' within {0, pi}^2 is the XOR of g and g'.
# ---------------------------------------------------------------------------

SUBLATTICE_OFFSETS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

G_VECTORS = jnp.array([[0.0, 0.0], [jnp.pi, 0.0], [0.0, jnp.pi], [jnp.pi, jnp.pi]])

_OFFSETS = jnp.asarray(SUBLATTICE_OFFSETS, dtype=float)

#                (bx,  by, family, d),  family 0 = J1 (nn), 1 = J2 (diagonal)
_BONDS = ((1.0, 0.0, 0, 0), (0.0, 1.0, 0, 1), (1.0, 1.0, 1, 0), (1.0, -1.0, 1, 1))


def as_couplings(J, name="J"):
    """Broadcast a coupling spec to a (4, 2) array. Works on jax tracers.

    scalar / (1,) -> uniform;      (2,) -> per direction;
    (4,)          -> per sublattice;  (8,) or (4, 2) -> full pattern.
    """
    J = jnp.asarray(J)
    if J.ndim == 0:
        return jnp.broadcast_to(J, (4, 2))
    if J.shape == (4, 2):
        return J

    flat = J.reshape(-1)
    n = flat.shape[0]
    if n == 1:
        return jnp.broadcast_to(flat[0], (4, 2))
    if n == 2:
        return jnp.broadcast_to(flat[None, :], (4, 2))
    if n == 4:
        return jnp.broadcast_to(flat[:, None], (4, 2))
    if n == 8:
        return flat.reshape(4, 2)

    raise ValueError(f"{name} must be a scalar or have 1, 2, 4, 8 or (4, 2) entries, " f"got shape {J.shape}")


def harmonics(J):
    """jhat[g, d] = (1/4) sum_s J[s, d] exp(-i G_g . delta_s).

    Real, because every G . delta is 0 or pi.  jhat[0] is the cell average;
    jhat[g != 0] == 0 for all g means the pattern mixes no momenta.
    """
    phases = jnp.cos(G_VECTORS @ _OFFSETS.T)  # exp(-i G.delta) = +-1
    return (phases @ as_couplings(J)) / 4.0  # (4 G, 2 d)


def exchange_matrix(kx, ky, J1, J2):
    """4x4 Hermitian matrix over the momentum quartet k + G.

    Same spectrum as the 4-sublattice construction (they are related by the
    4x4 DFT on the 2x2 cell), but the transform here is over single-site
    translations: no supercell is introduced.
    """
    jh = jnp.stack([harmonics(J1), harmonics(J2)])  # (family, G, d)
    k = jnp.stack([kx, ky])

    rows = []
    for g in range(4):
        row = []
        for gp in range(4):
            entry = 0.0 + 0.0j
            for bx, by, f, d in _BONDS:
                b = jnp.array([bx, by])
                entry = entry + jh[f, g ^ gp, d] * (
                    jnp.exp(1j * jnp.dot(k + G_VECTORS[gp], b)) + jnp.exp(-1j * jnp.dot(k + G_VECTORS[g], b))
                )
            row.append(entry)
        rows.append(jnp.stack(row))

    return jnp.stack(rows)


def k_grid(Lx, Ly):
    """Momenta kx, ky in {2 pi n / L} cap [0, pi).

    The quartet k + G already covers the other three quadrants, so this grid
    times 4 momenta is the full Lx * Ly Brillouin zone, counted exactly once.
    Even Lx, Ly are required for k + (pi, 0) to be an allowed momentum (and for
    the 2x2 coupling pattern to be consistent with periodic boundaries).
    """
    if Lx % 2 or Ly % 2:
        raise ValueError(f"Lx = {Lx}, Ly = {Ly}: the 2x2 coupling pattern needs even sizes.")
    nx = jnp.arange(Lx // 2)
    ny = jnp.arange(Ly // 2)
    NX, NY = jnp.meshgrid(nx, ny, indexing="ij")
    KX = (2 * jnp.pi * NX / Lx).reshape(-1)
    KY = (2 * jnp.pi * NY / Ly).reshape(-1)
    return KX, KY


def _luttinger_tisza_jax(Lx, Ly, J1, J2, tol=1e-9):
    """Minimize the lowest band over the quartet-folded BZ.

    Returns (k_opt, c_opt, Q_full, E_min, is_plain, band_gap).
    Lx, Ly are static Python ints, J1 / J2 may be traced.
    """
    KX, KY = k_grid(Lx, Ly)
    mats = jax.vmap(lambda kx, ky: exchange_matrix(kx, ky, J1, J2))(KX, KY)
    evals, evecs = jnp.linalg.eigh(mats)  # ascending eigenvalues
    E_min = jnp.min(evals[:, 0])

    BIG = 1e6

    # ---- plain-spiral branch -------------------------------------------
    # diag(M) IS the old scalar dispersion at k + G, so scanning it costs
    # nothing extra.  If it reaches E_min the minimizer is a plain spiral:
    # c is a unit vector, the spin length is automatically constant, and the
    # full-BZ momentum is k + G.  This also sidesteps the arbitrary mixture
    # eigh returns at points where the lowest band is degenerate.
    diag = jnp.real(jnp.diagonal(mats, axis1=1, axis2=2))  # (nk, 4)
    plain_mask = diag < E_min + tol  # >= E_min always
    is_plain = jnp.any(plain_mask)

    QFX = KX[:, None] + G_VECTORS[None, :, 0]
    QFY = KY[:, None] + G_VECTORS[None, :, 1]
    flat = jnp.argmin(jnp.where(plain_mask, QFX * BIG + QFY, jnp.inf).reshape(-1))
    ik_p, ig_p = flat // 4, flat % 4
    k_plain = jnp.stack([KX[ik_p], KY[ik_p]])
    c_plain = jax.nn.one_hot(ig_p, 4).astype(mats.dtype)

    # ---- general 4-momentum branch -------------------------------------
    mask = jnp.abs(evals[:, 0] - E_min) < tol
    idx = jnp.argmin(jnp.where(mask, KX * BIG + KY, jnp.inf))
    k_gen = jnp.stack([KX[idx], KY[idx]])
    c_gen = evecs[idx, :, 0]
    g_ref = jnp.argmax(jnp.abs(c_gen))  # gauge fix
    c_gen = c_gen * jnp.conj(c_gen[g_ref]) / jnp.abs(c_gen[g_ref])

    k_opt = jnp.where(is_plain, k_plain, k_gen)
    c_opt = jnp.where(is_plain, c_plain, c_gen)

    # dominant momentum of the quartet = the old full-BZ ordering vector
    Q_full = jnp.mod(k_opt + G_VECTORS[jnp.argmax(jnp.abs(c_opt))], 2 * jnp.pi)
    band_gap = evals[idx, 1] - evals[idx, 0]

    return k_opt, c_opt, Q_full, E_min, is_plain, band_gap


def spin_field(k, c, positions):
    """z_r = sum_G c_G exp(i (k + G) . r), one complex number per site.

    theta_r = angle(z_r) and |z_r| is the spin length: |z_r| must be constant
    for the state to be a legal classical configuration.  This is the
    Luttinger-Tisza constraint in its per-site form.
    """
    kq = jnp.asarray(k)[None, :] + G_VECTORS  # (4, 2)
    return jnp.exp(1j * (jnp.asarray(positions) @ kq.T)) @ c


def site_phases(L, J1, couplings, positions):
    """theta(r) = angle(sum_G c_G exp(i (k+G) . r))  mod 2 pi, one per site.

    No sublattice bookkeeping: the positions enter the transform directly.
    """
    J2 = as_couplings(couplings, "J2")

    k_opt, c_opt, _, _, _, _ = _luttinger_tisza_jax(L, L, J1, J2)
    z = spin_field(k_opt, c_opt, positions)

    return jnp.mod(jnp.angle(z), 2 * jnp.pi)


def compute_phases_8coups(L, J1, couplings, positions):
    """Same packing as before: the couplings, then one phase per site.

    `couplings` is the length-8 diagonal pattern (a scalar or a length-2/4
    array still works and reproduces the more symmetric models).
    Output length is 8 + len(positions).
    """
    J2 = as_couplings(couplings, "J2")

    phases = jnp.concatenate([J2.reshape(-1), site_phases(L, J1, J2, positions)])
    return phases
