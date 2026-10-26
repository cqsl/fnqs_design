"""J1-J2 Heisenberg Hamiltonian on the square lattice with 8 J2 couplings on a 2x2 unit cell.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from netket.jax.sharding import shard_along_axis
from netket.operator.spin import sigmax, sigmay, sigmaz

from utils.afun_chunk import afun_chunk

# ---------------------------------------------------------------------------
# 2x2 unit cell: 4 sub-cells x 2 bond directions = 8 couplings per bond family
#
#   sub-cell index   c = 2 * (i % 2) + (j % 2)
#
#              j%2 = 0   j%2 = 1
#     i%2 = 0    c = 0     c = 1
#     i%2 = 1    c = 2     c = 3
#
#   direction index  d = 0 -> bond (i, j) -- (i+1, j)   [+x]
#                    d = 1 -> bond (i, j) -- (i, j+1)   [+y]
#                    (for J2: d = 0 -> +x+y, d = 1 -> +x-y)
#
#   flat coupling index = 2 * c + d, i.e. the length-8 vector is
#   [ J_{c0,x}, J_{c0,y}, J_{c1,x}, J_{c1,y},
#     J_{c2,x}, J_{c2,y}, J_{c3,x}, J_{c3,y} ]
#
#   Each bond is assigned the coupling of the sub-cell of its "origin" site
#   (i, j), so every bond appears exactly once.
#
# Foundation version: the J1 pattern is FIXED (baked into a single operator),
# while the 8 J2 couplings are FREE parameters carried by the samples. The
# samples are laid out as
#
#   coups_x = [ J2_0, J2_1, ..., J2_7, x_0, ..., x_{N-1} ]   (n_couplings = 8)
#
# with the J2 entries in the same flat 2 * c + d order as above.
# ---------------------------------------------------------------------------


def cell_index(i, j):
    """Sub-cell index c in {0,1,2,3} of site (i, j) within the 2x2 unit cell."""
    return 2 * (i % 2) + (j % 2)


def as_couplings(J, name="J"):
    """Broadcast a coupling specification to a (4, 2) array.

    Accepted shapes:
      scalar   -> uniform over all sub-cells and both directions
      (2,)     -> (d=0, d=1), uniform over the 4 sub-cells
      (4,)     -> per sub-cell, isotropic in direction
      (8,)     -> full pattern, flat index 2 * c + d
      (4, 2)   -> full pattern, explicit
    """
    J = np.asarray(J, dtype=float)

    if J.ndim == 0:
        return np.full((4, 2), float(J))
    if J.shape == (4, 2):
        return J
    flat = J.reshape(-1)
    if flat.size == 2:
        return np.tile(flat[None, :], (4, 1))
    if flat.size == 4:
        return np.tile(flat[:, None], (1, 2))
    if flat.size == 8:
        return flat.reshape(4, 2)

    raise ValueError(f"{name} must be a scalar or have 2, 4, 8 or (4, 2) entries, got shape {J.shape}")


def _heisenberg_bond(hi, a, b, sign=1.0):
    """(1/4) * (sign * XX + sign * YY + ZZ) on sites a, b."""
    return 0.25 * (
        sign * sigmax(hi, a) @ sigmax(hi, b) + sign * sigmay(hi, a) @ sigmay(hi, b) + sigmaz(hi, a) @ sigmaz(hi, b)
    )


def _lattice_size(N):
    L = int(round(N**0.5))
    if L * L != N:
        raise ValueError(f"N = {N} is not a perfect square")
    if L % 2 != 0:
        raise ValueError(
            f"L = {L} is odd: a 2x2 unit cell is not compatible with periodic "
            "boundary conditions. Use an even L (L >= 4 to avoid double bonds)."
        )
    return L


def create_J1_term(N, hi, J1, marshall=False):
    """Nearest-neighbour term, with the fixed 8-coupling 2x2 pattern baked in.

    The J1 values are constants of the foundation model, so they multiply the
    matrix elements once and for all here (no runtime rescaling needed).
    """
    L = _lattice_size(N)
    Jc = as_couplings(J1, "J1")
    sign = -1.0 if marshall else 1.0

    ham = 0.0 + 1j * 0.0

    for i in range(L):
        for j in range(L):
            site = i * L + j
            c = cell_index(i, j)

            neighbors = (
                ((i + 1) % L) * L + j,  # d = 0, +x
                i * L + (j + 1) % L,  # d = 1, +y
            )

            for d, site_d in enumerate(neighbors):
                ham += Jc[c, d] * _heisenberg_bond(hi, site, site_d, sign)

    ham = ham.to_jax_operator()

    return ham


def create_J2_term_single(N, hi, c, d):
    """Next-nearest-neighbour term restricted to the single coupling (c, d).

    Only the diagonal bonds whose origin site belongs to sub-cell c and whose
    direction is d (0 -> +x+y, 1 -> +x-y) are included, with unit coefficient:
    the coupling itself is applied at runtime, as it is a free parameter.
    """
    L = _lattice_size(N)

    ham = 0.0 + 1j * 0.0

    for i in range(L):
        for j in range(L):
            if cell_index(i, j) != c:
                continue

            site = i * L + j

            if d == 0:
                site_d = ((i + 1) % L) * L + (j + 1) % L  # +x+y
            else:
                site_d = ((i + 1) % L) * L + (j - 1) % L  # +x-y

            ham += _heisenberg_bond(hi, site, site_d)

    ham = ham.to_jax_operator()

    return ham


# Foundation Hamiltonian for Heisenberg J1-J2 with a 2x2 magnetic unit cell,
# with the 8 J2 couplings optimized separately
class FoundationHeisenbergJ1J2:
    n_couplings = 8

    def __init__(self, hi, N, n_systems, J1=1.0, marshall=False):
        self.n_systems = n_systems
        self.J1 = as_couplings(J1, "J1")

        # J1 (nearest-neighbour) term: fixed couplings, already in the mels
        self.operator_J1 = create_J1_term(N, hi, J1, marshall=marshall)

        # J2 (next-nearest-neighbour) terms: free parameters, one operator per
        # (sub-cell, diagonal), ordered by the flat index 2 * c + d
        self.operators_J2 = tuple(create_J2_term_single(N, hi, c, d) for c in range(4) for d in range(2))

        # NetKet JAX operators lazily build internal connectivity tables on the
        # first get_conn_padded() call. Doing that inside a jitted function
        # leaks traced values through object mutation, so force setup eagerly.
        self.operator_J1._setup()
        for op in self.operators_J2:
            op._setup()

    @partial(
        jax.jit,
        static_argnames=(
            "self",
            "afun",
            "n_chunks",
        ),
    )
    def compute_connected_elements(self, afun, variables, coups_x, n_chunks):
        n_c = self.n_couplings

        coups_x = coups_x.reshape(-1, coups_x.shape[-1])  # (N_mc, n_couplings + N)

        J2 = coups_x[..., :n_c]  # (N_mc, 8)  free parameters
        x = coups_x[..., n_c:]  # (N_mc, N)  spin configuration

        xp_J1, mels_J1 = self.operator_J1.get_conn_padded(x)

        conn_J2 = [op.get_conn_padded(x) for op in self.operators_J2]
        xp_J2 = tuple(c[0] for c in conn_J2)
        mels_J2 = tuple(c[1] for c in conn_J2)

        # keep this concatenation order consistent with compute_mels
        xp = jnp.concatenate((xp_J1, *xp_J2), axis=-2)

        J2_xp = J2[:, None, :]  # (N_mc, 1, 8)
        J2_xp = jnp.tile(J2_xp, (1, xp.shape[-2], 1))  # (N_mc, n_conn, 8)
        coups_xp = jnp.concatenate((J2_xp, xp), axis=-1)  # (N_mc, n_conn, N+8)

        coups_x = shard_along_axis(coups_x, axis=0)

        logpsi_x = afun(variables, coups_x)

        dtype = logpsi_x.dtype

        shape_conn = coups_xp.shape[:-1]
        coups_xp = coups_xp.reshape(-1, coups_xp.shape[-1])
        assert coups_xp.shape[0] % n_chunks == 0, "n_chunks must be changed"
        logpsi_xp = afun_chunk(afun, variables, coups_xp, n_chunks, dtype)
        logpsi_xp = logpsi_xp.reshape(shape_conn)

        return logpsi_x, logpsi_xp, mels_J1, mels_J2

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def compute_mels(self, coups_x, mels_J1, mels_J2, logpsi_x, logpsi_xp):
        J2 = coups_x[..., : self.n_couplings]  # (N_mc, 8)

        # the J1 couplings are already contained in mels_J1
        mels_J2 = tuple(mels * J2[..., k : k + 1] for k, mels in enumerate(mels_J2))

        # same order as the xp concatenation above
        mels = jnp.concatenate((mels_J1, *mels_J2), axis=-1)

        eloc = jnp.sum(mels * jnp.exp(logpsi_xp - jnp.expand_dims(logpsi_x, axis=-1)), axis=-1)

        return eloc

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def reshaping(self, coups_x, logpsi_x, logpsi_xp, mels_J1, mels_J2):
        coups_x = coups_x.reshape(self.n_systems, -1, coups_x.shape[-1])
        logpsi_x = logpsi_x.reshape(self.n_systems, -1)
        logpsi_xp = logpsi_xp.reshape(self.n_systems, -1, logpsi_xp.shape[-1])
        mels_J1 = mels_J1.reshape(self.n_systems, -1, mels_J1.shape[-1])
        mels_J2 = tuple(mels.reshape(self.n_systems, -1, mels.shape[-1]) for mels in mels_J2)

        return coups_x, logpsi_x, logpsi_xp, mels_J1, mels_J2

    def get_eloc(self, vstate, n_chunks=1):
        coups_x = vstate.samples
        coups_x = coups_x.reshape(-1, coups_x.shape[-1])  # (N_mc, n_couplings + N)

        logpsi_x, logpsi_xp, mels_J1, mels_J2 = self.compute_connected_elements(
            vstate._apply_fun, vstate.variables, coups_x, n_chunks
        )

        coups_x, logpsi_x, logpsi_xp, mels_J1, mels_J2 = self.reshaping(coups_x, logpsi_x, logpsi_xp, mels_J1, mels_J2)

        elocs = jax.vmap(self.compute_mels, in_axes=(0, 0, 0, 0, 0))(coups_x, mels_J1, mels_J2, logpsi_x, logpsi_xp)

        return elocs

    def get_eloc_onesystem(self, vstate, coups_x, n_chunks=1):
        logpsi_x, logpsi_xp, mels_J1, mels_J2 = self.compute_connected_elements(
            vstate._apply_fun, vstate.variables, coups_x, n_chunks
        )

        elocs = self.compute_mels(coups_x, mels_J1, mels_J2, logpsi_x, logpsi_xp)

        return elocs
