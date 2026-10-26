import numpy as np
from netket.operator.spin import sigmax, sigmay, sigmaz


def spin_corr(hi, L, translations=False, cell=(2, 2)):
    """
    Spin-spin correlation operators  S_a . S_r  for the static structure factor,
    for a state invariant under translations by a `cell` supercell.

    Site ordering follows the Hamiltonian:

        site = i * L + j        i -> +x (stride L),  j -> +y (stride 1)
        c    = (i % ci) * cj + (j % cj)     == 2 * (i % 2) + (j % 2) for 2x2

    The reference site 0 is replaced by the n_b = ci*cj sites of the home
    supercell, taken in that same flat order (a = 0,1,2,3 -> sites 0, 1, L, L+1).

    Parameters
    ----------
    hi : nk.hilbert.Spin                  # s=1/2, N = L*L sites
    L  : int                              # linear size of the square lattice
    translations : bool
                   If True, use the N/n_b *supercell* translations (shifts by
                   whole cells, the only ones that preserve the sub-cell label).
                   If False, use only the identity translation.
    cell : (ci, cj)                       # supercell periods along +x, +y

    Returns
    -------
    list of operators, length n_b * N * n_t, flat index

        idx = (a * N + r) * n_t + t   ->   S_{t[i_a]} . S_{t[r]}

    so the expectation values reshape as (n_b, N, n_t) and, after averaging
    over t, give C[a, r] = C_{ab}(D) with

        Q^2 = 1/(n_b * N) * sum_a sum_r C[a, r]**2
    """
    ci, cj = cell
    N = L * L
    ci * cj
    assert hi.size == N
    assert L % ci == 0 and L % cj == 0, "supercell must tile the lattice"

    # S = sigma/2   =>   S_i . S_j = (1/4) sum_a sigma^a_i sigma^a_j
    def SdotS(i, j):
        i, j = int(i), int(j)
        return 0.25 * (sigmax(hi, i) @ sigmax(hi, j) + sigmay(hi, i) @ sigmay(hi, j) + sigmaz(hi, i) @ sigmaz(hi, j))

    if translations:

        def _translate(di, dj):
            sites = np.arange(N).reshape(L, L)
            sites = np.roll(sites, shift=(di, dj), axis=(0, 1))
            return sites.reshape(N)

        translations = [_translate(di, dj) for di in range(0, L, ci) for dj in range(0, L, cj)]
    else:
        translations = [np.arange(N)]

    refs = [ai * L + aj for ai in range(ci) for aj in range(cj)]

    ops = []
    for i_ref in refs:
        for r in range(N):
            for t in translations:
                ops.append(SdotS(t[i_ref], t[r]))

    return ops
