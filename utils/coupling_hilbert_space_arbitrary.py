import jax
import jax.numpy as jnp
import numpy as np
from netket.hilbert.discrete_hilbert import DiscreteHilbert
from netket.utils.dispatch import dispatch


class CouplingHilbert(DiscreteHilbert):
    r"""Hilbert space with an explicit list of allowed vector-valued states."""

    def __init__(self, values):
        values = np.asarray(values)

        if values.ndim != 2:
            raise ValueError("`values` must have shape (n_states, vec_dim).")

        self._values_np = values.copy()
        self._values = jnp.asarray(values)
        self._n_states, self._vec_dim = values.shape

        # Hashable representation
        self._values_tuple = tuple(tuple(row.tolist()) for row in values)

        # Map state -> basis index
        self._state_to_number = {tuple(row.tolist()): i for i, row in enumerate(values)}

        # `shape` is only used by parts of NetKet expecting a discrete hilbert.
        # Here `size = vec_dim`, i.e. each state is a vector of length 2.
        super().__init__(shape=(self._n_states,) * self._vec_dim)

    @property
    def size(self) -> int:
        return self._vec_dim

    @property
    def n_states(self) -> int:
        return self._n_states

    @property
    def is_finite(self) -> bool:
        return True

    @property
    def constrained(self) -> bool:
        return True

    @property
    def is_indexable(self) -> bool:
        return True

    @property
    def _attrs(self):
        return (self._values_tuple,)

    def states_at_index(self, i: int):
        return np.unique(self._values_np[:, i]).tolist()

    def size_at_index(self, i: int) -> int:
        return len(self.states_at_index(i))

    def _numbers_to_states(self, numbers: jax.Array) -> jax.Array:
        return self._values[numbers]

    def _states_to_numbers(self, states: jax.Array) -> jax.Array:
        states = np.asarray(states)
        out = np.empty(states.shape[0], dtype=np.int32)

        for i, state in enumerate(states):
            key = tuple(np.asarray(state).tolist())
            if key not in self._state_to_number:
                raise ValueError(f"State {state} is not in the allowed basis.")
            out[i] = self._state_to_number[key]

        return jnp.asarray(out)

    def all_states(self):
        return self._values

    def __repr__(self):
        return f"CouplingHilbert(n_states={self._n_states}, vec_dim={self._vec_dim})"


@dispatch
def random_state(hilb: CouplingHilbert, key, size: int, *, dtype):
    idx = jax.random.randint(key, shape=(size,), minval=0, maxval=hilb.n_states)
    return hilb._values[idx].astype(dtype)
