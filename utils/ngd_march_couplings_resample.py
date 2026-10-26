"""One natural-gradient step on the couplings, with March momentum. The loss is the Edwards-Anderson order parameter.
"""

import time
from functools import partial

import jax
import jax.numpy as jnp
from netket.jax.sharding import shard_along_axis

from utils.compute_eloc import compute_eloc
from utils.magnetization_operator import get_ea, get_grad_ea


@partial(
    jax.jit,
    static_argnames=("afun",),
)
def _logpsi_fn(x, afun, variables, coups_single):
    coups_single = jnp.expand_dims(coups_single, axis=0)  # (n_systems, n_couplings)
    coups = jnp.repeat(coups_single, repeats=x.shape[0], axis=0)
    coups_x = jnp.concatenate((coups, x), axis=-1)
    return afun(variables, coups_x)


def centering(local_estimator):
    expval = jnp.mean(local_estimator, axis=-1)
    local_estimator = (local_estimator - expval) / local_estimator.shape[0]
    return local_estimator, expval


@partial(
    jax.jit,
    static_argnames=(
        "afun",
        "n_chunks",
    ),
)
def compute_qgt(x, afun, variables, coup, n_chunks):
    n_samples = x.shape[0]
    n_couplings = coup.shape[-1]

    assert n_samples % n_chunks == 0, "n_chunks must be changed"
    chunk_size = n_samples // n_chunks

    def jacobian(x_batch):
        def logpsi_fn(coups):
            return _logpsi_fn(x_batch, afun, variables, coups)

        return jax.jacfwd(logpsi_fn)(coup)  # (chunk_size, n_couplings)

    dtype = jax.eval_shape(jacobian, x[:chunk_size]).dtype

    S_sum = jnp.zeros((n_couplings, n_couplings), dtype=dtype)
    O_sum = jnp.zeros((n_couplings,), dtype=dtype)

    def accumulate_chunk(idx, carry):
        S_sum, O_sum = carry

        start = idx * chunk_size

        x_batch = jax.lax.dynamic_slice(
            x,
            (start, 0),
            (chunk_size, x.shape[-1]),
        )

        x_batch = shard_along_axis(x_batch, axis=0)

        O = jacobian(x_batch)

        S_sum = S_sum + jnp.einsum("si,sj->ij", O.conj(), O)
        O_sum = O_sum + jnp.sum(O, axis=0)

        return S_sum, O_sum

    S_sum, O_sum = jax.lax.fori_loop(0, n_chunks, accumulate_chunk, (S_sum, O_sum))

    O_expval = O_sum / n_samples

    qgt = S_sum / n_samples - jnp.outer(O_expval.conj(), O_expval)

    return qgt.real  # Psi is real


@jax.jit
def solve_qgt(qgt, grad, diag_shift):
    qgt = qgt + diag_shift * jnp.eye(qgt.shape[-1], dtype=qgt.dtype)

    nat_grad = jax.scipy.linalg.solve(qgt, grad)

    return nat_grad


def ngd_grad(
    vstate,
    afun,
    variables,
    coup,
    n_reps,
    corr_list,
    n_couplings,
    n_chunks,
):
    # Store losses, grads and quantum geometric tensors
    losses = []
    grads = []
    qgts = []

    # Take the samples
    for rep in range(n_reps):
        coups_x = vstate.sample()
        coups_x = coups_x.reshape(-1, coups_x.shape[-1])

        # Unpack the coupling
        x = coups_x[:, n_couplings:]

        # Logpsi to differentiate
        def logpsi_fn(coups):
            return _logpsi_fn(x, afun, variables, coups)

        # Derivative
        _, vjp_fun = jax.vjp(logpsi_fn, coup)

        grad_terms = []
        loss_terms = []

        for corr in corr_list:
            corr_loc = compute_eloc(corr, coups_x, n_couplings, afun, variables, n_chunks)

            corr_loc, corr_expval = centering(corr_loc)  # Centering the local estimator

            corr_grad = 2 * jnp.real(vjp_fun(corr_loc)[0])  # Psi is real

            loss_terms.append(corr_expval.real)
            grad_terms.append(corr_grad)

        loss_terms = jnp.array(loss_terms)
        grad_terms = jnp.array(grad_terms)

        loss = get_ea(loss_terms).real

        grad = get_grad_ea(loss_terms, grad_terms).real

        # Quantum geometric tensor on the same samples
        qgt = compute_qgt(x, afun, variables, coup, n_chunks)

        losses.append(loss)
        grads.append(grad)
        qgts.append(qgt)

    losses = jnp.array(losses)
    grads = jnp.array(grads)
    qgts = jnp.array(qgts)

    loss = jnp.mean(losses, axis=0)
    grad = jnp.mean(grads, axis=0)
    error = jnp.std(losses, axis=0) / jnp.sqrt(n_reps)
    qgt = jnp.mean(qgts, axis=0)

    return loss, grad, qgt, error


def init_march_state(coup):
    phi = jnp.zeros_like(coup)
    velocity = jnp.ones_like(coup)
    previous_update = jnp.zeros_like(coup)
    t = 1

    return phi, velocity, previous_update, t


@jax.jit
def march_update(
    qgt,
    grad,
    march_state,
    has_march_state,
    diag_shift,
    mu,
    beta,
    march_eps,
):
    phi, velocity, previous_update, t = march_state

    # Bias corrected velocity
    velocity_hat = velocity / (1 - beta**t)
    velocity_inv_quarter = 1.0 / (jnp.sqrt(jnp.sqrt(velocity_hat)) + march_eps)

    # Residual of the linear system, warm started at phi
    residual = jnp.where(has_march_state, grad - qgt @ phi, grad)

    # Quantum geometric tensor in the rescaled coordinates: A S A
    qgt_march = velocity_inv_quarter[:, None] * qgt * velocity_inv_quarter[None, :]

    # Preconditioning of the gradient with the (rescaled) quantum geometric tensor
    nat_grad = solve_qgt(qgt_march, velocity_inv_quarter * residual, diag_shift)

    update = phi + velocity_inv_quarter * nat_grad

    # March state update
    phi_new = mu * update
    diff = update - previous_update
    velocity_new = beta * velocity + (1 - beta) * jnp.abs(diff) ** 2

    march_state = (phi_new, velocity_new, update, t + 1)

    return update, march_state


def ngd_march_onestep(
    vstate,
    afun,
    variables,
    coup,
    n_reps,
    corr_list,
    n_couplings,
    lr,
    best_loss,
    best_coup,
    ema_loss,
    ema_beta,
    n_chunks,
    march_state=None,
    mu=0.95,
    beta=0.995,
    march_eps=1e-8,
    diag_shift=1e-4,
):
    t_start = time.perf_counter()
    loss, grad, qgt, error = ngd_grad(
        vstate,
        afun,
        variables,
        coup,
        n_reps,
        corr_list,
        n_couplings,
        n_chunks,
    )
    loss, grad, qgt, error = jax.block_until_ready((loss, grad, qgt, error))

    if jax.process_index() == 0:
        print(f"\n Gradient time: {time.perf_counter() - t_start:.3f} s \n", flush=True)

    if loss < best_loss:
        best_loss = loss
        best_coup = coup

    ema_loss = loss if ema_loss is None else ema_beta * ema_loss + (1 - ema_beta) * loss

    # March state
    has_march_state = march_state is not None
    if march_state is None:
        march_state = init_march_state(coup)

    # NGD update with March
    update, march_state = march_update(
        qgt,
        grad,
        march_state,
        has_march_state,
        diag_shift,
        mu,
        beta,
        march_eps,
    )

    coup = jnp.maximum(coup - lr * update, 0.0)

    if jax.process_index() == 0:
        print(f"\n Coupling: {coup}, Loss: {loss}, EMA: {ema_loss}, lr: {lr} \n", flush=True)

    return coup, loss, error, best_loss, best_coup, ema_loss, march_state
