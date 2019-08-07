"""Benchmarking against cirq's various unitary multiplication fallbacks."""
import pytest

import numpy as np
import tensorflow as tf

import cirq
from cirq.contrib.tpu import (
    circuit_to_tensorflow_runnable
)

np.random.seed(31415926)


TRIAL_RUNS = [
    (4, 10),
    (5, 10),
    (6 ,10)
]



def _cirq_to_cirq_execute(c):
    """Execute ops in cirq; output will be baseline benchmark truth."""

    return cirq.Simulator().simulate(c)


def _cirq_to_tpu_execute(c):
    """Compile ops from cirq into XLA-compatibile ops via cirq.contrib.tpu.

    Note: Timing on the compilation step is being benchmarked because the
    cirq tpu add-on relies heavily on linalg operations.
    """
    r = circuit_to_tensorflow_runnable(c)
    # todo: simpler operation than returning full wf? like:
    #expectation = lambda: tf.norm(r.compute()[:128], 2)

    with tf.Session() as session:
        output = session.run(r.compute(), feed_dict=r.feed_dict)
    return output


def _cirq_to_eigen_backend_execute(c):
    """TODO."""


def _cirq_to_tf_cirq_backend_execute(r, feed_dict):
    """Run a tf graph that has been compiled via cirq-tf.

    Args:
        r: Compiled tensorflow graph
        feed_dict: Feeder of the form {placeholder: value}

    Note: Timing for compilation step is _not_ included, since the compilation
    is purely graph construction.
    """
    with tf.Session() as session:
        output = session.run(r, feed_dict=r.feed_dict)
    return output
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# TYPE-ZERO TESTS: ALL CIRCUIT OPERATIONS HAVE A SPECIALIZED UNITARY OPERATION
# THAT ALLOWS EFFICIENT UNITARY ACTION DURING CIRCUIT SIMULATION

OPS_LIST_0 = [
    cirq.X,
    cirq.Y,
    cirq.Z,
    cirq.I,
    cirq.H,
    cirq.CZ,
    cirq.CNOT,
    cirq.SWAP,
    cirq.ISWAP,
]


def _generator_type_zero(n_qubits, depth):
    """Construct a (possibly dense) circuit from OPS_LIST_0."""
    qubits = cirq.LineQubit.range(n_qubits)
    ops = []
    for layer in range(depth):

        m = int(np.random.randint(2, n_qubits+1))
        # an exclusive list of m qubits with index in (0, n_qubits)
        qubits_this_layer = np.random.choice(np.arange(n_qubits), size=m, replace=False)
        qubits_this_layer = [qubits[i] for i in qubits_this_layer]

        # a set of initialized gates to apply this layer
        gates_this_layer = np.random.randint(0, high=len(OPS_LIST_0), size=m)
        initialized = []

        for i, j in zip(gates_this_layer, range(m)):
            q0 = qubits_this_layer[j]
            q1 = qubits_this_layer[(j+1)%m]
            # two-qubit on `nearest neighbor`
            try:
                initialized.append(OPS_LIST_0[i](q0, q1 ))
            except:
                initialized.append(OPS_LIST_0[i](q0))

        ops += initialized
    return cirq.Circuit.from_ops(ops)


def _cirq_to_cirq_type_zero(n_qubits, depth):
    trial_ops = _generator_type_zero(n_qubits, depth)
    return _cirq_to_cirq_execute(trial_ops)


@pytest.mark.parametrize('n_qubits,depth', TRIAL_RUNS)
def test_specialized_unitary_fallback(benchmark, n_qubits, depth):
    """
    Perform a circuit simulation that uses purely `specialized` operations
    for computing the output state. This is the first type of operation that
    is tried by cirq.protocols.apply_unitary. It typically consists of applying
    a permutation and eigenvalue-based negation on a specific qubit subspace.
    Relevant ops:
        cirq.X (cirq.XPowGate(1))
        cirq.Y (cirq.YPowGate(1))
        cirq.Z (cirq.ZPowGate(1))
        cirq.I
        cirq.H
        cirq.CZ (cirq.CZPowGate(1))
        cirq.CNOT
        cirq.SWAP
        cirq.ISWAP
    (For this gateset minus H, it is obvious that matrix permutation is more
    efficient than einsum)
    """
    result = benchmark(_cirq_to_cirq_type_zero, n_qubits, depth)


def _cirq_to_tpu_type_zero(n_qubits, depth):
    trial_ops = _generator_type_zero(n_qubits, depth)
    return _cirq_to_tpu_execute(trial_ops)


@pytest.mark.parametrize('n_qubits,depth', TRIAL_RUNS)
def test_tpu_specialized_unitary_fallback(benchmark, n_qubits, depth):
    """See above."""
    result = benchmark(_cirq_to_tpu_type_zero, n_qubits, depth)



def test_cirq_to_tf_specialized_unitary_fallback(benchmark, n_qubits, depth):
    """See above."""
    trial_ops = _generator_type_zero(n_qubits, depth)
    return _cirq_to_tf_cirq_backend_execute(trial_ops)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# TYPE-ONE TESTS: ALL CIRCUIT OPERATIONS HAVE A SUBSPACE MULTIPLICATION METHOD


OPS_LIST_1 = [
    cirq.X,
    cirq.Y,
    cirq.Z,
    cirq.H,
]


def _generator_type_one(n_qubits, depth):
    """Construct a (possibly dense) circuit from OPS_LIST_1."""
    qubits = cirq.LineQubit.range(n_qubits)
    ops = []
    for layer in range(depth):

        m = int(np.random.randint(2, n_qubits+1))
        # an exclusive list of m qubits with index in (0, n_qubits)
        qubits_this_layer = np.random.choice(np.arange(n_qubits), size=m, replace=False)
        qubits_this_layer = [qubits[i] for i in qubits_this_layer]

        # a set of exponents not equal to one; doesn't matter what they are
        # for efficient computation
        exponents_this_layer = np.abs(np.random.randn(m))
        for k, v in enumerate(exponents_this_layer):
            if np.isclose(v, 1):
                exponents_this_layer[k] = v + 0.01
        # a set of initialized gates to apply this layer
        gates_this_layer = np.random.randint(0, high=len(OPS_LIST_1), size=m)
        gates_this_layer = [
            OPS_LIST_1[i](j)**k for i,j,k in zip(gates_this_layer, qubits_this_layer, exponents_this_layer)
        ]

        ops += gates_this_layer
    return cirq.Circuit.from_ops(ops)


def _cirq_to_cirq_type_one(n_qubits, depth):
    trial_ops = _generator_type_one(n_qubits, depth)
    return _cirq_to_cirq_execute(trial_ops)


@pytest.mark.parametrize('n_qubits,depth', TRIAL_RUNS)
def test_cirq_single_qubit_fallback(benchmark, n_qubits, depth):
    """
    Perform a circuit simulation that uses purely single-qubit gates with
    efficient multiplication via cirq.linalg.apply_matrix_to_slices but do not
    define `_apply_unitary_`.
    Relevant ops:
        cirq.XPowGate(!1)
        cirq.YPowGate(!1)
        cirq.ZPowGate(!1)
        cirq.H ** (!1)

    """
    result = benchmark(_cirq_to_cirq_type_one, n_qubits, depth)


def _cirq_to_tpu_type_one(n_qubits, depth):
    trial_ops = _generator_type_one(n_qubits, depth)
    return _cirq_to_tpu_execute(trial_ops)


@pytest.mark.parametrize('n_qubits,depth', TRIAL_RUNS)
def test_tpu_single_qubit_fallback(benchmark, n_qubits, depth):
    """See above."""
    result = benchmark(_cirq_to_tpu_type_one, n_qubits, depth)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# TYPE-TWO TESTS: CIRCUIT IS SIMULATED VIA EINSUM.

# reminder: these gates will be exponentiated
OPS_LIST_2 = [
    cirq.CZ,
    cirq.CNOT,
    cirq.SWAP,
    cirq.ISWAP,
]


def _generator_type_two(n_qubits, depth):
    """Construct a (possibly dense) circuit from OPS_LIST_1."""
    qubits = cirq.LineQubit.range(n_qubits)
    ops = []
    for layer in range(depth):

        m = int(np.random.randint(2, n_qubits+1))
        # an exclusive list of m qubits with index in (0, n_qubits)
        # two-qubit gate may reach beyond this list.
        qubits_this_layer = np.random.choice(np.arange(n_qubits), size=m, replace=False)
        qubits_this_layer = [qubits[i] for i in qubits_this_layer]

        # a set of exponents not equal to one; doesn't matter what they are
        # for efficient computation
        exponents_this_layer = np.abs(np.random.randn(m))
        for k, v in enumerate(exponents_this_layer):
            if np.isclose(v, 1):
                exponents_this_layer[k] = v + 0.01
        # a set of initialized gates to apply this layer
        gates_this_layer = np.random.randint(0, high=len(OPS_LIST_2), size=m)
        gates_this_layer = [
            OPS_LIST_2[i](qubits_this_layer[j], qubits_this_layer[(j+1)%m])**k for i,j,k in zip(gates_this_layer, range(m), exponents_this_layer)
        ]

        ops += gates_this_layer
    return cirq.Circuit.from_ops(ops)


def _cirq_to_cirq_type_two(n_qubits, depth):
    trial_ops = _generator_type_two(n_qubits, depth)
    return _cirq_to_cirq_execute(trial_ops)


@pytest.mark.parametrize('n_qubits,depth', TRIAL_RUNS)
def test_cirq_einsum_fallback(benchmark, n_qubits, depth):
    """
    Perform a circuit siulation that uses purely two-qubit gates subject to
    cirq.linalg.targeted_left_multiply (wrapper for np.einsum) but do NOT
    define `_apply_unitary_`
    relevant ops:
        cirq.CZPowGate(!1)
        cirq.CNOT ** (!1)
        cirq.SWAP ** (!1)
        cirq.ISWAP ** (!1)
    """
    result = benchmark(_cirq_to_cirq_type_two, n_qubits, depth)


def _cirq_to_tpu_type_two(n_qubits, depth):
    trial_ops = _generator_type_two(n_qubits, depth)
    return _cirq_to_tpu_execute(trial_ops)


@pytest.mark.parametrize('n_qubits,depth', TRIAL_RUNS)
def test_tpu_einsum_fallback(benchmark, n_qubits, depth):
    """See above."""
    result = benchmark(_cirq_to_tpu_type_two, n_qubits, depth)
