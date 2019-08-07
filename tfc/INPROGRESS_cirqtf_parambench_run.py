"""Benchmark resolution of parameters like in an outer-loop optimizer."""
import random

from timeit import default_timer as timer
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import sympy
import cirq
from tf_simulator import (
    TFWaveFunctionSimulator
)

def get_parametrized_two_qubit_gates():
    return [
        cirq.SwapPowGate,
        cirq.CNotPowGate,
        cirq.ISwapPowGate,
        cirq.ZZPowGate,
    ]


def get_parametrized_single_qubit_gates():
    return [
        cirq.Rx,
        cirq.Ry,
        cirq.Rz,
    ]


ALL_GATES = get_parametrized_single_qubit_gates() + \
            get_parametrized_two_qubit_gates()


def random_single_qubit_gates_layer(qubits, params):
    """Compose a list of random one-qubit parametrized gates."""

    rand_oneq = np.random.choice(get_parametrized_single_qubit_gates(), size=len(qubits), replace=True)
    return [g(e)(q) for g,q,e in zip(rand_oneq, qubits, params)]


def random_two_qubit_gates_layer(qubits, params):
    """Compose a list of random two-qubit parametrized gates."""
    n = len(qubits)
    qubit_pairs = ((qubits[i], qubits[(i+1)%n]) for i in range(n))
    rand_twoq = np.random.choice(get_parametrized_two_qubit_gates(), size=n, replace=True)
    return [g(exponent=e)(qi,qj) for g,(qi,qj),e in zip(rand_twoq, qubit_pairs, params)]


def update_params(circuit, params):
    """Competitor method to cirq.ParamResolver."""
    new_op_tree = []
    for op, param in zip(circuit.all_operations(), params):
        new_op_tree.append(op.gate._with_exponent(param/np.pi)(*op.qubits))
    return cirq.Circuit.from_ops(new_op_tree)


def trial(depth, qubits, params):
    """Initialize a circuit according to a set of parameters.

        `params` should be shape (n_qubits * depth, )

        returns a nested list of ops (OP_TREE)
    """
    out = []
    n_qubits = len(qubits)
    for d in range(depth):
        # slice params into layer-sized chunks
        param_subset = params[n_qubits*d:n_qubits**(d+1)]
        out.append(random_single_qubit_gates_layer(qubits, param_subset))
        # FIXME: _with_exponent doesn't work as expected with 2-qubit gates..
        # if not (d % 2):
        #
        # out.append(random_single_qubit_gates_layer(qubits, param_subset))
        # else:
        #     out.append(random_two_qubit_gates_layer(qubits, param_subset))
    return out


def timeit_n_rounds_k_updates(qubits, depth, sim_trials, n_param_updates):
    """
        Args:
            qubits: QubitId representation of circuit qubits
            depth (int): Circuit depth.
            sim_trials (int): Number of trials to run each simulation for the
                purpose of averaging
            n_param_updates (int): Number of updates to invoke per trial.

        Returns:
            Array of shape (sim_trials, ) containing times for each trial
    """

    """
    TIMING:
    Op placeholder resolution: Each trial consists of the time to run all of:
      1. call to TFWaveFunctionSimulator().simulate with feed_dict kwarg
      2. put a random wavefunction element to outcomes[v_ind]

    note that the feed dicts are constructed ahead of time, which
    is actually generous to the timing.

    circuit reconstructions: Each trial consists of the time to run all of:
      1. Iteratively copy the existing circuit op-wise, inserting new
           angles according to randomly generated params
      2. call to cirq.Simulator() using this new state
      3. put a random wavefunction element to outcomes[v_ind]
    """
    tfcirq_times = []
    float_times = []
    n_qubits = len(qubits)
    for k in range(sim_trials):
        """
        VERIFICATION:
        Results between two resolved circuits will be compared
        according to the amplitude of the wavefunction at a random index
        `v_ind`, for every trial, for every parameter in the param updates.
        """
        tfcirq_outcomes = np.zeros(n_param_updates).astype(np.complex64)
        float_outcomes = np.zeros(n_param_updates).astype(np.complex64)
        v_ind = np.random.randint(2**n_qubits - 1)

        # precompute all parameter updates to apply to both circuits
        all_params = np.random.rand(n_param_updates, n_qubits*depth)
        all_params = np.ones((n_param_updates, n_qubits*depth))
        # initialize a persistent sympy-parametrized circuit
        symbol_strings = []
        for i in range(n_qubits*depth):
            symbol_strings.append("{}".format(i) )
        layer_symbols = [sympy.Symbol(s) for s in symbol_strings]
        global trial
        layers = trial(depth, qubits, layer_symbols)
        base_circuit = cirq.Circuit.from_ops([g for l in layers for g in l])

        # consistent initial state prep
        x = np.ones(2**n_qubits, dtype=np.complex64) / np.sqrt(2**n_qubits)

        # Convert circuit parameters over to placeholders
        placeholders = [tf.placeholder(tf.complex64, shape=(), name=s) for s in symbol_strings]
        placeholder_names = ["{}:0".format(s) for s in symbol_strings]
        tfcirq_circuit = update_params(base_circuit, placeholders)
        feed_dicts = [dict(zip(placeholder_names, all_params[j][:])) for j in range(n_param_updates)]

        start = timer()
        for j in range(n_param_updates):
            final_state = TFWaveFunctionSimulator(dtype=tf.complex64).simulate(
                tfcirq_circuit, initial_state=np.copy(wf))
            with tf.Session() as sess:
                tfcirq_outcomes[j] = sess.run(final_state)[v_ind]

        tfcirq_times_trial_time = timer() - start
        tfcirq_times.append(tfcirq_times_trial_time)

        # initialize a copy of the circuit, this time using hard-coded angles
        # the symbols will be overwritten with the first update.
        float_circuit = base_circuit.copy()
        start = timer()
        for j in range(n_param_updates):
            # Regenerate _entire_ circuit with updates to float values
            # each time includes the circuit construction time
            float_circuit = update_params(float_circuit, all_params[j][:])
            float_outcomes[j] = cirq.Simulator().simulate(float_circuit, initial_state=np.copy(x)).final_state[v_ind]

        float_trial_time = timer() - start
        float_times.append(float_trial_time)

        np.testing.assert_array_almost_equal(float_outcomes, sympy_outcomes)
        print("trial {}:")
        print("  cirq-tf: ", sympy_trial_time)
        print("  float: ", float_trial_time)

    return np.asarray(sympy_times), np.asarray(float_times)

"""
Benchmark description:

    For number of parameters N_PARAMS, construct a circuit of mixed one- and
    two-qubit parametrized gates of total depth DEPTH. Compare the following
    param resolution methods:
        (a) Sympy variables with native cirq ParamResolver feed dict
        (b) Circuit reconstructed directly from new parameters
"""
SIM_TRIALS = 20  # average runtime over this many runs
N_PARAMS = 20
N_PARAM_UPDATES = 100 # how many times to replace the parameters
# DEPTH = 10  # depth of dense matrix layers
MAX_N = 5  # run trials for n_qubits = 2...MAX_N
# N_QUBITS = list(range(2, MAX_N))
N_QUBITS = np.asarray([5])
DEPTHS = np.asarray(range(1,10))
all_sympy_runs = np.zeros((len(N_QUBITS), len(DEPTHS), SIM_TRIALS))
all_float_runs = np.zeros((len(N_QUBITS), len(DEPTHS), SIM_TRIALS))

for i, n_qubits in enumerate(N_QUBITS):
    for j, depth in enumerate(DEPTHS):
        qubits = cirq.LineQubit.range(n_qubits)
        print("simulating {} qubits".format(n_qubits))
        s1, f1 = timeit_n_rounds_k_updates(qubits, depth, SIM_TRIALS, N_PARAM_UPDATES)
        all_sympy_runs[i][j][:] = s1
        all_float_runs[i][j][:] = f1

np.save('cirq_sympy_bench.npy', np.asarray(all_sympy_runs))
np.save('cirq_float_bench.npy', np.asarray(all_float_runs))
np.save('cirq_sympy_meta.npy', np.array([N_QUBITS, SIM_TRIALS, DEPTHS, N_PARAM_UPDATES]))
