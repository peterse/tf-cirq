import random

from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt

import cirq


def get_parametrized_two_qubit_gates():
    return [
        cirq.SwapPowGate,
        cirq.CNotPowGate,
        cirq.ISwapPowGate,
        cirq.ZZPowGate,
        cirq.CZ,
    ]

def get_two_qubit_gates():
    return [
        cirq.CNOT,
        cirq.SWAP,
    ]

def get_parametrized_single_qubit_gates():
    return [
        cirq.Rx,
        cirq.Ry,
        cirq.Rz,
    ]

def get_single_qubit_gates():
    return [
        cirq.X,
        cirq.Y,
        cirq.Z,
        cirq.H,
        cirq.S,
        cirq.T,
    ]


ALL_GATES = get_single_qubit_gates() + \
            get_parametrized_single_qubit_gates() + \
            get_two_qubit_gates() + \
            get_parametrized_two_qubit_gates()


def random_single_qubit_gates_layer(qubits, parametrized=False):
    """Compose dense layer of single-qubit gates."""

    all_gates = get_single_qubit_gates()
    if parametrized:
        all_gates = get_parametrized_single_qubit_gates()

    gates_out = []
    for q1 in qubits:
        gate_func = random.choice(all_gates)
        rand_param = random.uniform(-np.pi, np.pi)
        if parametrized:
            gate = gate_func(rand_param)(q1)
        else:
            gate = gate_func(q1)
        gates_out.append(gate)

    return gates_out


def random_two_qubit_gates_layer(qubits, parametrized=False):
    """Compose dense layer of single-qubit gates."""

    all_gates = get_two_qubit_gates()
    if parametrized:
        all_gates = get_parametrized_two_qubit_gates()

    gates_out = []
    for i2 in range(1, len(qubits), 2):
        q2 = qubits[i2]
        q1 = qubits[i2 - 1]
        gate_func = random.choice(all_gates)
        rand_param = random.uniform(-np.pi, np.pi)
        if parametrized:
            gate = gate_func(rand_param)(q1, q2)
        else:
            gate = gate_func(q1, q2)
        gates_out.append(gate)

    return gates_out


def specialized_kernel_circuit(qubits):
    """This circuit is N choose 2 exp(ZZ) gates interspersed with H wall."""
    out = []
    for i in range(2):
        H_gates = [cirq.H(q) for q in qubits]
        ZZ_gates = []
        for n, qj in enumerate(qubits):
            for qk in qubits[n + 1:]:
                theta = random.uniform(-np.pi, np.pi)
                ZZ_gates.append(cirq.ZZPowGate(exponent=theta)(qj, qk))
        out.append(H_gates + ZZ_gates)
    return out


TRIAL_TYPES = []
# set up two trials:
# 1. interspersed single and two-qubit unparametrized gates
# 2. interspersed single and two-qubit parametrized gates
# 3. specialized circuit for quantum kernel method
def trial(depth, qubits, parametrized):
    out = []
    for d in range(depth):
        if not (d % 2):
            out.append(random_single_qubit_gates_layer(qubits, parametrized))
        else:
            out.append(random_two_qubit_gates_layer(qubits, False))
    return out


def timeit_n_rounds(qubits, layers, sim_trials):
    cirqtimes = []
    n_qubits = len(qubits)
    for _ in range(sim_trials):
        # initial state prep
        x = np.zeros((2**n_qubits, ), dtype=np.complex64)
        x[np.random.randint(2**n_qubits)] = 1

        # prepare cirq apply unitary routine on flattened gatelist
        circuit = cirq.Circuit.from_ops([g for l in layers for g in l])
        start = timer()
        cirq.Simulator().simulate(circuit, initial_state=np.copy(x))
        end = timer()
        trial = end - start
        cirqtimes.append(trial)

    return np.asarray(cirqtimes)


SIM_TRIALS = 20  # average runtime over this many runs
DEPTH = 20  # depth of dense matrix layers
MAX_N = 20  # run trials for n_qubits = 2...MAX_N
N_QUBITS = list(range(2,MAX_N))
VERSION = cirq.__version__
avg_cirq = []
avg_cirq_param = []
avg_cirq_kernel = []
for n_qubits in N_QUBITS:
    if VERSION == '0.4.0':
        qubits = list(range(n_qubits))
    elif VERSION == '0.5.0':
        qubits = cirq.LineQubit.range(n_qubits)
    print("simulating {} qubits".format(n_qubits))
    no_param = trial(DEPTH, qubits, False)
    c1 = timeit_n_rounds(qubits, no_param, SIM_TRIALS)
    avg_cirq.append(np.mean(c1))

    param = trial(DEPTH, qubits, True)
    c2 = timeit_n_rounds(qubits, param, SIM_TRIALS)
    avg_cirq_param.append(np.mean(c2))

    kernel = specialized_kernel_circuit(qubits)
    c3 = timeit_n_rounds(qubits, kernel, SIM_TRIALS)
    avg_cirq_kernel.append(np.mean(c3))

np.save('cirq_v{}_bench.npy'.format(VERSION), np.array([N_QUBITS, avg_cirq, avg_cirq_param, avg_cirq_kernel]))
np.save('cirq_v{}_meta.npy'.format(VERSION), np.array([SIM_TRIALS, DEPTH]))
