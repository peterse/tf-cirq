"""Evan's tests for experimental code. Do not edit."""
import random
import pytest
import numpy as np

import tensorflow as tf
import cirq

from tfc.sim import TFWaveFunctionSimulator


def q(i):
    return cirq.LineQubit(i)

# !hep-qml CHANGES: testing assertion
def get_parametrized_two_qubit_gates():
    return [
        cirq.SwapPowGate,
        cirq.CNotPowGate,
        # cirq.ISwapPowGate,
        cirq.ZZPowGate,
        # cirq.CZ # !hep-qml FIXME: doesn't accept same kwargs as above...
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
            gate = gate_func(exponent=rand_param)(q1, q2)
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


# <<<<<<<

def test_tf_wavefunction_simulator_instantiate():
    _ = TFWaveFunctionSimulator()


@pytest.mark.parametrize('dtype', [tf.complex64, tf.complex128])
def test_run_bit_flips(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = TFWaveFunctionSimulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit.from_ops((cirq.X**b0)(q0), (cirq.X**b1)(q1))
            circuit_op = simulator.simulate(circuit)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            wf = sess.run(circuit_op)
        measurements = cirq.sample_state_vector(wf, [0, 1])
        np.testing.assert_array_almost_equal(measurements[0], [b0, b1])
        expected_state = np.zeros(shape=(2, 2))
        expected_state[b0][b1] = 1.0
        cirq.testing.assert_allclose_up_to_global_phase(wf.reshape(-1), np.reshape(expected_state, 4), atol=1e-6)


@pytest.mark.parametrize('dtype', [tf.complex64, tf.complex128])
def test_run_correlations(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = TFWaveFunctionSimulator(dtype=dtype)
    circuit = cirq.Circuit.from_ops(cirq.H(q0), cirq.CNOT(q0, q1))
    for _ in range(10):
        circuit_op = simulator.simulate(circuit)
        with tf.Session() as sess:
            wf = sess.run(circuit_op)
        measurements = cirq.sample_state_vector(wf.reshape(-1), [0, 1])
    # TODO: assertion?


# # !hep-qml CHANGES: testing assertion
# def test_tf_wavefunction_simulator_random_circuit():
#
#     # generate a random circuit for validation run
#     # rely on implicit scalar-tensor upconversion for parameters
#     depth = 5
#     qubits = cirq.LineQubit.range(4)
#     gates = []
#     for i in range(depth):
#         gates +=  random_single_qubit_gates_layer(qubits, parametrized=True)
#         gates += random_two_qubit_gates_layer(qubits, parametrized=True)
#     circuit = cirq.Circuit.from_ops(gates)
#     cirq_result = cirq.Simulator().simulate(circuit).final_state
#
#     circuit_op = TFWaveFunctionSimulator().simulate(circuit)
#     with tf.Session() as sess:
#         tf_result = sess.run(circuit_op).reshape((2**len(qubits), ))
#     np.testing.assert_array_almost_equal(tf_result, cirq_result)
#     # <<<<<<<
#
# # !hep-qml CHANGES: testing assertion
@pytest.mark.parametrize('g', get_single_qubit_gates())
def test_tf_wavefunction_simulator_vs_cirq_single_qubit_gates(g):

    wf = np.complex64(cirq.testing.random_superposition(2))
    # print(rand_init)
    for wf in [np.array([1, 0], dtype=np.complex64), np.array([0, 1], dtype=np.complex64)]:
        circuit = cirq.Circuit.from_ops(g(q(0)))
        cirq_result = cirq.Simulator().simulate(
            circuit, initial_state=np.copy(wf)).final_state

        circuit_op = TFWaveFunctionSimulator(dtype=tf.complex64).simulate(
            circuit, initial_state=np.copy(wf))
        with tf.Session() as sess:
            tf_result = sess.run(circuit_op).reshape(-1)
    # !hep-qml CHANGES: testing assertion
    np.testing.assert_array_almost_equal(tf_result, cirq_result)

@pytest.mark.parametrize('g', get_parametrized_single_qubit_gates())
@pytest.mark.parametrize('e', [0, 1, np.random.rand()])
def test_tf_wavefunction_simulator_vs_cirq_parametrized_single_qubit_gates(g, e):
    inst = g(e)(q(0))
    circuit = cirq.Circuit.from_ops(inst)
    # wrapped = tf_gate_wrapper(inst)
    wf = cirq.testing.random_superposition(2).astype(np.complex64)
    cirq_result = cirq.Simulator().simulate(
        circuit, initial_state=np.copy(wf)).final_state
    circuit_op = TFWaveFunctionSimulator(dtype=tf.complex64).simulate(
        circuit, initial_state=np.copy(wf))
    with tf.Session() as sess:
        tf_result = sess.run(circuit_op).reshape(-1)
    np.testing.assert_array_almost_equal(tf_result, cirq_result, decimal=4)


@pytest.mark.parametrize('g', get_two_qubit_gates())
def test_tf_wavefunction_simulator_vs_cirq_two_qubit_gates(g):
    circuit = cirq.Circuit.from_ops(g(q(0), q(1)))
    wf = cirq.testing.random_superposition(4).astype(np.complex64)
    cirq_result = cirq.Simulator().simulate(
        circuit, initial_state=np.copy(wf)).final_state
    circuit_op = TFWaveFunctionSimulator(dtype=tf.complex64).simulate(
        circuit, initial_state=np.copy(wf))
    with tf.Session() as sess:
        tf_result = sess.run(circuit_op).reshape(-1)
    np.testing.assert_array_almost_equal(tf_result, cirq_result)


@pytest.mark.parametrize('g', get_parametrized_two_qubit_gates())
@pytest.mark.parametrize('e', [0, 1, np.random.rand()])
def test_tf_wavefunction_simulator_vs_cirq_parametrized_two_qubit_gates(g, e):
    circuit = cirq.Circuit.from_ops(g(exponent=e)(q(0), q(1)))
    wf = cirq.testing.random_superposition(4).astype(np.complex64)
    cirq_result = cirq.Simulator().simulate(
        circuit, initial_state=np.copy(wf)).final_state
    circuit_op = TFWaveFunctionSimulator(dtype=tf.complex64).simulate(
        circuit, initial_state=np.copy(wf))
    with tf.Session() as sess:
        tf_result = sess.run(circuit_op).reshape(-1)
    np.testing.assert_array_almost_equal(tf_result, cirq_result)


def test_resolve_scalar_placeholder():
    """Resolve scalar placeholders via feed_dict."""
    wf = cirq.testing.random_superposition(4).astype(np.complex64)
    params = [0.83, 1.2]
    theta_x = tf.placeholder(tf.complex64, shape=(), name="theta_x")
    theta_y = tf.placeholder(tf.complex64, shape=(), name="theta_y")
    placeholder_circuit = cirq.Circuit.from_ops([
        cirq.Rx(theta_x)(q(0)),
        cirq.Ry(theta_y)(q(1)),
    ])
    circuit_op = TFWaveFunctionSimulator(dtype=tf.complex64).simulate(
        placeholder_circuit, initial_state=wf)
    feed_dict = dict(zip([theta_x, theta_y], params))
    with tf.Session() as sess:
        tf_result = sess.run(circuit_op, feed_dict=feed_dict).reshape(-1)

    circuit = cirq.Circuit.from_ops([
        cirq.Rx(params[0])(q(0)),
        cirq.Ry(params[1])(q(1)),
    ])
    cirq_result = cirq.Simulator().simulate(
        circuit, initial_state=np.copy(wf)).final_state
    np.testing.assert_array_almost_equal(tf_result, cirq_result)



def test_resolve_named_placeholder():
    """Resolve named placeholders via feed_dict."""
    wf = cirq.testing.random_superposition(2).astype(np.complex64)

    # placeholders can be instantiated without references in circuit gates!
    var_names = ["v_{}".format(i) for i in range(5)]
    params = np.random.randn(5)
    placeholder_circuit = cirq.Circuit()
    for i in range(5):
        placeholder_circuit += cirq.Rx(
            tf.placeholder(tf.complex64, shape=(), name=f"theta_{i}"))(q(0))

    circuit_op = TFWaveFunctionSimulator(dtype=tf.complex64).simulate(
        placeholder_circuit, initial_state=wf)
    placeholder_names = ["theta_{}:0".format(i) for i in range(5)]
    feed_dict = dict(zip(placeholder_names, params))
    with tf.Session() as sess:
        tf_result = sess.run(circuit_op, feed_dict=feed_dict).reshape(-1)

    circuit = cirq.Circuit()
    for theta in params:
        circuit += cirq.Rx(theta)(q(0))
    cirq_result = cirq.Simulator().simulate(
        circuit, initial_state=np.copy(wf)).final_state
    np.testing.assert_array_almost_equal(tf_result, cirq_result)
