"""Evan's tests for experimental code. DO NOT EDIT."""
import pytest

import cirq
import tensorflow as tf
import numpy as np
from tfc.protocols import tf_gate_wrapper, ALL_WRAPPERS


def q(i):
    return cirq.LineQubit(i)

@pytest.mark.parametrize('g', [cirq.X, cirq.Y, cirq.Z, cirq.H])
def test_tf_gate_wrapper_gate_inheritance(g):
    """Wrapping a gate instance with a wrapper base class."""
    inst = g(q(0))
    wrapped = tf_gate_wrapper(inst, tf.complex64)
    with tf.Session() as sess:
        tf_inst = sess.run(wrapped._tensor)
    np.testing.assert_array_almost_equal(cirq.unitary(inst), tf_inst)


@pytest.mark.parametrize('g', [cirq.Rx, cirq.Ry, cirq.Rz])
def test_tf_gate_wrapper_gate_inheritance_parametrized(g):
    inst = g(0.01)(q(0))
    wrapped = tf_gate_wrapper(inst)
    unitary = cirq.unitary(inst)
    wrapped = tf_gate_wrapper(inst, tf.complex64)
    with tf.Session() as sess:
        tf_inst = sess.run(wrapped._tensor)
    np.testing.assert_array_almost_equal(cirq.unitary(inst), tf_inst)


@pytest.mark.parametrize('g', [cirq.CNOT, cirq.SWAP])
def test_tf_gate_wrapper_gate_inheritance(g):
    inst = g(q(0), q(1))
    wrapped = tf_gate_wrapper(inst, tf.complex64)
    with tf.Session() as sess:
        tf_inst = sess.run(wrapped._tensor).reshape((4,4))
    np.testing.assert_array_almost_equal(cirq.unitary(inst), tf_inst)


@pytest.mark.parametrize('g', [cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate, cirq.HPowGate])
def test_tf_gate_wrapper_single_qubit_eigengate(g):
    inst = g(exponent=1.5)(q(0))
    wrapped = tf_gate_wrapper(inst, tf.complex64)
    with tf.Session() as sess:
        tf_inst = sess.run(wrapped._tensor)
    np.testing.assert_array_almost_equal(cirq.unitary(inst), tf_inst)


@pytest.mark.parametrize('g', [cirq.CNotPowGate, cirq.SwapPowGate,])
def test_tf_gate_wrapper_single_qubit_eigengate(g):
    inst = g(exponent=3.84)(q(0), q(1))
    wrapped = tf_gate_wrapper(inst)
    with tf.Session() as sess:
        tf_inst = sess.run(wrapped._tensor).reshape((4,4))
    np.testing.assert_array_almost_equal(cirq.unitary(inst), tf_inst)


def test_tf_gate_wrapper_parity_gate():
    for g in [cirq.ZZPowGate]:
        inst = g(exponent=3.84)(q(0), q(1))
        wrapped = tf_gate_wrapper(inst)
        with tf.Session() as sess:
            tf_inst = sess.run(wrapped._tensor).reshape((4,4))
        np.testing.assert_array_almost_equal(cirq.unitary(inst), tf_inst)

def test_tf_gate_wrapper_tensor_inputs():
    # TODO
    init_t = np.pi /2
    t = tf.Variable(init_t)
    inst = cirq.YPowGate(exponent=t)(q(0))
    wrapped = tf_gate_wrapper(inst)
    print(wrapped._tensor)

if __name__ == "__main__":
    test_tf_gate_wrapper_gate_inheritance()
