"""Evan's tests for experimental code. Do not edit."""
import numpy as np
import tensorflow as tf
import cirq
import sys
sys.path.insert(0, ".")
from tf_apply_unitary import (
    tf_apply_unitary,
    ApplyTFUnitaryArgs,
)

INITIAL_STATE = np.asarray([1, 0])
TEST_VAR = tf.Variable(1.0)
TEST_GATES = [
    cirq.YPowGate(exponent=TEST_VAR)(cirq.LineQubit(0)),
    cirq.YPowGate(exponent=TEST_VAR)(cirq.LineQubit(0))
]
TEST_CIRCUIT = cirq.Circuit.from_ops(TEST_GATES)


m = tf.cast(tf.diag([1, -1]), tf.complex64)

class HasUnitary:
    def _unitary_(self) -> tf.Tensor:
        return m

class HasApplyReturnsNotImplementedButHasUnitary:
    def _apply_unitary_(self, args: ApplyTFUnitaryArgs):
        return NotImplemented

    def _unitary_(self) -> tf.Tensor:
        return m

class HasApplyOutputInBuffer:
    """Enforce control dependencies during buffer usage."""
    def _apply_unitary_(self, args: ApplyTFUnitaryArgs) -> tf.Tensor:
        zero = args.subspace_index(0)
        one = args.subspace_index(1)
        inds = [zero, one]
        ref0 = args.target_tensor[zero]
        ref1 = tf.scalar_mul(-1, args.target_tensor[one])
        refs = [ref0, ref1]
        x = args.available_buffer
        with tf.control_dependencies([x[inds[i]].assign(refs[i]) for i in range(2)]): # should give the list of slice assignment here
            x = tf.identity(x) #convert to a tensor
        return x

class HasApplyMutateInline:
    """Promotion to variable for inline mutation."""
    def _apply_unitary_(self, args: ApplyTFUnitaryArgs) -> tf.Tensor:
        # FIXME: NOT SURE IF THIS IS GOING TO TURN OUT WELL...
        one = [args.subspace_index(1)]
        ref1 = [tf.scalar_mul(-1, args.target_tensor[one[0]])]
        x = tf.Variable(args.target_tensor)
        x = x[one[0]].assign(ref1[0])
        return x


passes = [
    HasUnitary(),
    HasApplyReturnsNotImplementedButHasUnitary(),
    HasApplyOutputInBuffer(),
    HasApplyMutateInline(),
]


def make_input():
    return tf.cast(tf.ones((2, 2)), tf.complex64)


def test_apply_tf_unitary():

    buf = tf.Variable(tf.zeros((2, 2), dtype=tf.complex64))
    def assert_works(val):
        expected_outputs = [
            np.array([1, 1, -1, -1]).reshape((2, 2)),
            np.array([1, -1, 1, -1]).reshape((2, 2)),
        ]
        for axis in range(2):
            result = tf_apply_unitary(
                val, ApplyTFUnitaryArgs(make_input(), buf, [axis]))

            sess.run(tf.global_variables_initializer())
            result = sess.run(result)
            np.testing.assert_allclose(result, expected_outputs[axis])
            print(f"{val} works")

    for s in passes:
        sess = tf.Session()
        assert_works(s)
        assert tf_apply_unitary(
            s,
            ApplyTFUnitaryArgs(make_input(), buf, [0]),
            default=None) is not None
