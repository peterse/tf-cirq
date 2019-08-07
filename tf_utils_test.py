import tensorflow as tf
import cirq
import sys
sys.path.insert(0, ".")
from tf_simulator import (
    TFWaveFunctionSimulator
)
from tf_utils import tensorboard_session

import numpy as np

def q(i):
    return cirq.LineQubit(i)

def compile_tensorboard_session():
    theta = tf.Variable(np.pi)
    circuit = cirq.Circuit.from_ops(
        cirq.Ry(theta)(q(0)),
        cirq.CNOT(q(0), q(1))
    )
    tf.summary.scalar('theta', theta)

    initial_state = np.asarray([1, 0, 0, 0])
    circuit_op = TFWaveFunctionSimulator(dtype=tf.complex64).simulate(circuit, initial_state=initial_state)
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     wf = sess.run(circuit_op)
    # return

    tensorboard_session(circuit_op, {theta:np.pi/2})

if __name__ == "__main__":
    compile_tensorboard_session()
