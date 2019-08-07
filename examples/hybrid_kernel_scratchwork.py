import numpy as np
import itertools
import cirq
import tensorflow as tf

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "..")
from tf_simulator import (
    TFWaveFunctionSimulator
)
from utils import data_tools, qkernel
import sklearn
from sklearn import svm
import cirq



class TFKernelEncoderCircuit(qkernel.QuantumPreperationCircuit):
    """TF simulation of the quantum kernel circuit."""

    def __init__(self, n_qubits, n_layers=1):
        """Stripped-down TFSimulator for producing kernel computations.

        Gate phases are stored as PLACEHOLDERS in this model, and therefore
        expect a feed_dict during graph execution, and will not interface
        with an optimizer.

        NOTE: Cannot use singleZ, does not support density matrix simulation.
        """
        super().__init__(n_qubits, n_layers=n_layers, use_singleZ=False,
            simulator=TFWaveFunctionSimulator())

    def _create_U(self):
        """overwrite to remove Sympy parametrization."""
        U_circuit = cirq.Circuit()
        p = 0
        for n, j in enumerate(self.qubits):
            for k in self.qubits[n + 1:]:
                theta = tf.placeholder(dtype=tf.complex64, shape=(), name="Phi_{}".format(p))
                U_circuit.append([cirq.ZZPowGate(exponent=theta)(j, k)],
                                  strategy=cirq.InsertStrategy.EARLIEST)
                p += 1
        return U_circuit

    def _simulate_circuit(self, init_state, phi_batch):
        """Simulates the kernel circuit for a given initial state and phases.

        Helper method for `get_final_state` and `get_final_density_matrix`.
        """
        states = []
        for ib, phi in enumerate(phi_batch[0]):
            feed_dict = {"Phi_{}:0".format(i): p for i, p in enumerate(phi)}
            circuit_op = self.simulator.simulate(
                self.circuit, initial_state=np.copy(init_state))
            with tf.Session() as sess:
                wf = sess.run(circuit_op, feed_dict=feed_dict).reshape(-1)
            states.append(wf)
        return np.array(states)


class FeedForwardQuantumKernelCircuit(TFKernelEncoderCircuit):
    """Provides a layer to pass Phi(x) -> psi(Phi(x))."""

    def __init__(self, variables, n_qubits, n_layers=1):
        """Stripped-down TFSimulator for interfacing kernel computations.

        Gate phases are stored as VARIABLES in this model, and therefore
        expect execute operations on readouts of these variables during
        execution.

        NOTE: Cannot use singleZ, does not support density matrix simulation.

        Args:
            variable (tf.Variable): Variables to assign as gate parameters in
                the circuit. Optionally, pass in:
                    FeedForwardQuantumKernelCircuit.declare_variables()
            n_qubits (int): Number of qubits in the circuit.
            n_layers (int): Number of layers to construct in the circuit.
        """
        self.variables = variables
        super().__init__(n_qubits, n_layers=n_layers, use_singleZ=False,
            simulator=TFWaveFunctionSimulator())


    @classmethod
    def declare_variables(qubits):
        """Optional helper method to declare the input variables.

        Use this if this circuit model is the first (input) op for the tf
        graph being executed.
        """
        out = {}
        p = 0
        for n, j in enumerate(qubits):
            for k in qubits[n + 1:]:
                var_name = "Phi_{}".format(p)
                theta = tf.Variable(dtype=tf.complex64, shape=(), name=var_name)
                out[var_name] = theta
                p += 1
        return out

    def _create_U(self):
        """Overwrite to implement Variables for non-I/O intermediate layer.

        During this process, fetch variables (slices) in a fixed order from
        self.variables and assign them as gate parameters.
        """
        U_circuit = cirq.Circuit()
        p = 0
        for n, j in enumerate(self.qubits):
            for k in self.qubits[n + 1:]:
                U_circuit.append(
                    [cirq.ZZPowGate(exponent=self.variables[p])(j, k)],
                    strategy=cirq.InsertStrategy.EARLIEST)
                p += 1
        return U_circuit

    def _simulate_circuit(self, phi):
        """Simulates the kernel circuit for a single choice of phase function.

        Given a set of dC2 phases corresponding to a data point of length d,
        feed forward a wavefunction generated from the kernel encoding circuit
        applying these phi(x) phases.

        Returns:
            circuit_op: A TF Graph construct for the circuit
            feed_dict

        """
        init_state = (np.ones(self.n_states, dtype=np.complex64) /
                      np.sqrt(self.n_states))
        circuit_op = self.simulator.simulate(
            self.circuit, initial_state=np.copy(init_state))
        return circuit_op


def test_tf_backend_consistency_kernel_circuit():
    """Validate tf performance for known kernel circuit."""

    data = data_tools.load_star_galaxy_pca(d=8)
    n_bench = 20
    Xbench = data['train_x'][:n_bench]

    n_qubits = 8
    n_layers = 1
    data_dim = 8
    tf_model = qkernel.QuantumLinearKernel(n_qubits=n_qubits,
                                           data_dim=data_dim,
                                           n_layers=n_layers,
                                           use_singleZ=False,
                                           noise=False)

    tf_model.quantum_estimator = TFKernelEncoderCircuit(n_qubits=n_qubits,
                                                        n_layers=n_layers)

    cirq_model = qkernel.QuantumLinearKernel(n_qubits=n_qubits,
                                             data_dim=data_dim,
                                             n_layers=n_layers,
                                             use_singleZ=False,
                                             noise=False)

    # Initialize both models
    for model in [tf_model, cirq_model]:
        w_pairs = np.zeros_like(model.W)
        # set a pairwise mask for linear kernel
        for i, p in enumerate(itertools.combinations(range(8), 2)):
            w_pairs[p[0], i] = 1
            w_pairs[p[1], i] = 1
        c = 0.4
        model.set_params(c * w_pairs)

    tf_results = tf_model.map_data(np.copy(Xbench))
    cirq_results = cirq_model.map_data(np.copy(Xbench))
    for tf_x, cirq_x in zip(tf_results, cirq_results):
        np.testing.assert_array_almost_equal(tf_x, cirq_x, decimal=3)


def main():
    """Connect a shallow NN to a kernel encoder circuit."""

    # number of neurons in each layer
    input_num_units = 4
    hidden_num_units = 8
    output_num_units = 6 # dC2 = d(d-1)/2

    # Data I/O
    data = data_tools.load_star_galaxy_pca(d=8)
    n_bench = 20
    Xbench = data['train_x'][:n_bench]

    # define placeholders for resolving encoder circuit
    x = tf.placeholder(tf.complex64, [None, input_num_units])

    # hyperparameters
    seed = 31415
    epochs = 5
    batch_size = 128
    learning_rate = 0.01

    # hidden layer
    w1 = tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed))
    b1 = tf.Variable(tf.random_normal([hidden_num_units], seed=seed))

    # output layer
    w2 = tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))
    b2 = tf.Variable(tf.random_normal([output_num_units], seed=seed))

    # Define the architecture as RELU(Wx + b)
    hidden_layer = tf.add(tf.matmul(x, w1), b1)
    hidden_layer = tf.nn.relu(hidden_layer)
    output_layer = tf.add(tf.matmul(hidden_layer, w2), b2)

    # define circuit architecture
    n_qubits = 8
    n_layers = 1
    data_dim = 8

    # construct TF model for circuit and
    # overwrite the qkernel circuit with an op, feeding in variables as parameters
    tf_model = qkernel.QuantumLinearKernel(n_qubits=n_qubits,
                                           data_dim=data_dim,
                                           n_layers=n_layers,
                                           use_singleZ=False,
                                           noise=False)
    tf_model.quantum_estimator = FeedForwardQuantumKernelCircuit(
        variables=output_layer, n_qubits=n_qubits, n_layers=n_layers)

    # define loss to compute over the output wavefunction
    
