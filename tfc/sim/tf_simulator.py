import tensorflow as tf
import numpy as np

from typing import Any, Dict, List, Union
from cirq.sim import simulator
from cirq import circuits, ops, study

from tfc.protocols import tf_apply_unitary, ApplyTFUnitaryArgs, tf_gate_wrapper



# Mutable named tuple to hold state and a buffer.
class _StateAndBuffer():
    def __init__(self, state, buffer):
        self.state = state
        self.buffer = buffer

#
# class TFWaveFunctionSimulatorState:
#     """
#     Container class for a quantum state and its target qubits.
#     """
#
#     def __init__(self, state, qubits):
#         """Create a new State from a wavefunction or density matrix representation.
#         Args:
#             tensor: A vector or tensor of state amplitudes
#             qubits: A sequence of qubit indices
#                 FIXME: interface with location..? do I care?
#         """
#
#         # TODO: input validation
#         self._tensor = tf.convert_to_tensor(
#             value=state, dtype=tf.complex64
#         )
#         self._qubits = qubits
#

class TFWaveFunctionSimulator(
    simulator.SimulatesSamples,
    simulator.SimulatesFinalState,
    ):
    """
    methods:
        self.simulate(
            program: Union[circuits.Circuit, schedules.Schedule],
            param_resolver = None,
            qubit_order  ops.QubitOrder.DEFAULT,
            initial_state = None
        )

    TODO:
        - validate input state
        - default initial State
        - proper typing on intial state
    """
    def __init__(self, *, dtype=tf.complex64):
        """A sparse matrix simulator.

        Args:
            dtype: The `tf.dtype` used by the simulation. One of
            `tf.complex64` or `tf.complex128`
        """
        if dtype not in {tf.complex64, tf.complex128}:
            raise ValueError(
                'dtype must be complex64 or complex128 but was {}'.format(
                    dtype))
        self._dtype = dtype

    # SimulatesSample abc method
    def _run(
        self,
        circuit: circuits.Circuit,
        param_resolver: study.ParamResolver,
        repetitions: int
    ) -> Dict[str, np.ndarray]:
        """Run a simulation, mimicking quantum hardware.

        Args:
            circuit: The circuit to simulate.
            param_resolver: Parameters to run with the program.
            repetitions: Number of times to repeat the run.

        Returns:
            A dictionary from measurement gate key to measurement
            results. Measurement results are stored in a 2-dimensional
            numpy array, the first dimension corresponding to the repetition
            and the second to the actual boolean measurement results (ordered
            by the qubits being measured.)
        """
        raise NotImplementedError()

    def _simulate_unitary(self, op: ops.Operation, data: _StateAndBuffer,
            indices: List[int]) -> _StateAndBuffer:
        """Core method: Compose the next chunk of the computation graph."""
        data.state = tf_apply_unitary(
            op,
            args=ApplyTFUnitaryArgs(target_tensor=data.state,
                                    available_buffer=data.buffer,
                                    axes=indices))
        return _StateAndBuffer(data.state, None) # Flush the buffer
        # CHECKME: drop any connection to previous state info?
        # if result is data.buffer:
        #     data.buffer = data.state
        # data.state = result

    # SimulatesFinalState abc method
    # FIXME: I don't want to use cirq's built-in param resolution here...
    def simulate_sweep(
        self,
        program: Union[circuits.Circuit],
        params: study.Sweepable,
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
        initial_state: Any = None) -> List[tf.Tensor]:
        """Simulates the supplied Circuit or Schedule.

        Note: This differs from cirq.WaveFunctionSimulator in the return
        type, which is List[tf.tensor] instead of List[SimulationTrialResult].

        This method returns a result which allows access to the entire
        wave function. In contrast to simulate, this allows for sweeping
        over different parameter values.

        Args:
            program: The circuit or schedule to simulate.
            params: Parameters to run with the program.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            initial_state: The initial state for the simulation. The form of
                this state depends on the simulation implementation.  See
                documentation of the implementing class for details.

        Returns:
            List of wavefunctions for this run, one for each possible parameter
                resolver.
        """
        # 1. checking/promotion of initial state to track qubits
        # TODO: enforce sizing on initial_state; don't want any stray/missing qubits
        if initial_state is None:
            initial_state = np.zeros((2**len(program.all_qubits()),))
            initial_state[0] = 1
        if isinstance(initial_state, np.ndarray):
            initial_qubits = range(int(np.log2(initial_state.shape[0])))
            # enforce tensor shape 2,2,2,2,...
            initial_state = initial_state.reshape((2,)*len(initial_qubits))
            state = tf.convert_to_tensor(
                value=initial_state, dtype=self._dtype)
            # fixme: util here:
            # state = TFWaveFunctionSimulatorState(state, initial_qubits)

        # TODO: gather a set of all qubits acted on in this circuit

        # FIXME: initialize buffer variable
        # buf = tf.Variable(tf.zeros_like(state, dtype=self._dtype), name='buffer')
        # Moment-wise construction of a set of matrices to apply wall
        self.ops = []
        self.indices = []
        for moment in program:
            # FIXME: empty moment?
            for op in moment.operations:
                self.ops.append(tf_gate_wrapper(op, dtype=self._dtype))
                self.indices.append([q.x for q in op.qubits])

        param_resolvers = study.to_resolvers(params)
        trial_results = []
        qubit_order = ops.QubitOrder.as_qubit_order(qubit_order)
        for param_resolver in param_resolvers:

            # prepare to iteratively construct graph down the line of ops
            state_and_buff = _StateAndBuffer(state, None) # initializer
            for op, inds in zip(self.ops, self.indices):
                # can't flush the buffer!!
                # buf = tf.assign(tf.get_variable(name='buffer'), tf.zeros_like(state, dtype=self._dtype))
                buf = tf.zeros_like(state, dtype=self._dtype)
                state_and_buff = _StateAndBuffer(state_and_buff.state, buf)
                state_and_buff = self._simulate_unitary(op, state_and_buff, inds)

            # TODO: actually make sweep usable by supplying parameters here..?
            # trial_results.append(
            #     SimulationTrialResult(
            #         params=param_resolver,
            #         measurements={},
            #         final_simulator_state=state_and_buff.state))
            trial_results.append(state_and_buff.state)

        return trial_results
