from typing import Any, List, Union
import numbers

import cirq
import tensorflow as tf
import numpy as np

"""
LOCAL TODO:
    - wrapper constructs for every cirq gate
    - Check that these gate matrix definitions are self consistent; cirq
        has a weird tendency to introduce/cut global phases out...
    - unit tests for each individual instantiation
"""


class BaseTFGate(cirq.SupportsUnitary, cirq.SupportsConsistentApplyUnitary):
    """
    Main wrapper for a cirq gate. The purpose of this object is to
        1) wrap an initialized cirq gate (that may include placeholders)
        2) divert further processing away from the cirq native pipeline,
                towards tensorflow native evaluation

    Meanwhile, this should be generally compatible with cirq protocols and
    operations
    """


    def __init__(self, ):
        # TODO: can I consolidate some of my __init__'s into here?
        pass

    def _apply_unitary_(self, state):
        """Apply the action of this gate upon a state"""
        return NotImplemented

    @property
    def _has_unitary_(self):
        return True

    def _unitary_(self):
        """Overwrite of _unitary_ to block cirq.unitary protocol."""
        return self._tensor

    def _tensor_from_eigencomponents(self) -> tf.Tensor:
        """Compose a valid tensor from this gate's eigencomponents.

        This is a stand-in for cirq._unitary_ automatically composing the
        tensor from eigencomponents.

        This relies on shape and dtype specified at initialization of
        components in `_eigen_components`.
        """
        for s, (half_turns, component) in enumerate(self._eigen_components()):
            if s == 0:
                tensor = tf.zeros_like(component, dtype=self._dtype)
            e = self._exponent
            g = self._global_shift
            eig_exp = tf.exp(1j * np.pi * e * (half_turns + g))
            component = tf.scalar_mul(eig_exp, component)
            tensor = tf.add(component, tensor)
        return tensor


class WrapXPowGate(BaseTFGate):

    def __init__(self, qubit: int,
                 theta: tf.Tensor = None,
                 global_shift: float = None,
                 dtype = tf.complex64,
                 name = None):
        """Wrap a XPowGate instance.
        learnability is handled at exponent instantiation.

        """
        self._exponent = theta
        self._global_shift = global_shift
        self._qubits = [qubit.x]
        self._dtype = dtype
        # self._tensor = tf.convert_to_tensor([
        #     [tf.cos(np.pi * theta / 2), -1.0j * tf.sin(np.pi * theta / 2)],
        #     [-1.0j * tf.sin(np.pi * theta / 2), tf.cos(np.pi * theta / 2)],
        # ], name=name)
        # self._tensor = tf.scalar_mul(
        #     tf.exp(1j * (np.pi*theta/2 + 2*np.pi*global_shift)), self._tensor)
        self._tensor = self._tensor_from_eigencomponents()

        # TODO: different classing structure that will let me track qubits
        # super().__init__(tensor, [qubit], [theta])
    def _eigen_components(self):

        return [
            (tf.constant(0, dtype=self._dtype),
                tf.convert_to_tensor(np.array([[0.5, 0.5], [0.5, 0.5]]), dtype=self._dtype,
                    name="eig_XPowGate_0")), # !hep-qml FIXME
            (tf.constant(1, dtype=self._dtype),
                tf.convert_to_tensor(np.array([[0.5, -0.5], [-0.5, 0.5]]), dtype=self._dtype,
                    name="eig_XPowGate_1")), # !hep-qml FIXME
        ]


class WrapYPowGate(BaseTFGate):

    def __init__(self, qubit: int,
                 theta: tf.Tensor = None,
                 global_shift: float = None,
                 dtype = tf.complex64,
                 name = None,):
        """Wrap a YPowGate instance.
        """
        self._exponent = theta
        self._global_shift = global_shift
        self._qubits = [qubit.x]
        self._dtype = dtype
        # self._tensor = tf.convert_to_tensor([
        #     [tf.cos(np.pi * theta / 2), -tf.sin(np.pi * theta / 2)],
        #     [tf.sin(np.pi * theta / 2), tf.cos(np.pi * theta / 2)]
        # ], name=name)
        # self._tensor = tf.scalar_mul(
        #     tf.exp(1j * theta * np.pi * (global_shift + 0.5) / 2), self._tensor)
        self._tensor = self._tensor_from_eigencomponents()

    def _eigen_components(self):
        return [
            (tf.constant(0, dtype=self._dtype),
                tf.convert_to_tensor(np.array([[0.5, -0.5j], [0.5j, 0.5]]), dtype=self._dtype,
                    name="eig_YPowGate_0")), # !hep-qml FIXME
            (tf.constant(1, dtype=self._dtype),
                tf.convert_to_tensor(np.array([[0.5, 0.5j], [-0.5j, 0.5]]), dtype=self._dtype,
                    name="eig_YPowGate_1")), # !hep-qml FIXME
        ]


class WrapZPowGate(BaseTFGate):

    def __init__(self, qubit: int,
                 theta: tf.Tensor = None,
                 global_shift: float = None,
                 dtype = tf.complex64,
                 name = None):
        """Wrap a ZPowGate instance.
        """
        self._exponent = theta
        self._global_shift = global_shift
        self._qubits = [qubit.x]
        self._dtype = dtype
        # self._tensor = tf.convert_to_tensor([
        #     [1, 0],
        #     [0, tf.exp(1j * theta * np.pi * (global_shift + 0.5))]
        # ], name=name)
        self._tensor = self._tensor_from_eigencomponents()

    def _eigen_components(self):
        return [
            (tf.constant(0, dtype=self._dtype),
                tf.convert_to_tensor(np.diag([1, 0]), dtype=self._dtype,
                    name="eig_ZPowGate_0")), # !hep-qml FIXME
            (tf.constant(1, dtype=self._dtype),
                tf.convert_to_tensor(np.diag([0, 1]), dtype=self._dtype,
                    name="eig_ZPowGate_1")), # !hep-qml FIXME
        ]


class WrapHPowGate(BaseTFGate):

    def __init__(self, qubit: int,
                 theta: tf.Tensor = None,
                 global_shift: float = None,
                 dtype = tf.complex64,
                 name = None):
        """Wrap a HPowGate instance.
        """
        self._exponent = theta
        self._global_shift = global_shift
        self._qubits = [qubit.x]
        self._dtype = dtype
        # self._tensor = tf.convert_to_tensor([
        #     [tf.cos(np.pi * theta / 2) - 1j * tf.sin(np.pi * theta / 2)/np.sqrt(2), -1j * tf.sin(np.pi * theta / 2)/np.sqrt(2)],
        #     [-1j * tf.sin(np.pi * theta / 2)/np.sqrt(2), tf.cos(np.pi * theta / 2) + 1j * tf.sin(np.pi * theta / 2)/np.sqrt(2)]
        # ], name=name)
        # self._tensor = tf.scalar_mul(tf.exp(1j * theta * np.pi * (global_shift + 0.5) / 2), self._tensor)
        self._tensor = self._tensor_from_eigencomponents()

    def _eigen_components(self):
        s = np.sqrt(2)
        return [
            (tf.constant(0, dtype=self._dtype),
                tf.convert_to_tensor(np.array([
                    [3 + 2 * s, 1 + s],
                    [1 + s, 1]
                ]) / (4 + 2 * s),
                dtype=self._dtype,
                name="eig_HPowGate_0")), # !hep-qml FIXME
            (tf.constant(1, dtype=self._dtype),
                tf.convert_to_tensor(np.array([
                    [3 - 2 * s, 1 - s],
                    [1 - s, 1]]) / (4 - 2 * s),
                dtype=self._dtype,
                name="eig_HPowGate_1")), # !hep-qml FIXME
        ]


class WrapCNotPowGate(BaseTFGate):

    def __init__(self, *qubits: List[int],
                 theta: tf.Tensor = None,
                 global_shift: float = None,
                 dtype = tf.complex64,
                 name = None):
        """Wrap a CNotPowGate instance.
        """
        self._exponent = theta
        self._global_shift = global_shift
        self._qubits = [qubits[0].x, qubits[1].x]
        self._dtype = dtype

        # self._tensor = tf.convert_to_tensor([
        #     [1, 0, 0, 0],
        #     [0, 1, 0, 0],
        #     [0, 0, tf.exp(1j * theta * np.pi * (global_shift + 0.5) / 2) * tf.cos(np.pi * theta / 2),
        #         -1j * tf.exp(1j * theta * np.pi * (global_shift + 0.5) / 2) * tf.sin(np.pi * theta / 2)],
        #     [0, 0, -1j * tf.exp(1j * theta * np.pi * (global_shift + 0.5) / 2) * tf.sin(np.pi * theta / 2),
        #         tf.exp(1j * theta * np.pi * (global_shift + 0.5) / 2) * tf.cos(np.pi * theta / 2)]
        # ], name=name)
        #
        self._tensor = self._tensor_from_eigencomponents()

    def _eigen_components(self):
        return [
            (tf.constant(0, dtype=self._dtype),
                tf.convert_to_tensor(
                    np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0.5, 0.5],
                              [0, 0, 0.5, 0.5]]).reshape(2,2,2,2),
                    dtype=self._dtype,
                    name="eig_CNotPowGate_0")), # !hep-qml FIXME
            (tf.constant(1, dtype=self._dtype),
                tf.convert_to_tensor(
                    np.array([[0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0.5, -0.5],
                              [0, 0, -0.5, 0.5]]).reshape(2,2,2,2),
                    dtype=self._dtype,
                    name="eig_CNotPowGate_1")), # !hep-qml FIXME
        ]


class WrapSwapPowGate(BaseTFGate):

    def __init__(self, *qubits: List[int],
                 theta: tf.Tensor = None,
                 global_shift: float = None,
                 dtype = tf.complex64,
                 name = None,
    ):
        """Wrap a SwapPowGate instance.
        """
        self._exponent = theta
        self._global_shift = global_shift
        self._qubits = [qubits[0].x, qubits[1].x]
        self._dtype = dtype

        # self._tensor = tf.convert_to_tensor([
        #     [1, 0, 0, 0],
        #     [0, tf.exp(1j * theta * np.pi * (global_shift + 0.5) / 2) * tf.cos(np.pi * theta / 2), -1j * tf.exp(1j * theta * np.pi * (global_shift + 0.5) / 2) * tf.sin(np.pi * theta / 2), 0],
        #     [0, -1j * tf.exp(1j * theta * np.pi * (global_shift + 0.5) / 2) * tf.sin(np.pi * theta / 2), tf.exp(1j * theta * np.pi * (global_shift + 0.5) / 2) * tf.cos(np.pi * theta / 2), 0],
        #     [0, 0, 0, 1]
        # ], name=name)
        # self._tensor = tf.reshape(self._tensor, (2,2,2,2))
        self._tensor = self._tensor_from_eigencomponents()

    def _eigen_components(self):
        return [
            (tf.constant(0, dtype=self._dtype),
                tf.convert_to_tensor(
                    np.array([[1, 0,   0,   0],
                              [0, 0.5, 0.5, 0],
                              [0, 0.5, 0.5, 0],
                              [0, 0,   0,   1]]).reshape(2,2,2,2),
                    dtype=self._dtype,
                    name="eig_SWAPPowGate_0")), # !hep-qml FIXME
            (tf.constant(1, dtype=self._dtype),
                tf.convert_to_tensor(
                    np.array([[0,  0,    0,   0],
                              [0,  0.5, -0.5, 0],
                              [0, -0.5,  0.5, 0],
                              [0,  0,    0,   0]]).reshape(2,2,2,2),
                    dtype=self._dtype,
                    name="eig_SWAPPowGate_1")), # !hep-qml FIXME
        ]

class WrapZZPowGate(BaseTFGate):

    def __init__(self, *qubits: List[int],
                 theta: tf.Tensor = None,
                 global_shift: float = None,
                 dtype = tf.complex64,
                 name=None,
    ):
        """Wrap a ZZPowGate instance.
        """
        self.name = "expZZ" # !hep-qml FIXME
        self._exponent = theta
        self._global_shift = global_shift
        self._dtype = dtype
        self._tensor = self._tensor_from_eigencomponents()
        self._qubits = [qubits[0].x, qubits[1].x]

    def _eigen_components(self):
        """Overwrite EigenGate np arrays to avoid messy casting."""
        return [
            (tf.constant(0, dtype=self._dtype),
                tf.convert_to_tensor(
                    np.diag([1, 0, 0, 1]).reshape(2,2,2,2),
                    dtype=self._dtype,
                    name="eig_ZZPowGate_0".format(self.name))), # !hep-qml FIXME
            (tf.constant(1, dtype=self._dtype),
                tf.convert_to_tensor(
                    np.diag([0, 1, 1, 0]).reshape(2,2,2,2),
                    dtype=self._dtype,
                    name="eig_ZZPowGate_1".format(self.name))), # !hep-qml FIXME
        ]


ALL_WRAPPERS = {
    cirq.ops.pauli_gates._PauliX: WrapXPowGate,
    cirq.ops.pauli_gates._PauliY: WrapYPowGate,
    cirq.ops.pauli_gates._PauliZ: WrapZPowGate,
    cirq.XPowGate: WrapXPowGate,
    cirq.YPowGate: WrapYPowGate,
    cirq.ZPowGate: WrapZPowGate,
    cirq.HPowGate: WrapHPowGate,
    cirq.CNotPowGate: WrapCNotPowGate,
    cirq.SwapPowGate: WrapSwapPowGate,
    cirq.ZZPowGate: WrapZZPowGate,
    cirq.I: NotImplemented,
    cirq.S: NotImplemented,
    cirq.T: NotImplemented,

}


def _promote_and_cast(v: Any, dtype=tf.complex64) -> Union[tf.Tensor, tf.Variable]:
    """Convert numerics to Tensors of specified type."""
    if isinstance(v, (tf.Variable, tf.Tensor)):
        #TODO: typechecking to avoid unecessary casting ops
        return tf.cast(v, dtype)
    if isinstance(v, numbers.Number):
        return tf.constant(v, dtype=dtype)
    raise NotImplementedError(
        "Cannot promote type {} -> tf.Tensor".format(type(v)))


# FIXME: is inst always an eigengate..?
def tf_gate_wrapper(inst: cirq.EigenGate, dtype=tf.complex64) -> BaseTFGate:

    # WARNING: theta is promoted to theta * pi/2 before being passed into the
    # wrappers. Wrappers must expect to get this rescaled theta in their
    # tensor op chain.

    # todo: notimplemented case checking
    theta = _promote_and_cast(getattr(inst._gate, 'exponent', 1), dtype=dtype)
    # todo: update docs to reflect rad input
    global_shift = _promote_and_cast(getattr(inst._gate, '_global_shift', 0), dtype=dtype)
    wrapper = ALL_WRAPPERS.get(type(inst._gate), NotImplemented)
    if wrapper is not NotImplemented:
        return wrapper(*inst.qubits,
                       theta=theta,
                       global_shift=global_shift,
                       dtype=dtype,
                       name=None)

    raise NotImplementedError(
        "gate {} not implemented in gate wrappers".format(type(inst)))


### DO NOT DELETE
# Below is working prototype code for WrapXPowGate._apply_unitary
#     def _apply_unitary_(self, args: ApplyTFUnitaryArgs
#                         ) -> tf.Tensor:
#
#         if self._exponent != 1:
#             return None
#         zero = args.subspace_index(0)
#         one = args.subspace_index(1)
#         inds = [zero, one]
#         ref0 = args.target_tensor[one]
#         ref1 = args.target_tensor[zero]
#         refs = [ref0, ref1]
#         x = args.available_buffer
#         with tf.control_dependencies([x[inds[i]].assign(refs[i]) for i in range(2)]):
#             x = tf.identity(x)
#
#         p = 1j**(2 * self._exponent * self._global_shift)
#         if p != 1:
#             x = tf.scalar_mul(p, x)
#         return x
