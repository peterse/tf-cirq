# tf-cirq

A basic wrapper for converting cirq.Circuit into a tensorflow graph.

### Installation
git clone git@github.com:peterse/tf-cirq

cd tf-cirq

pip install -e .

### Examples

WARNING: This supports only a restricted gateset, pending resolution of #5

Wrap a basic circuit:
```
q = cirq.LineQubit.range(2)
circuit = cirq.Circuit.from_ops(cirq.X(q[0]), cirq.CNOT(q[0], q[1]))
tfc_op = tfc.TFWaveFunctionSimulator().simulate(circuit)
with tf.Session() as sess:
  final_wavefunction = sess.run(tfc_op)
```

Parametrize a circuit by placeholders and resolve using a feeder dict
```
q = cirq.LineQubit.range(2)
theta = tf.placeholder(tf.complex64, shape=(), name="theta")
circuit = cirq.Circuit.from_ops(cirq.Rx(theta)(q[0]), cirq.CNOT(q[0], q[1]))
tfc_op = tfc.TFWaveFunctionSimulator().simulate(circuit)

feed_dict = {theta: np.pi/2}
with tf.Session() as sess:
  final_wavefunction = sess.run(tfc_op, feed_dict=feed_dict)
```

Parametrize a circuit by variables that are manipulated in another graph
TODO!
