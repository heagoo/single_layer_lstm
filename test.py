import tensorflow as tf
import numpy as np

with tf.Graph().as_default():
  #cell = tf.nn.rnn_cell.LSTMCell(1024, 1.0, state_is_tuple = True, reuse = tf.get_variable_scope().reuse)
  cell = tf.contrib.rnn.BasicLSTMCell(1024, 1.0, state_is_tuple = True, reuse = tf.get_variable_scope().reuse)
  cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(1)], state_is_tuple = True)


  lstm_input = tf.placeholder(tf.float32, [None, 4, 1024])
  state = cell.zero_state(1, tf.float32)
  outputs = []
  with tf.variable_scope('RNN', initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=100)):

    for time_step in range(4):
      output, state = cell(lstm_input[:, time_step, :], state)

    print(output)
    print(state)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    values = tf.trainable_variables()
    for v in values:
      print(v)
      print("name: " + v.name)
      print("shape: " + str(v.shape))
  
    v = sess.run('RNN/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0')
    v = np.transpose(v)
    with open('weights.bin', 'wb') as f:
      v.tofile(f)
  
    v = sess.run('RNN/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0')
    v = np.transpose(v)
    with open('bias.bin', 'wb') as f:
      v.tofile(f)
  
 
    x = np.ones((1, 4, 1024))
    out, s = sess.run([output, state], feed_dict={lstm_input: x})
    print("Output:")
    print(out)
    print("State:")
    print(s)

    # Save the output
    #with open('out.bin', 'wb') as f:
    #  out.tofile(f)
