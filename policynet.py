import tensorflow as tf
import numpy as np

from util import _kernel_img_summary, _activation_summary, _input_summary

def weight_variable_conv(name, shape):
  return tf.get_variable(name, shape=shape,
    initializer=tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG',
                                                               factor=2.0))

def weight_variable(name, shape):
  return tf.get_variable(name, shape=shape,
    initializer=tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG',
                                                               factor=1.0))

def bias_variable(name, shape, value):
  return tf.get_variable(name,
      shape=shape, initializer=tf.constant_initializer(value=value))

def conv2d(x, W, strides):
  return tf.nn.conv2d(x, W, strides=strides, padding='VALID')

class ThreadModel(object):
  """
  ThreadModel implements the policy and value networks for A3C algorithm.
  """
  def __init__(self, input_shape, num_actions, global_network, config):
    """
    Inits ThreadModel config and optionally global_network.

    Args:
      input_shape: The shape of inputs for the conv net
      num_actions: Number of actions the policy chooses from
      global_network: None for constructing a global network. A global
        ThreadModel object for the thread models.
      config: Defines tunable hyperparameters of the network
    """
    rnn_size = config['rnn_size']
    entropy_beta = config['entropy_beta']
    use_rnn = config['use_rnn']
    rms_decay = config['rms_decay']
    rms_epsilon = config['rms_epsilon']
    initial_lr = config['lr']
    initial_min_lr = config['min_lr']
    lr_decay_no_steps = config['lr_decay_no_steps']
    initial_lr_decay = (initial_lr - initial_min_lr)/lr_decay_no_steps
    num_input_frames = input_shape[2]

    self.ob = tf.placeholder(tf.float32, (None, *input_shape), name='ob')
    self.ac = tf.placeholder(tf.int32, (None, 1), name='ac')
    self.rew = tf.placeholder(tf.float32, (None, 1), name='rew')

    # Defines the conv net and optionally the recurrent net used by the policy
    with tf.variable_scope('policy_value_network') as thread_scope:
      with tf.variable_scope('conv1'):
        W_conv1 = weight_variable_conv('W', [8,8,num_input_frames,16])
        b_conv1 = bias_variable('b', [16], 0.0)

        h_conv1 = tf.nn.relu(conv2d(self.ob, W_conv1, [1,4,4,1]) +
          b_conv1, name='h')

      with tf.variable_scope('conv2'):
        W_conv2 = weight_variable_conv('W', [4,4,16,32])
        b_conv2 = bias_variable('b', [32], 0.0)

        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, [1,2,2,1]) +
          b_conv2, name='h')

      with tf.variable_scope('fc1'):
        conv2_out_size = 2592
        W_fc1 = weight_variable('W', [conv2_out_size, 256])
        b_fc1 = bias_variable('b', [256], 0.0)

        h_conv2_flat = tf.reshape(h_conv2, [-1, conv2_out_size])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1, name='h')

      nn_outputs = h_fc1
      nn_output_size = 256

      if use_rnn:
        with tf.variable_scope('rnn') as rnn_scope:
          lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)

          self.rnn_state_initial_c = tf.placeholder(tf.float32, (1, rnn_size))
          self.rnn_state_initial_h = tf.placeholder(tf.float32, (1, rnn_size))
          self.rnn_state_initial = tf.nn.rnn_cell.LSTMStateTuple(
            self.rnn_state_initial_c, self.rnn_state_initial_h)

          # Reshapes nn_outputs to have a false batch dimension for rnns.
          time_windows = tf.reshape(nn_outputs, (1, -1, nn_output_size))
          rnn_outputs, self.rnn_state_after = tf.nn.dynamic_rnn(
            lstm_cell, time_windows, initial_state=self.rnn_state_initial,
            dtype=tf.float32, time_major=False, scope=rnn_scope)
        nn_output_size = rnn_size
        nn_outputs = tf.reshape(rnn_outputs, (-1, rnn_size))

      W_softmax = weight_variable('W_softmax', [nn_output_size, num_actions])
      b_softmax = bias_variable('b_softmax', [num_actions], 0.0)

      W_linear = weight_variable('W_linear', [nn_output_size, 1])
      b_linear = bias_variable('b_linear', [1], 0.0)

    # Collect all trainable variables defined by this ThreadModel for updates
    self.trainable_variables = tf.get_collection(
      tf.GraphKeys.TRAINABLE_VARIABLES, scope=thread_scope.name)

    # Global model stores the RMSProp moving averages.
    # Thread models contain the evaluation and update logic.
    if global_network is None:
      self.gradient_mean_square = [tf.Variable(np.zeros(var.get_shape(),
        dtype=np.float32)) for var in self.trainable_variables]
      self.saver = tf.train.Saver(tf.global_variables())

      self.lr = tf.get_variable(
        'lr_decay', initializer=tf.constant(initial_lr))
      self.lr_decay = tf.get_variable(
        'lr', initializer=tf.constant(initial_lr_decay))
      self.min_lr = tf.get_variable(
        'min_lr', initializer=tf.constant(initial_min_lr))
    else:
      with tf.name_scope('outputs'):
        # The softmax output of policy and the linear output of value network
        logits = tf.nn.bias_add(tf.matmul(nn_outputs, W_softmax), b_softmax)
        self.pol_prob = tf.nn.softmax(logits)
        log_prob = tf.nn.log_softmax(logits)
        self.val = tf.nn.bias_add(tf.matmul(nn_outputs, W_linear),
          b_linear)

      with tf.name_scope('targets'):
        actions_one_hot = tf.reshape(tf.one_hot(self.ac, num_actions),
          (-1, num_actions))
        masked_log_prob = tf.reduce_sum(actions_one_hot * log_prob,
          reduction_indices=1, keep_dims=True)
        entropy = -tf.reduce_sum(log_prob * self.pol_prob, reduction_indices=1,
          keep_dims=True) * entropy_beta
        # Gradients should not pass the value node from the policy loss
        td_error_no_grad = self.rew - tf.stop_gradient(self.val)
        policy_loss = -(masked_log_prob * td_error_no_grad + entropy)
        value_loss = 0.5 * (self.rew - self.val) ** 2
        total_loss = tf.reduce_sum(policy_loss + 0.5 * value_loss)

      with tf.name_scope('gradients'):
        self.grads = tf.gradients(total_loss, self.trainable_variables)

      with tf.name_scope('updates'):
        self.updates = self.get_rms_updates(
          global_network.trainable_variables,
          self.trainable_variables,
          self.grads,
          global_network.gradient_mean_square,
          global_network.lr,
          decay=rms_decay,
          epsilon=rms_epsilon)

        self.copy_to_local = self.get_global_to_local_updates(
          global_network.trainable_variables,
          self.trainable_variables)

        lr_update = tf.cond(tf.greater(global_network.lr,
          global_network.min_lr),
          lambda: global_network.lr.assign(
            global_network.lr - global_network.lr_decay),
          lambda: global_network.lr)
        self.updates += [lr_update]

      # Get summaries for thread 0
      if 'thread_0' in thread_scope.name:
        with tf.variable_scope('summaries'):
          _kernel_img_summary(W_conv1, [8,8,1,16], 'conv1 kernels')
          _activation_summary(h_conv1, (20,20,16), 'conv1 activation')

          tf.summary.scalar('value_loss', tf.reduce_sum(value_loss))
          tf.summary.scalar('policy_loss', tf.reduce_sum(policy_loss))
          tf.summary.scalar('total_loss', tf.reduce_sum(total_loss))
          tf.summary.scalar('max_prob', tf.reduce_max(self.pol_prob))
          tf.summary.scalar('td_error', tf.reduce_sum(td_error_no_grad))
          tf.summary.scalar('learning_rate', global_network.lr)
          tf.summary.scalar('entropy', tf.reduce_sum(entropy))

          for grad in self.grads:
            summary_name = 'grad_' + '/'.join(grad.name.split('/')[-3:])
            tf.summary.scalar(summary_name, tf.reduce_mean(grad))

          for msq in global_network.gradient_mean_square:
            summary_name = 'msq_' + '/'.join(msq.name.split('/')[-3:])
            tf.summary.scalar(summary_name, tf.reduce_mean(msq))

          for var in self.trainable_variables:
            summary_name = 'var_' + '/'.join(var.name.split('/')[-3:])
            tf.summary.scalar(summary_name, tf.reduce_mean(var))

          self.summary_op = tf.summary.merge_all()

  def get_rms_updates(self, global_vars, local_vars, grads, grad_msq, lr,
                      decay=0.99, epsilon=0.01):
    """
    Compute shared RMSProp updates for local_vars.

    Args:
      global_vars: stores the global variables shared by all threads
      local_vars: thread local variables that are used for gradient computation
      grads: gradients of local_vars
      grad_msq: globally stored mean of squared gradients
      decay: the momentum parameter
      epsilon: for numerical stability
    """
    updates = []
    for Wg, grad, msq in zip(global_vars, grads, grad_msq):
      msq_update = msq.assign(decay * msq + (1. - decay) * tf.pow(grad, 2))
      # control dependecies make sure msq is updated before gradients
      with tf.control_dependencies([msq_update]):
        gradient_update = -lr * grad / tf.sqrt(msq + epsilon)
        local_to_global = Wg.assign_add(gradient_update)

      updates += [gradient_update, local_to_global, msq_update]

    return updates

  def get_global_to_local_updates(self, global_vars, local_vars):
    """
    Defines ops to copy global variables to local variables
    """
    updates = []
    for Wg, Wl in zip(global_vars, local_vars):
      global_to_local = Wl.assign(Wg)
      updates += [global_to_local]

    return updates
