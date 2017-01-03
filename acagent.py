import gym
import tensorflow as tf
import numpy as np
import threading
from collections import deque
from PIL import Image, ImageOps

from util import _kernel_img_summary, _activation_summary

def categorical_sample(prob):
  """
  Sample from categorical distribution,
  specified by a vector of class probabilities
  """
  prob = np.asarray(prob)
  csprob = np.cumsum(prob)
  return (csprob > np.random.rand()).argmax()

def weight_variable(name, shape, stddev):
  return tf.get_variable(name,
      shape=shape, initializer=tf.truncated_normal_initializer(stddev=stddev))

def bias_variable(name, shape, value):
  return tf.get_variable(name,
      shape=shape, initializer=tf.constant_initializer(value=value))

def conv2d(x, W, strides):
  return tf.nn.conv2d(x, W, strides=strides, padding='VALID')

def resize_observation(observation, shape, centering):
  img = Image.fromarray(observation)
  img = ImageOps.fit(img, shape[:2], centering=centering)
  img = img.convert('L')
  return np.reshape(np.array(img), (1, *shape)) * 1./255.

class ThreadModel(object):
  def __init__(self, input_shape, output_size, global_network, config):
    W0_size = config['hidden_1']
    rnn_size = config['rnn_size']
    entropy_beta = config['entropy_beta']
    grad_norm_clip_val = config['grad_norm_clip_val']
    use_rnn = config['use_rnn']
    rms_decay = config['rms_decay']

    self.ob = tf.placeholder(tf.float32, (None, *input_shape), name='ob')
    self.ac = tf.placeholder(tf.int32, (None, 1), name='ac')
    self.rew = tf.placeholder(tf.float32, (None, 1), name='rew')
    self.lr = tf.placeholder(tf.float32, name='lr')

    with tf.variable_scope('policy_value_network') as thread_scope:
      with tf.variable_scope('conv1'):
        stddev = 2./(8*8*input_shape[2])
        W_conv1 = weight_variable('W', [8, 8, input_shape[2], 16], stddev)
        b_conv1 = bias_variable('b', [16], 0.1)

        h_conv1 = tf.nn.relu(conv2d(self.ob, W_conv1, [1,4,4,1]) +
            b_conv1, name='h')

      with tf.variable_scope('conv2'):
        stddev = 2./(4*4*16)
        W_conv2 = weight_variable('W', [4,4,16,32], stddev)
        b_conv2 = bias_variable('b', [32], 0.01)

        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, [1,2,2,1]) +
            b_conv2, name='h')

      with tf.variable_scope('fc1'):
        conv2_out_size = 2592
        stddev = 2./(conv2_out_size)
        W_fc1 = weight_variable('W', [conv2_out_size, 256], stddev)
        b_fc1 = bias_variable('b', [256], 0.001)

        h_conv2_flat = tf.reshape(h_conv2, [-1, conv2_out_size])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1, name='h')

      nn_outputs = h_fc1
      nn_output_size = 256

      if use_rnn:
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)

        self.rnn_state_initial_c = tf.placeholder(tf.float32, (1, rnn_size))
        self.rnn_state_initial_h = tf.placeholder(tf.float32, (1, rnn_size))
        self.rnn_state_initial = tf.nn.rnn_cell.LSTMStateTuple(
          self.rnn_state_initial_c, self.rnn_state_initial_h)

        time_windows = tf.reshape(nn_outputs, (1, -1, nn_output_size))
        rnn_outputs, self.rnn_state_after = tf.nn.dynamic_rnn(
          lstm_cell, time_windows, initial_state=self.rnn_state_initial,
          dtype=tf.float32)
        nn_output_size = rnn_size
        nn_outputs = tf.reshape(rnn_outputs, (-1, rnn_size))

      stddev = 1./nn_output_size
      W_softmax = weight_variable('W_softmax', [nn_output_size, output_size], stddev)
      b_softmax = bias_variable('b_softmax', [output_size], 0.0)

      W_linear = weight_variable('W_linear', [nn_output_size, 1], stddev)
      b_linear = bias_variable('b_linear', [1], 0.0)

    # Variable collections for update computations
    self.trainable_variables = tf.get_collection(
      tf.GraphKeys.TRAINABLE_VARIABLES, scope=thread_scope.name)

    # Global model stores the RMSProp moving averages.
    # Thread models contain the evaluation and update logic.
    if global_network is None:
      self.gradient_mean_square = [tf.Variable(np.zeros(var.get_shape(),
        dtype=np.float32)) for var in self.trainable_variables]
      self.saver = tf.train.Saver(tf.all_variables())
    else:
      with tf.name_scope('outputs'):
        self.pol_prob = tf.nn.softmax(tf.nn.bias_add(tf.matmul(
          nn_outputs, W_softmax), b_softmax))
        self.val = tf.nn.bias_add(tf.matmul(nn_outputs, W_linear),
          b_linear)

      with tf.name_scope('targets'):
        actions_one_hot = tf.reshape(tf.one_hot(self.ac, output_size),
          (-1, output_size))
        masked_prob = tf.reduce_sum(actions_one_hot * self.pol_prob,
          reduction_indices=1, keep_dims=True)
        log_masked_prob = tf.log(tf.clip_by_value(masked_prob, 1.e-22, 1.0))
        td_error = self.rew - self.val
        entropy = -tf.reduce_sum(masked_prob * log_masked_prob, reduction_indices=1, keep_dims=True) * entropy_beta
        policy_loss = -(log_masked_prob * td_error + entropy)
        value_loss = td_error ** 2. / 2.
        #total_loss = tf.reduce_sum(policy_loss + 0.5 * value_loss)
        total_loss = policy_loss + 0.5 * value_loss

      with tf.name_scope('gradients'):
        self.grads = tf.gradients(total_loss, self.trainable_variables)

      with tf.name_scope('updates'):
        self.updates = self.get_rms_updates(global_network.trainable_variables,
          self.trainable_variables, self.grads,
          global_network.gradient_mean_square, decay=rms_decay)

      # Get summaries for thread 0
      if 'thread_0' in thread_scope.name:
        _kernel_img_summary(W_conv1, [8,8,1,16], 'conv1 kernels')
        _activation_summary(h_conv1, (20,20,16), 'conv1 activation')

        tf.scalar_summary('value_loss', tf.reduce_mean(value_loss))
        tf.scalar_summary('policy_loss', tf.reduce_mean(policy_loss))
        tf.scalar_summary('total_loss', tf.reduce_mean(total_loss))
        tf.scalar_summary('entropy', tf.reduce_mean(entropy))
        tf.scalar_summary('max_prob', tf.reduce_max(self.pol_prob))
        tf.scalar_summary('td_error', tf.reduce_mean(td_error))

        for grad in self.grads:
          summary_name = 'grad_' + '/'.join(grad.name.split('/')[-3:])
          tf.scalar_summary(summary_name, tf.reduce_mean(grad))

        for msq in global_network.gradient_mean_square:
          summary_name = 'msq_' + '/'.join(msq.name.split('/')[-3:])
          tf.scalar_summary(summary_name, tf.reduce_mean(msq))

        for var in self.trainable_variables:
          summary_name = 'var_' + '/'.join(var.name.split('/')[-3:])
          tf.scalar_summary(summary_name, tf.reduce_mean(var))

        self.summary_op = tf.merge_all_summaries()

  def get_rms_updates(self, global_vars, local_vars, grads, grad_msq,
                  decay=0.99, epsilon=1e-10, grad_norm_clip=50.):
    updates = []
    for Wg, grad, msq in zip(global_vars, grads, grad_msq):
      grad = tf.clip_by_norm(grad, grad_norm_clip)
      msq_update = msq.assign(decay * msq + (1. - decay) * tf.pow(grad, 2))
      with tf.control_dependencies([msq_update]):
        gradient_update = -self.lr * grad / tf.sqrt(msq + epsilon)
        l_to_g = Wg.assign_add(gradient_update)

      updates += [gradient_update, l_to_g, msq_update]

    with tf.control_dependencies(updates):
      for Wg, Wl in zip(global_vars, local_vars):
        g_to_l = Wl.assign(Wg)
        updates += [g_to_l]

    return updates

global_model = None
thr_models = []
envs = None
sess = None

class ACAgent(object):
  def __init__(self, **usercfg):
    self.config = dict(
        n_iter = 100,
        t_max = 10,
        gamma = 0.98,
        lr = 0.01,
        min_lr = 0.002,
        lr_decay_no_steps = 10000,
        hidden_1 = 20,
        rnn_size = 10,
        use_rnn = True,
        num_threads = 1,
        env_name = '',
        entropy_beta = 0.01,
        grad_norm_clip_val = 100.,
        rms_decay = 0.99,
      )

    self.config.update(usercfg)

    global envs, thr_models, global_model
    envs = [gym.make(self.config['env_name']) for _ in
      range(self.config['num_threads'])]
    input_shape = (84,84,1)
    action_num = envs[0].action_space.n
    self.use_rnn = self.config['use_rnn']

    with tf.variable_scope('global'):
      global_model = ThreadModel(input_shape, action_num, None, self.config)

    for thr in range(self.config['num_threads']):
      with tf.variable_scope('thread_{}'.format(thr)):
        thr_model = ThreadModel(input_shape, action_num, global_model,
          self.config)
        thr_models.append(thr_model)

    global sess
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

  def reset_rnn_state(self, rnn_size):
    # TODO: This is wrong. rnn_states should be per thread, not shared.
    self.running_rnn_state = tf.nn.rnn_cell.LSTMStateTuple(
      np.zeros([1, rnn_size]), np.zeros([1, rnn_size]))
    self.training_rnn_state = tf.nn.rnn_cell.LSTMStateTuple(
      np.zeros([1, rnn_size]), np.zeros([1, rnn_size]))

  def act(self, ob, model):
    global sess
    if self.use_rnn:
      prob, self.running_rnn_state = sess.run(
        [model.pol_prob, model.rnn_state_after],
        {model.ob:ob,
         model.rnn_state_initial_c:self.running_rnn_state[0],
         model.rnn_state_initial_h:self.running_rnn_state[1],
        })
    else:
      prob = sess.run(model.pol_prob, {model.ob:ob})
    action = categorical_sample(prob)
    return action


  def learning_thread(self, thread_id):
    global envs, thr_models, sess
    t_max = self.config['t_max']
    lr = self.config['lr']
    min_lr = self.config['min_lr']
    lr_decay_no_steps = self.config['lr_decay_no_steps']
    lr_decay_step = (lr - min_lr)/lr_decay_no_steps
    self.use_rnn = self.config['use_rnn']
    rnn_size = self.config['rnn_size']

    env = envs[thread_id]
    model = thr_models[thread_id]

    ob_shape = (84,84,1)
    crop_centering = (0.5, 0.7)
    obs = np.zeros((t_max, *ob_shape))
    acts = np.zeros((t_max))
    rews = np.zeros((t_max))

    ep_rews = 0
    ep_count = 0
    rews_acc = deque(maxlen=100)

    if thread_id == 0:
      self.summary_writer = tf.train.SummaryWriter('train', sess.graph)

    if self.use_rnn:
      self.reset_rnn_state(rnn_size)

    t = 0
    done = True
    for iteration in range(self.config['n_iter']):
      if done:
        if thread_id == 0 and ep_count % 100 == 0:
          global_model.saver.save(sess, 'train/model.ckpt', global_step=iteration)
        if self.use_rnn:
          self.reset_rnn_state(rnn_size)
        done = False
        ob = env.reset()
        ob = resize_observation(ob, ob_shape, crop_centering)
        rews_acc.append(ep_rews)
        ep_count += 1
        print('Thread: {} Episode: {} Rews: {} RunningAvgRew: '
              '{:.1f} lr: {}'.format(thread_id, ep_count, ep_rews,
              np.mean(rews_acc), lr))
        ep_rews = 0
      else:
        obs[t] = ob
        action = self.act(ob, model)
        ob, rew, done, _ = env.step(action)
        ob = resize_observation(ob, ob_shape, crop_centering)
        acts[t] = action
        rews[t] = rew if not done else 0
        t += 1
        ep_rews += rew

      if done or t == t_max:
        obs_arr = np.reshape(obs[:t], (t, *ob_shape))
        acts_arr = np.reshape(acts[:t], (t, 1))

        if done:
          R = 0
        else:
          if self.use_rnn:
            R = sess.run(
              model.val,
              {model.ob:ob,
               model.rnn_state_initial_c:self.running_rnn_state[0],
               model.rnn_state_initial_h:self.running_rnn_state[1],
              })
          else:
            R = sess.run(model.val, {model.ob:ob})

        R_arr = np.zeros((t, 1))
        for i in reversed(range(t)):
          R = rews[i] + self.config['gamma'] * R
          R_arr[i, 0] = R

        if self.use_rnn:
          _, self.training_rnn_state = sess.run(
            [model.updates, model.rnn_state_after],
            {model.ob:obs_arr,
             model.ac:acts_arr,
             model.rew:R_arr,
             model.lr:lr,
             model.rnn_state_initial_c:self.training_rnn_state[0],
             model.rnn_state_initial_h:self.training_rnn_state[1],
            })
        else:
          _ = sess.run(
            [model.updates],
            {model.ob:obs_arr,
             model.ac:acts_arr,
             model.rew:R_arr,
             model.lr:lr,
            })

        if thread_id == 0 and ep_count % 10 == 0 and done == True:
          summary_str = sess.run(model.summary_op,
            {model.ob:obs_arr,
             model.ac:acts_arr,
             model.rew:R_arr})
          self.summary_writer.add_summary(summary_str, iteration)
          #print(dbg)

  #      if thread_id == 0:
  #        for var, name in zip(dbg, ['pol_prob', 'masked_prob', 'td_error', 'entropy', 'policy_loss', 'value_loss','total_loss']):
  #          print('================')
  #          print(name)
  #          print(var)
  #          print('================')

        lr = lr - lr_decay_step if lr > min_lr else lr

        #import pdb; pdb.set_trace()

        acts[:] = 0
        rews[:] = 0
        obs[:] = 0
        t = 0

  def learn(self):
    ths = [threading.Thread(target=self.learning_thread, args=(i,)) for i in
      range(self.config['num_threads'])]

    for t in ths:
      t.start()

def main():
  agent = ACAgent(gamma=0.99, n_iter=1000000, num_threads=8, t_max=5,
    lr=0.001, min_lr=0.00001, lr_decay_no_steps=1000000, hidden_1=100,
    rnn_size=100, env_name='Breakout-v0', use_rnn=False, entropy_beta=0.01,
    rms_decay=0.99)
  agent.learn()

if __name__ == "__main__":
  main()
