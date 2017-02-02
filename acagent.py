import gym
import tensorflow as tf
import numpy as np
import threading
from collections import deque
from PIL import Image, ImageOps

from policynet import ThreadModel

def resize_observation(observation, shape, centering):
  img = Image.fromarray(observation)
  img = ImageOps.fit(img, shape[:2], centering=centering)
  img = img.convert('L')
  return np.reshape(np.array(img), (1, *shape)) * 1./255.

def categorical_sample(prob):
  prob = np.asarray(prob)
  csprob = np.cumsum(prob)
  return (csprob > np.random.rand()).argmax()

def act(ob, model, session, running_rnn_state):
  if running_rnn_state is not None:
    prob, running_rnn_state = session.run(
      [model.pol_prob, model.rnn_state_after],
      {model.ob:ob,
       model.rnn_state_initial_c:running_rnn_state[0],
       model.rnn_state_initial_h:running_rnn_state[1],
      })
  else:
    prob = session.run(model.pol_prob, {model.ob:ob})
  action = categorical_sample(prob)
  return action, running_rnn_state

def bootstrap_return(session, model, observation, running_rnn_state):
  if running_rnn_state is not None:
    R = session.run(
      model.val,
      {model.ob:observation,
       model.rnn_state_initial_c:running_rnn_state[0],
       model.rnn_state_initial_h:running_rnn_state[1],
      })
  else:
    R = session.run(model.val, {model.ob:observation})
  return R

def run_updates(session, model, obs_arr, acts_arr, R_arr, training_rnn_state):
  if training_rnn_state is not None:
    _, training_rnn_state = session.run(
      [model.updates, model.rnn_state_after],
      {model.ob:obs_arr,
       model.ac:acts_arr,
       model.rew:R_arr,
       model.rnn_state_initial_c:training_rnn_state[0],
       model.rnn_state_initial_h:training_rnn_state[1],
      })
  else:
    _ = session.run(
      [model.updates],
      {model.ob:obs_arr,
       model.ac:acts_arr,
       model.rew:R_arr,
      })
  return training_rnn_state

def reset_rnn_state(rnn_size):
  running_rnn_state = tf.nn.rnn_cell.LSTMStateTuple(
    np.zeros([1, rnn_size]), np.zeros([1, rnn_size]))
  training_rnn_state = tf.nn.rnn_cell.LSTMStateTuple(
    np.zeros([1, rnn_size]), np.zeros([1, rnn_size]))
  return running_rnn_state, training_rnn_state

def write_summaries(summary_writer, session, model, obs_arr, acts_arr, R_arr,
                    iteration, training_rnn_state):
  if training_rnn_state is not None:
    summary_str = session.run(model.summary_op,
      {model.ob:obs_arr,
       model.ac:acts_arr,
       model.rew:R_arr,
       model.rnn_state_initial_c:training_rnn_state[0],
       model.rnn_state_initial_h:training_rnn_state[1],
      })
  else:
    summary_str = session.run(model.summary_op,
      {model.ob:obs_arr,
       model.ac:acts_arr,
       model.rew:R_arr,
      })
  summary_writer.add_summary(summary_str, iteration)

def discounted_returns(rews, gamma, R, t):
  R_arr = np.zeros((t, 1))
  for i in reversed(range(t)):
    R = rews[i] + gamma * R
    R_arr[i, 0] = R
  return R_arr

def new_random_game(env, random_starts, action_size):
  ob = env.reset()
  no_rnd = np.random.randint(0, random_starts)
  for i in range(no_rnd):
    ob, _, _, _ = env.step(np.random.randint(action_size))
  return ob

def save_observation(obs, ob, iteration, obs_mem_size, ob_shape, crop_centering):
  obs_current_index = iteration % obs_mem_size
  ob = resize_observation(ob, ob_shape, crop_centering)
  obs[obs_current_index] = ob

def get_obs_window(obs, iteration, obs_mem_size, window_size):
  obs_current_index = iteration % obs_mem_size
  obs_start_index = obs_current_index - window_size
  obs_window_indices = np.arange(obs_start_index, obs_current_index) + 1
  ob_w = obs[obs_window_indices]
  ob_w = np.transpose(ob_w, (3,1,2,0))
  return ob_w

def get_training_window(obs, iteration, obs_mem_size, t, window_size):
  obs_current_index = iteration % obs_mem_size
  obs_start_index = obs_current_index - t
  obs_indices = [np.arange(i - window_size + 1, i + 1) for i in
    np.arange(obs_start_index, obs_current_index + 1)]
  obs_indices = np.reshape(obs_indices, (t + 1, window_size)) % obs_mem_size
  obs_arr = obs[obs_indices]
  obs_arr = np.transpose(np.reshape(obs_arr, (t + 1, window_size, 84, 84)), (0,2,3,1))
  return obs_arr

def learning_thread(thread_id, config, session, model, global_model, env):
  t_max = config['t_max']
  use_rnn = config['use_rnn']
  rnn_size = config['rnn_size']
  window_size = config['window_size']
  gamma = config['gamma']
  random_starts = config['random_starts']
  num_of_iterations = config['n_iter']

  ob_shape = (84,84,1)
  crop_centering = (0.5, 0.7)

  obs_mem_size = t_max + window_size
  obs = np.zeros((obs_mem_size, *ob_shape))
  acts = np.zeros((t_max))
  rews = np.zeros((t_max))

  # variables for accounting
  ep_rews = 0
  ep_count = 0
  rews_acc = deque(maxlen=100)

  running_rnn_state = None
  training_rnn_state = None

  if thread_id == 0:
    summary_writer = tf.summary.FileWriter('train', session.graph)

  if use_rnn:
    running_rnn_state, training_rnn_state = reset_rnn_state(rnn_size)

  t_start = 0
  done = False
  ob = new_random_game(env, random_starts, env.action_space.n)
  obs[:] = resize_observation(ob, ob_shape, crop_centering)

  # Training loop
  for iteration in range(num_of_iterations):
    if t_start == iteration:
      session.run(model.copy_to_local)

    save_observation(obs, ob, iteration, obs_mem_size, ob_shape, crop_centering)
    observation_window = get_obs_window(obs, iteration, obs_mem_size, window_size)
    action, running_rnn_state = act(observation_window, model, session,
      running_rnn_state)
    ob, rew, done, _ = env.step(action)

    t = iteration - t_start
    acts[t] = action
    rews[t] = rew if not done else 0
    ep_rews += rew

    if done or iteration - t_start == t_max - 1:
      obs_arr = get_training_window(obs, iteration, obs_mem_size, t,
        window_size)
      acts_arr = np.reshape(acts[:t + 1], (t + 1, 1))

      if done:
        R = 0
      else:
        R = bootstrap_return(session, model, observation_window,
          running_rnn_state)

      R_arr = discounted_returns(rews, gamma, R, t + 1)

      training_rnn_state = run_updates(session, model, obs_arr, acts_arr,
        R_arr, training_rnn_state)

      if thread_id == 0 and iteration % 100 == 0:
        write_summaries(summary_writer, session, model, obs_arr, acts_arr,
          R_arr, iteration, training_rnn_state)

      if thread_id == 0 and iteration % 10000 == 0:
        global_model.saver.save(session, 'train/model.ckpt',
          global_step=iteration)

      acts[:] = 0
      rews[:] = 0
      t_start = iteration + 1

    if done:
      done = False
      ob = new_random_game(env, random_starts, env.action_space.n)
      obs[:] = resize_observation(ob, ob_shape, crop_centering)

      if use_rnn:
        running_rnn_state, training_rnn_state = reset_rnn_state(rnn_size)

      rews_acc.append(ep_rews)
      ep_count += 1
      if thread_id == 0:
        print('Thread: {} Episode: {} Rews: {} RunningAvgRew: '
              '{:.1f}'.format(thread_id, ep_count, ep_rews,
              np.mean(rews_acc)))
      ep_rews = 0

class ACAgent(object):
  def __init__(self, **usercfg):
    self.config = dict(
        n_iter = 100,
        t_max = 10,
        gamma = 0.98,
        lr = 0.01,
        min_lr = 0.002,
        lr_decay_no_steps = 10000,
        rnn_size = 10,
        use_rnn = True,
        num_threads = 1,
        window_size = 4,
        env_name = '',
        entropy_beta = 0.01,
        rms_decay = 0.99,
        rms_epsilon = 0.01,
        random_starts=30,
        load_path=None,
      )

    self.config.update(usercfg)

    self.envs = [gym.make(self.config['env_name']) for _ in
      range(self.config['num_threads'])]

    window_size = self.config['window_size']

    self.input_shape = (84,84,window_size)
    self.action_num = self.envs[0].action_space.n
    self.use_rnn = self.config['use_rnn']

    with tf.variable_scope('global'), tf.device('/cpu:0'):
      self.global_model = ThreadModel(self.input_shape, self.action_num, None,
        self.config)

    self.thread_models = []
    for thr in range(self.config['num_threads']):
      with tf.variable_scope('thread_{}'.format(thr)):
        thr_model = ThreadModel(self.input_shape, self.action_num,
          self.global_model, self.config)
        self.thread_models.append(thr_model)

    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())
    if self.config['load_path'] is not None:
      self.global_model.saver.restore(self.session, self.config['load_path'])

  def learn(self):
    threads = []
    for i in range(self.config['num_threads']):
      threads.append(threading.Thread(target=learning_thread, args=(
        i, self.config, self.session, self.thread_models[i], self.global_model,
        self.envs[i])))

    self.session.graph.finalize()

    for thread in threads:
      thread.start()

def main():
  agent = ACAgent(gamma=0.99, n_iter=10000000, num_threads=8, t_max=5, lr=0.001,
    min_lr=0.000001, lr_decay_no_steps=10000000, rnn_size=256, window_size=1,
    env_name='Breakout-v0', use_rnn=True, entropy_beta=0.01, rms_decay=0.99)
  agent.learn()

if __name__ == "__main__":
  main()
