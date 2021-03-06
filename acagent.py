import gym
import tensorflow as tf
import numpy as np
import threading
from collections import deque
from PIL import Image, ImageOps

from policynet import ThreadModel

def resize_observation(observation, shape, centering):
  """
  Resizes the observation from the environment. Normalizes values to (0,1)

  Args:
    observation: single observation from the environment
    shape: input shape for the network. for example (84,84)
    centering: the input is cropped into square. centering defines the crop
      ratio. for example (0.5, 0.7)
  """
  img_shape = shape[:2]
  img = Image.fromarray(observation)
  img = ImageOps.fit(img, img_shape, centering=centering)
  img = img.convert('L')
  return np.reshape(np.array(img), (1, *shape)) * 1./255.

def categorical_sample(prob):
  """
  Sample from a categorical distribution specified by a vector of probabilities
  """
  prob = np.asarray(prob)
  csprob = np.cumsum(prob)
  return (csprob > np.random.rand()).argmax()

def act(observation, model, session, running_rnn_state, evaluate):
  """
  Given an observation and a model return an action sampled from the policy.

  Args:
    observation: observation or a stack of observations from the environment
    model: ThreadModel object for the current thread
    session: TensorFlow session
    running_rnn_state: None for ff version of the algorithm, LSTMStateTuple for
      the LSTM version
  """
  if running_rnn_state is not None:
    prob, running_rnn_state = session.run(
      [model.pol_prob, model.rnn_state_after],
      {model.ob:observation,
       model.rnn_state_initial_c:running_rnn_state[0],
       model.rnn_state_initial_h:running_rnn_state[1],
      })
  else:
    prob = session.run(model.pol_prob, {model.ob:observation})
  if not evaluate:
    action = categorical_sample(prob)
  else:
    action = np.argmax(prob)
  return action, running_rnn_state

def bootstrap_return(observation, model, session, running_rnn_state):
  """
  Given an observation and a model return the value evaluated at observation.

  Args:
    observation: observation or a stack of observations from the environment
    model: ThreadModel object for the current thread
    session: TensorFlow session
    running_rnn_state: None for ff version of the algorithm, LSTMStateTuple for
      the LSTM version
  """
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

def run_updates(session, model, training_observations, training_actions,
    discounted_returns, training_rnn_state):
  """
  Run one update step for the model.

  Args:
    session: TensorFlow session
    model: ThreadModel object for the current thread
    training_observations: an array of consecutive observations (or windows of
      observation) from the environment
    training_actions: an array of actions selected
    discounted_returns: an array of discounted returns
    training_rnn_state: None for ff version of the algorithm, LSTMStateTuple
      for the LSTM version
  """
  if training_rnn_state is not None:
    _, training_rnn_state = session.run(
      [model.updates, model.rnn_state_after],
      {model.ob:training_observations,
       model.ac:training_actions,
       model.rew:discounted_returns,
       model.rnn_state_initial_c:training_rnn_state[0],
       model.rnn_state_initial_h:training_rnn_state[1],
      })
  else:
    _ = session.run(
      [model.updates],
      {model.ob:training_observations,
       model.ac:training_actions,
       model.rew:discounted_returns,
      })
  return training_rnn_state

def reset_rnn_state(rnn_size):
  """
  Returns an all-zeros LSTMStateTuple for the given rnn_size
  """
  running_rnn_state = tf.nn.rnn_cell.LSTMStateTuple(
    np.zeros([1, rnn_size]), np.zeros([1, rnn_size]))
  training_rnn_state = tf.nn.rnn_cell.LSTMStateTuple(
    np.zeros([1, rnn_size]), np.zeros([1, rnn_size]))
  return running_rnn_state, training_rnn_state

def write_summaries(summary_writer, session, model, training_observations,
    training_actions, discounted_returns, iteration, training_rnn_state):
  """
  Evaluate the model and write summaries

  Args:
    summary_writes: TensorFlow summary writer
    session: TensorFlow session
    model: ThreadModel object for the current thread
    training_observations: an array of consecutive observations (or windows of
      observation) from the environment
    training_actions: an array of actions selected
    discounted_returns: an array of discounted returns
    iteration: number of global steps taken.
    training_rnn_state: None for ff version of the algorithm, LSTMStateTuple
      for the LSTM version
  """
  if training_rnn_state is not None:
    summary_str = session.run(model.summary_op,
      {model.ob:training_observations,
       model.ac:training_actions,
       model.rew:discounted_returns,
       model.rnn_state_initial_c:training_rnn_state[0],
       model.rnn_state_initial_h:training_rnn_state[1],
      })
  else:
    summary_str = session.run(model.summary_op,
      {model.ob:training_observations,
       model.ac:training_actions,
       model.rew:discounted_returns,
      })
  summary_writer.add_summary(summary_str, iteration)

def discount_returns(rews, gamma, R, t):
  """
  Compute exponential discounts on the returns

  Args:
    rews: an array of rewards from the environment
    gamma: the discounting factor
    R: bootstrap return
    t: number of timesteps to compute
  """
  discounted_returns = np.zeros((t, 1))
  for i in reversed(range(t)):
    R = rews[i] + gamma * R
    discounted_returns[i, 0] = R
  return discounted_returns

def new_random_game(env, random_starts, action_size):
  """
  Step the environment for a random number of steps

  Args:
    env: OpenAI Gym environment
    random_starts: maximum number of random steps to take
    action_size: number of actions available
  """
  ob = env.reset()
  no_rnd = np.random.randint(0, random_starts)
  for i in range(no_rnd):
    ob, _, _, _ = env.step(np.random.randint(action_size))
  return ob

def save_observation(obs, ob, iteration, obs_mem_size, ob_shape,
                     crop_centering):
  """
  Saves an observation from the environment into the array obs

  Args:
    obs: a circular buffer stored as numpy ndarray for storing observations
    ob: the current observation
    iteration: local iteration number, used for accessing the buffer
    obs_mem_size: size of obs array, used for accessing the buffer
    ob_shape: the observation shape used by the model
    crop_centering: a tuple defining the horizontal and vertical centering of
      the crop for the observations
  """
  obs_current_index = iteration % obs_mem_size
  ob = resize_observation(ob, ob_shape, crop_centering)
  obs[obs_current_index] = ob

def get_obs_window(obs, iteration, obs_mem_size, window_size):
  """
  Gets the current observation window from obs

  Args:
    obs: a circular buffer stored as numpy ndarray for storing observations
    iteration: local iteration number, used for accessing the buffer
    obs_mem_size: size of obs array, used for accessing the buffer
    window_size: how many observations go into one window
  """
  obs_current_index = iteration % obs_mem_size
  obs_start_index = obs_current_index - window_size
  obs_window_indices = np.arange(obs_start_index, obs_current_index) + 1
  ob_w = obs[obs_window_indices]
  ob_w = np.transpose(ob_w, (1,2,0))[np.newaxis]
  return ob_w

def get_training_window(obs, iteration, obs_mem_size, t, window_size,
    observation_shape):
  """
  Gets training window from obs

  Args:
    obs: a circular buffer stored as numpy ndarray for storing observations
    iteration: local iteration number, used for accessing the buffer
    obs_mem_size: size of obs array, used for accessing the buffer
    t: how many timesteps to include
    window_size: how many observations in one window
    observation_shape: for example (84,84)
  """
  obs_current_index = iteration % obs_mem_size
  obs_start_index = obs_current_index - t
  obs_indices = [np.arange(i - window_size + 1, i + 1) for i in
    np.arange(obs_start_index, obs_current_index + 1)]
  obs_indices = np.reshape(obs_indices, (t + 1, window_size)) % obs_mem_size
  training_observations = obs[obs_indices]
  training_observations = np.transpose(np.reshape(
    training_observations,
    (t + 1, window_size, *observation_shape)),
    (0,2,3,1))
  return training_observations

def learning_thread(thread_id, config, session, model, global_model, env):
  """
  Executes the training loop. Called by each of the training threads.

  Args:
    thread_id: integer from 0 to num threads
    config: config for the training loop
    session: TensorFlow session
    model: The ThreadModel object for the current thread
    global_model: The global ThreadModel object
    env: OpenAI Gym environment specific to this thread
  """
  t_max = config['t_max']
  use_rnn = config['use_rnn']
  rnn_size = config['rnn_size']
  window_size = config['window_size']
  gamma = config['gamma']
  random_starts = config['random_starts']
  num_of_iterations = config['n_iter']
  train_dir = config['train_dir']
  evaluate = config['evaluate']
  render = config['render']

  observation_shape = config['ob_shape']
  crop_centering = config['crop_centering']

  obs_mem_size = t_max + window_size

  # A circular buffer that stores obs_mem_size observations accessed through
  # save_observation, get_obs_window and get_training_window functions,
  # which handle the indexing.
  observation_buffer = np.zeros((obs_mem_size, *observation_shape))
  actions = np.zeros((t_max))
  rewards = np.zeros((t_max))

  # variables for accounting
  episode_rewards = 0
  episode_count = 0
  previous_rewards = deque(maxlen=100)

  running_rnn_state = None
  training_rnn_state = None

  if thread_id == 0:
    summary_writer = tf.summary.FileWriter(train_dir, session.graph)

  if use_rnn:
    running_rnn_state, training_rnn_state = reset_rnn_state(rnn_size)

  t_start = 0
  done = False
  ob = new_random_game(env, random_starts, env.action_space.n)
  observation_buffer[:] = resize_observation(ob, observation_shape,
    crop_centering)

  # Training loop
  for iteration in range(num_of_iterations):
    if t_start == iteration:
      session.run(model.copy_to_local)

    save_observation(observation_buffer, ob, iteration, obs_mem_size,
      observation_shape, crop_centering)
    observation_window = get_obs_window(observation_buffer, iteration,
      obs_mem_size, window_size)
    action, running_rnn_state = act(observation_window, model, session,
      running_rnn_state, evaluate)
    ob, rew, done, _ = env.step(action)

    if render:
      env.render()

    t = iteration - t_start
    actions[t] = action
    rewards[t] = rew if not done else 0
    episode_rewards += rew

    # Training is triggered after each episode or after t_max steps
    if done or iteration - t_start == t_max - 1:
      training_observations = get_training_window(observation_buffer,
        iteration, obs_mem_size, t, window_size, observation_shape)
      training_actions = np.reshape(actions[:t + 1], (t + 1, 1))

      if done:
        final_return = 0
      else:
        final_return = bootstrap_return(observation_window, model,
          session, running_rnn_state)
      discounted_returns = discount_returns(
        rewards, gamma, final_return, t + 1)

      if not evaluate:
        training_rnn_state = run_updates(session, model, training_observations,
          training_actions, discounted_returns, training_rnn_state)

      if thread_id == 0 and iteration % 100 == 0:
        write_summaries(summary_writer, session, model, training_observations,
          training_actions, discounted_returns, iteration, training_rnn_state)

      if thread_id == 0 and iteration % 10000 == 0 and not evaluate:
        global_model.saver.save(session, train_dir + '/model.ckpt',
          global_step=iteration)

      actions[:] = 0
      rewards[:] = 0
      t_start = iteration + 1

    # Some management tasks are conducted after each episode
    if done:
      done = False
      ob = new_random_game(env, random_starts, env.action_space.n)
      observation_buffer[:] = resize_observation(ob, observation_shape,
        crop_centering)

      if use_rnn:
        running_rnn_state, training_rnn_state = reset_rnn_state(rnn_size)

      previous_rewards.append(episode_rewards)
      episode_count += 1
      if thread_id == 0:
        print('Thread: {} Episode: {} Rews: {} RunningAvgRew: '
              '{:.1f}'.format(thread_id, episode_count, episode_rewards,
              np.mean(previous_rewards)))
      episode_rewards = 0

class ACAgent(object):
  """
  ACAgent implements training of A3C algorithm for OpenAI Gym environments.
  """
  def __init__(self, **usercfg):
    """
    Args:
      usercfg: overrides for the default configuration
    """
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
        ob_shape=(84,84),
        crop_centering=(0.5,0.7),
        env_name = '',
        entropy_beta = 0.01,
        rms_decay = 0.99,
        rms_epsilon = 0.01,
        random_starts = 30,
        load_path = None,
        train_dir = 'train',
        evaluate = False,
        render = False,
      )

    self.config.update(usercfg)

    self.envs = [gym.make(self.config['env_name']) for _ in
      range(self.config['num_threads'])]

    action_num = self.envs[0].action_space.n

    with tf.variable_scope('global'), tf.device('/cpu:0'):
      self.global_model = ThreadModel(action_num, None, self.config)

    self.thread_models = []
    for thr in range(self.config['num_threads']):
      with tf.variable_scope('thread_{}'.format(thr)):
        thr_model = ThreadModel(action_num, self.global_model, self.config)
        self.thread_models.append(thr_model)

    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())
    if self.config['load_path'] is not None:
      self.global_model.saver.restore(self.session, self.config['load_path'])

  def learn(self):
    """
    Starts the learning threads.
    """
    threads = []
    for i in range(self.config['num_threads']):
      threads.append(threading.Thread(target=learning_thread, args=(
        i, self.config, self.session, self.thread_models[i], self.global_model,
        self.envs[i])))

    self.session.graph.finalize()

    for thread in threads:
      thread.start()
