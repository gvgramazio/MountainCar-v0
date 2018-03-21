# CartPole-v0
import numpy as np
import tensorflow as tf
from collections import deque

class DQNAgent:
    def __init__(
        self,
        n_actions,
        space_shape,
        batch_size = 32,
        learning_rate = 0.01,
        epsilon = 0.9,
        gamma = 0.9,
        target_replace_iter = 100,
        replay_memory_size = 2000,
        output_graph = False,
        run_time_stats = False, # True has no effect if output_graph == False
        run_time_stats_period = 100,
        restore_tf_variables = False,
        restore_tf_variables_path = "models/model.ckpt" # Has no effect if restore_tf_variables == False
        ):
        ''' Hyper parameters '''
        self.n_actions = n_actions
        self.space_shape = space_shape
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.target_replace_iter = target_replace_iter

        # Save the hyper parameters
        self.params = self.__dict__.copy()

        # Replay memory
        self.D = deque(maxlen=replay_memory_size)

        ''' Tensorflow placeholders '''
        self.observation = tf.placeholder(tf.float32, [None, space_shape], name='observation')
        self.action = tf.placeholder(tf.int32, [None, ], name='action')
        self.reward = tf.placeholder(tf.float32, [None, ], name='reward')
        self.observation_ = tf.placeholder(tf.float32, [None, space_shape], name='observation_')

        # Set seeds
        tf.set_random_seed(1)
        np.random.seed(1)

        kernel_initializer, bias_initializer = tf.random_normal_initializer(0, 0.3), tf.constant_initializer(0.1)

        with tf.variable_scope('eval_net'):        # evaluation network
            l_eval = tf.layers.dense(self.observation, 10, tf.nn.relu, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='eval_layer1')
            self.q = tf.layers.dense(l_eval, n_actions, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='eval_layer2')

        with tf.variable_scope('target_net'):   # target network, not to train
            l_target = tf.layers.dense(self.observation_, 10, tf.nn.relu, trainable=False, name='target_layer1')
            self.q_next = tf.layers.dense(l_target, n_actions, trainable=False, name='target_layer2')

        with tf.variable_scope('target_replacement'):
            t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
            e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
            self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        with tf.variable_scope('q_target'):
            self.q_target = self.reward + self.gamma * tf.reduce_max(self.q_next, axis=1)    # shape=(None, )

        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.action)[0], dtype=tf.int32), self.action], axis=1)
            q_eval_wrt_a = tf.gather_nd(params=self.q, indices=a_indices)                    # shape=(None, )

        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(tf.squared_difference(self.q_target, q_eval_wrt_a))

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.scalar('loss', loss)
            self.merger_op = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter("logs/", self.sess.graph)
            if run_time_stats:
                self.run_time_stats_period = run_time_stats_period

        self.sess.run(tf.global_variables_initializer())

        # Add op to save and restore all the variables.
        self.saver = tf.train.Saver()

        # Restore variables from disk.
        if restore_tf_variables == True:
            self.saver.restore(self.sess, restore_tf_variables_path)


    def choose_action(self, s):
        s = s[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q, feed_dict={self.observation: s})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def store_transition(self, observation, action, reward, observation_, done):
        self.D.append((observation, action, reward, observation_, done))

    def update_target(self):
        if not hasattr(self, 'learning_step_counter'):
            self.learning_step_counter = 0

        if self.learning_step_counter % self.target_replace_iter == 0:
            self.sess.run(self.replace_target_op)
        self.learning_step_counter += 1

    def learn(self):
        # Update target net
        self.update_target()

        # Collect a random batch of D
        batch_size = min(len(self.D), self.batch_size)
        batch_indeces = np.random.randint(0, len(self.D), batch_size)
        batch_observation = deque(maxlen=batch_size)
        batch_action = deque(maxlen=batch_size)
        batch_reward = deque(maxlen=batch_size)
        batch_observation_ = deque(maxlen=batch_size)
        batch_q_target = deque(maxlen=batch_size)
        batch_done = deque(maxlen=batch_size)
        for j in batch_indeces:
            observation_j, action_j, reward_j, observation_j_, done_j = self.D[j]
            batch_observation.append(observation_j)
            batch_action.append(action_j)
            batch_reward.append(reward_j)
            batch_observation_.append(observation_j_)
            batch_done.append(done_j)
        batch_q_target = self.sess.run(self.q_target, {
            self.observation_: batch_observation,
            self.reward: batch_reward})
        for j in range(0,len(batch_done)):
            if batch_done[j]:
                batch_q_target[j] = batch_reward[j]

        # Train and store summary
        if hasattr(self, 'run_time_stats_period') and self.learning_step_counter % self.run_time_stats_period == self.run_time_stats_period-1: # Record execution stats
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            _, summary = self.sess.run(
                [self.train_op, self.merger_op], {
                    self.observation: batch_observation,
                    self.action: batch_action,
                    self.reward: batch_reward,
                    self.observation_: batch_observation_},
                options=run_options,
                run_metadata=run_metadata)
            self.writer.add_run_metadata(run_metadata, 'step%03d' % self.learning_step_counter)
            self.writer.add_summary(summary, self.learning_step_counter)
        else: # Record a summary
            _, summary = self.sess.run([self.train_op, self.merger_op], {
                self.observation: batch_observation,
                self.action: batch_action,
                self.reward: batch_reward,
                self.observation_: batch_observation_})
            self.writer.add_summary(summary, self.learning_step_counter)

    def show_parameters(self):
        ''' Helper function to show the hyper parameters '''
        for key, value in self.params.items():
            print key, '=', value

    def save(self, path):
        save_path = self.saver.save(self.sess, path)
        print 'Model saved in path:', save_path
