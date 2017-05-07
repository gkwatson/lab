import numpy as np
import deepmind_lab
import pprint
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_x = np.zeros_like(x)
    running_add = 0
    for t in reversed(xrange(0, x.size)):
        running_add = running_add * gamma + x[t]
        discounted_x[t] = running_add
    return discounted_x

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

#in case we want to sample a bigger screen
def rebin84x84(arr)
    return arr.reshape((84, arr.shape[0]//84, 84, -1, 3)).mean(axis=3).mean(1)



# Construct and start the environment
width = 84
height = 84

env = deepmind_lab.Lab('nav_maze_static_01', ['RGB_INTERLACED'], config={
	'width': str(width),
	'height': str(height)})

env.reset()

# Create action dictionary
EX_ACTIONS = {
  'look_left': np.array((-20, 0, 0, 0, 0, 0, 0), dtype=np.intc),
  'look_right': np.array((20, 0, 0, 0, 0, 0, 0), dtype=np.intc),
  'look_up': np.array((0, 10, 0, 0, 0, 0, 0), dtype=np.intc),
  'look_down': np.array((0, -10, 0, 0, 0, 0, 0), dtype=np.intc),
  'strafe_left': np.array((0, 0, -1, 0, 0, 0, 0), dtype=np.intc),
  'strafe_right': np.array((0, 0, 1, 0, 0, 0, 0), dtype=np.intc),
  'forward': np.array((0, 0, 0, 1, 0, 0, 0), dtype=np.intc),
  'backward': np.array((0, 0, 0, -1, 0, 0, 0), dtype=np.intc),
  'fire': np.array((0, 0, 0, 0, 1, 0, 0), dtype=np.intc),
  'jump': np.array((0, 0, 0, 0, 0, 1, 0), dtype=np.intc),
  'crouch': np.array((0, 0, 0, 0, 0, 0, 1), dtype=np.intc)
}

action_list = [
  np.array((-20, 0, 0, 0, 0, 0, 0), dtype=np.intc),
  np.array((20, 0, 0, 0, 0, 0, 0), dtype=np.intc),
  np.array((0, 10, 0, 0, 0, 0, 0), dtype=np.intc),
  np.array((0, -10, 0, 0, 0, 0, 0), dtype=np.intc),
  np.array((0, 0, -1, 0, 0, 0, 0), dtype=np.intc),
  np.array((0, 0, 1, 0, 0, 0, 0), dtype=np.intc),
  np.array((0, 0, 0, 1, 0, 0, 0), dtype=np.intc),
  np.array((0, 0, 0, -1, 0, 0, 0), dtype=np.intc),
  np.array((0, 0, 0, 0, 1, 0, 0), dtype=np.intc),
  np.array((0, 0, 0, 0, 0, 1, 0), dtype=np.intc),
  np.array((0, 0, 0, 0, 0, 0, 1), dtype=np.intc)
]

#array of zeroes same size as screen * colors
arrayOfState = np.zeros(height * width * 3)




#Tensorflow stuff

num_episodes = 1000
max_episode_length = 300
gamma = .99 # discount rate for advantage estimation and reward discounting
s_size = arrayOfState.size # Observations are RGB of height * width * 3
a_size = 8 # Agent can choose 8 actions
load_model = False
model_path = './model'
global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
increment = global_episodes.assign_add(1)
trainer = tf.train.AdamOptimizer(learning_rate=1e-4)





#Setting up the tensorflow graph
#Input and visual encoding layers
inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
imageIn = tf.reshape(inputs, shape=[-1,width,height,3])
conv1 = slim.conv2d(activation_fn=tf.nn.elu, inputs=imageIn, num_outputs=16, kernel_size=[8,8],
                    stride=[4,4],padding='VALID')
conv2 = slim.conv2d(activation_fn=tf.nn.elu, inputs=conv1, num_outputs=32, kernel_size=[4,4], 
                    stride=[2,2],padding='VALID')
hidden = slim.fully_connected(slim.flatten(conv2),256,activation_fn=tf.nn.elu)

#Recurrent network for temporal dependencies
lstm_cell = tf.contrib.rnn.BasicLSTMCell(256,state_is_tuple=True)
c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
state_init = [c_init, h_init]
c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
state_in = (c_in, h_in)
rnn_in = tf.expand_dims(hidden, [0])
step_size = tf.shape(imageIn)[:1]
state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm_cell, rnn_in, initial_state=state_in, 
                                             sequence_length=step_size, time_major=False)
lstm_c, lstm_h = lstm_state
state_out = (lstm_c[:1, :], lstm_h[:1, :])
rnn_out = tf.reshape(lstm_outputs, [-1, 256])

#Output layers for policy and value estimations
policy = slim.fully_connected(rnn_out, a_size, activation_fn = tf.nn.softmax,  
                              weights_initializer = normalized_columns_initializer(0.01), biases_initializer = None)
value = slim.fully_connected(rnn_out, 1, activation_fn = None, 
                             weights_initializer = normalized_columns_initializer(1.0), biases_initializer = None)

#Ops for loss functions and gradient updating.
actions_input = tf.placeholder(shape=[None], dtype=tf.int32)
actions_onehot = tf.one_hot(actions_input, a_size, dtype=tf.float32)
target_v = tf.placeholder(shape=[None], dtype=tf.float32)
advantages_input = tf.placeholder(shape=[None], dtype=tf.float32)

responsible_outputs = tf.reduce_sum(policy * actions_onehot, [1])

#Loss functions
value_loss = 0.5 * tf.reduce_sum(tf.square(target_v - tf.reshape(value,[-1])))
entropy = - tf.reduce_sum(policy * tf.log(policy))
policy_loss = -tf.reduce_sum(tf.log(responsible_outputs)*advantages_input)
loss = 0.5 * value_loss + policy_loss - entropy * 0.01

#Get gradients from local network using local losses
local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
gradients = tf.gradients(loss,local_vars)
var_norms = tf.global_norm(local_vars)
grads, grad_norms = tf.clip_by_global_norm(gradients,40.0)

#Apply local gradients to global network
apply_grads = trainer.apply_gradients(zip(grads,local_vars))





#Training loop
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    episode_count = sess.run(global_episodes)
    episode_rewards = []
    episode_lengths = []
    episode_mean_values = []
    total_steps = 0
    print "Starting training"                

    for i in range(num_episodes):

        episode_buffer = []
        episode_values = []
        episode_frames = []
        episode_reward = 0
        episode_step_count = 0
        d = False
        
        env.reset()
        obs = env.observations()
        s = np.reshape(obs['RGB_INTERLACED'], (1, -1))
        s = s / float(s.size) #Normalize observations

        episode_frames.append(s)
        rnn_state = state_init
        
        while env.is_running():
            #Take an action using probabilities from policy network output.
            a_dist, v, rnn_state = sess.run([policy, value, state_out], feed_dict = {inputs:s, state_in[0]:rnn_state[0],
                                                                                     state_in[1]:rnn_state[1]})
            a = np.random.choice(a_dist[0], p = a_dist[0])
            a = np.argmax(a_dist == a)

            r = env.step(action_list[a], num_steps = 1)

            #print "Move distribution: " + str(a_dist)
            #print "Move chosen: " + str(action_list[a])

            d = env.is_running()

            if d:
                obs = env.observations()
                s1 = np.reshape(obs['RGB_INTERLACED'], (1, -1))
                s1 = s1 / float(s1.size) #Normalize observations

                episode_frames.append(s1)
            else:
                s1 = s
                
            episode_buffer.append([s, a, r, s1, d, v[0,0]])
            episode_values.append(v[0,0])

            episode_reward += r
            s = s1
            total_steps += 1
            episode_step_count += 1
            
            # If the episode hasn't ended, but the experience buffer is full, then we
            # make an update step using that experience rollout.
            if len(episode_buffer) == 30 and d and episode_step_count != max_episode_length - 1:
                # Since we don't know what the true final return is, we "bootstrap" from our current
                # value estimation.
                v1 = sess.run(value, feed_dict={inputs:s, state_in[0]:rnn_state[0], state_in[1]:rnn_state[1]})[0,0]

		rollout = np.array(episode_buffer)
		observations = rollout[:,0]
		actions = rollout[:,1]
		rewards = rollout[:,2]
		next_observations = rollout[:,3]
		values = rollout[:,5]
		
		# Here we take the rewards and values from the rollout, and use them to 
		# generate the advantage and discounted returns. 
		# The advantage function uses "Generalized Advantage Estimation"
		rewards_plus = np.asarray(rewards.tolist() + [v1])
		discounted_rewards = discount(rewards_plus, gamma)[:-1]
		value_plus = np.asarray(values.tolist() + [v1])
		advantages = rewards + gamma * value_plus[1:] - value_plus[:-1]
		advantages = discount(advantages, gamma)

		# Update the global network using gradients from loss
		# Generate network statistics to periodically save
		rnn_state = state_init
		feed_dict = {target_v : discounted_rewards, inputs : np.vstack(observations), actions_input : actions,
		             advantages_input : advantages, state_in[0] : rnn_state[0], state_in[1] : rnn_state[1]}
		v_l, p_l, e_l, g_n, v_n, _ = sess.run([value_loss, policy_loss, entropy, grad_norms, var_norms, apply_grads],
                                                      feed_dict = feed_dict)
                v_l = v_l / len(rollout)
                p_l = p_l / len(rollout)
                e_l = e_l / len(rollout)

                episode_buffer = []
                                    
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_step_count)
        episode_mean_values.append(np.mean(episode_values))
        
        # Update the network using the experience buffer at the end of the episode.
        if len(episode_buffer) != 0:
            v1 = 0.0
            rollout = np.array(episode_buffer)
            observations = rollout[:,0]
            actions = rollout[:,1]
            rewards = rollout[:,2]
            next_observations = rollout[:,3]
            values = rollout[:,5]

            # Here we take the rewards and values from the rollout, and use them to 
            # generate the advantage and discounted returns. 
            # The advantage function uses "Generalized Advantage Estimation"
            rewards_plus = np.asarray(rewards.tolist() + [v1])
            discounted_rewards = discount(rewards_plus, gamma)[:-1]
            value_plus = np.asarray(values.tolist() + [v1])
            advantages = rewards + gamma * value_plus[1:] - value_plus[:-1]
            advantages = discount(advantages, gamma)

            # Update the global network using gradients from loss
            # Generate network statistics to periodically save
            rnn_state = state_init
            feed_dict = {target_v : discounted_rewards, inputs : np.vstack(observations), actions_input : actions,
			 advantages_input : advantages, state_in[0] : rnn_state[0], state_in[1] : rnn_state[1]}
            v_l, p_l, e_l, g_n, v_n, _ = sess.run([value_loss, policy_loss, entropy, grad_norms, var_norms, apply_grads],
		                                  feed_dict = feed_dict)
            v_l = v_l / len(rollout)
            p_l = p_l / len(rollout)
            e_l = e_l / len(rollout)
                        
            
#        # Periodically save gifs of episodes, model parameters, and summary statistics.
#        if episode_count % 5 == 0 and episode_count != 0:
#            if self.name == 'worker_0' and episode_count % 25 == 0:
#                time_per_step = 0.05
#                images = np.array(episode_frames)
#                make_gif(images,'./frames/image'+str(episode_count)+'.gif',
#                    duration=len(images)*time_per_step,true_image=True,salience=False)
#            if episode_count % 250 == 0 and self.name == 'worker_0':
#                saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
#                print ("Saved Model")
#
#            mean_reward = np.mean(self.episode_rewards[-5:])
#            mean_length = np.mean(self.episode_lengths[-5:])
#            mean_value = np.mean(self.episode_mean_values[-5:])
#            summary = tf.Summary()
#            summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
#            summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
#            summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
#            summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
#            summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
#            summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
#            summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
#            summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
#            self.summary_writer.add_summary(summary, episode_count)
#
#            self.summary_writer.flush()

        print "Episode: " + str(episode_count)
        print "Episode Rewards: " + str(episode_rewards)
        print "Episode Lengths: " + str(episode_lengths)
        sess.run(increment)
        episode_count += 1



