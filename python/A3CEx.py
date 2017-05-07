import deepmind_lab
import numpy as np
import tensorflow as tf
import threading
import multiprocessing
import tensorflow.contrib.slim as slim
#import scipy
import os
#from time import sleep

# Create action dictionary
ACTIONS = {
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

#Tensorflow stuff

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# Processes Doom screen image to produce cropped and resized image. 
#def process_frame(frame):
#    s = frame[10:-10,30:-30]
#    s = scipy.misc.imresize(s,[84,84])
#    s = np.reshape(s,[np.prod(s.shape)]) / 255.0
#    return s

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




class AC_Network():
    def __init__(self,s_size,a_size,scope,trainer):
        with tf.variable_scope(scope):
            #Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
            self.imageIn = tf.reshape(self.inputs,shape=[-1,84,84,3])
            self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                inputs=self.imageIn,num_outputs=16,
                kernel_size=[8,8],stride=[4,4],padding='VALID')
            self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                inputs=self.conv1,num_outputs=32,
                kernel_size=[4,4],stride=[2,2],padding='VALID')
            hidden = slim.fully_connected(slim.flatten(self.conv2),256,activation_fn=tf.nn.elu)
            
            #Recurrent network for temporal dependencies
            #First LSTM Layer
            lstm_cell1 = tf.contrib.rnn.BasicLSTMCell(64,state_is_tuple=True)
            c_init1 = np.zeros((1, lstm_cell1.state_size.c), np.float32)
            h_init1 = np.zeros((1, lstm_cell1.state_size.h), np.float32)
            self.state_init1 = [c_init1, h_init1]
            c_in1 = tf.placeholder(tf.float32, [1, lstm_cell1.state_size.c])
            h_in1 = tf.placeholder(tf.float32, [1, lstm_cell1.state_size.h])
            self.state_in1 = (c_in1, h_in1)
            rnn_in = tf.expand_dims(hidden, [0]) #sets a batch size of 1 in front of the hidden array
            step_size = tf.shape(self.imageIn)[:1]
            state_in_tup1 = tf.contrib.rnn.LSTMStateTuple(self.state_in1[0], self.state_in1[1])
            with tf.variable_scope('lstm1'): #Need different named scopes for underlying LSTM weights to auto-name
                lstm_outputs1, lstm_state1 = tf.nn.dynamic_rnn(
                    lstm_cell1, rnn_in, initial_state=state_in_tup1, sequence_length=step_size,
                    time_major=False)
            lstm_c1, lstm_h1 = lstm_state1
            self.state_out1 = (lstm_c1[:1, :], lstm_h1[:1, :])

            #Second LSTM Layer
            lstm_cell2 = tf.contrib.rnn.BasicLSTMCell(256,state_is_tuple=True)
            c_init2 = np.zeros((1, lstm_cell2.state_size.c), np.float32)
            h_init2 = np.zeros((1, lstm_cell2.state_size.h), np.float32)
            self.state_init2 = [c_init2, h_init2]
            c_in2 = tf.placeholder(tf.float32, [1, lstm_cell2.state_size.c])
            h_in2 = tf.placeholder(tf.float32, [1, lstm_cell2.state_size.h])
            self.state_in2 = (c_in2, h_in2)
            state_in_tup2 = tf.contrib.rnn.LSTMStateTuple(self.state_in2[0], self.state_in2[1])
            with tf.variable_scope('lstm2'):
                lstm_outputs2, lstm_state2 = tf.nn.dynamic_rnn(
                    lstm_cell2, lstm_outputs1, initial_state=state_in_tup2, sequence_length=step_size,
                    time_major=False)
            lstm_c2, lstm_h2 = lstm_state2
            self.state_out2 = (lstm_c2[:1, :], lstm_h2[:1, :])
            rnn_out = tf.reshape(lstm_outputs2, [-1, 256])
            
            #Output layers for policy and value estimations
            self.policy = slim.fully_connected(rnn_out,a_size,
                activation_fn=tf.nn.softmax,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=None)
            self.value = slim.fully_connected(rnn_out,1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)
            
            #Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                #Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)
                
                #Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))







class Worker():
    def __init__(self,name,s_size,a_size,trainer,model_path,global_episodes):
        self.name = "worker_" + str(name)
        self.number = name        
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_"+str(self.number))

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(s_size,a_size,self.name,trainer)
        self.update_local_ops = update_target_graph('global',self.name)        
        
        #The Below code is related to setting up the environment
        self.actions = action_list
        
    def train(self,rollout,sess,gamma,bootstrap_value,rnn_state1,rnn_state2):
        rollout = np.array(rollout)
        observations = rollout[:,0]
        actions = rollout[:,1]
        rewards = rollout[:,2]
        next_observations = rollout[:,3]
        values = rollout[:,5]
        
        # Here we take the rewards and values from the rollout, and use them to 
        # generate the advantage and discounted returns. 
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages,gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        #rnn_state1 = self.local_AC.state_init1
        #rnn_state2 = self.local_AC.state_init2
        feed_dict = {self.local_AC.target_v:discounted_rewards,
            self.local_AC.inputs:np.vstack(observations),
            self.local_AC.actions:actions,
            self.local_AC.advantages:advantages,
            self.local_AC.state_in1[0]:rnn_state1[0],
            self.local_AC.state_in1[1]:rnn_state1[1],
            self.local_AC.state_in2[0]:rnn_state2[0],
            self.local_AC.state_in2[1]:rnn_state2[1]}
        v_l,p_l,e_l,g_n,v_n,_ = sess.run([self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.apply_grads],
            feed_dict=feed_dict)
        return v_l / len(rollout),p_l / len(rollout),e_l / len(rollout), g_n,v_n
        
    def work(self,max_episode_length,gamma,sess,coord,saver, is_training = True):
        self.env = deepmind_lab.Lab('nav_maze_static_01', ['RGB_INTERLACED'], config={
	'fps': str(60),
	'width': str(width),
	'height': str(height)})

        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        rnn_state1 = self.local_AC.state_init1
        rnn_state2 = self.local_AC.state_init2

        print ("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():                 
            while not coord.should_stop():

                print self.name + " running episode " + str(episode_count)

                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                d = False
                
                self.env.reset()

                obs = self.env.observations()
                s = np.reshape(obs['RGB_INTERLACED'], (1, -1))
                s = s / float(np.sum(s)) #Normalize observations
                episode_frames.append(s)
                #rnn_state1 = self.local_AC.state_init1
                #rnn_state2 = self.local_AC.state_init2
                
                while self.env.is_running():
                    #Take an action using probabilities from policy network output.
                    a_dist,v,rnn_state1,rnn_state2 = sess.run([self.local_AC.policy, self.local_AC.value, 
                        self.local_AC.state_out1, self.local_AC.state_out2], 
                        feed_dict={self.local_AC.inputs:s,
                        self.local_AC.state_in1[0]:rnn_state1[0],
                        self.local_AC.state_in1[1]:rnn_state1[1],
                        self.local_AC.state_in2[0]:rnn_state2[0],
                        self.local_AC.state_in2[1]:rnn_state2[1]})
                    a = np.random.choice(a_dist[0],p=a_dist[0])
                    a = np.argmax(a_dist == a)

                    r = self.env.step(self.actions[a], num_steps = 1)
                    d = self.env.is_running()
                    if d:
                        obs = self.env.observations()
                        s1 = np.reshape(obs['RGB_INTERLACED'], (1, -1))
                        s1 = s1 / float(np.sum(s1)) #Normalize observations

                        episode_frames.append(s1)
                    else:
                        s1 = s
                        
                    episode_buffer.append([s,a,r,s1,d,v[0,0]])
                    episode_values.append(v[0,0])

                    episode_reward += r
                    s = s1                    
                    total_steps += 1
                    episode_step_count += 1
                    
                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == 30:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        if is_training:
                            v1 = sess.run(self.local_AC.value, 
                                feed_dict={self.local_AC.inputs:s,
                                self.local_AC.state_in1[0]:rnn_state1[0],
                                self.local_AC.state_in1[1]:rnn_state1[1],
                                self.local_AC.state_in2[0]:rnn_state2[0],
                                self.local_AC.state_in2[1]:rnn_state2[1]})[0,0]
                            v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,v1,rnn_state1,rnn_state2)
                            sess.run(self.update_local_ops)

                        episode_buffer = []
                                            
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))
                
                # Update the network using the experience buffer at the end of the episode if there's unused frames
                if len(episode_buffer) != 0 and is_training:
                    v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,0.0,rnn_state1,rnn_state2)
                                
                    
                # Periodically save model parameters and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0 and is_training:
                    if episode_count % 10 == 0 and self.name == 'worker_0':
                        saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                        print "Saved Model"

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1




max_episode_length = 300
gamma = .99 # discount rate for advantage estimation and reward discounting
width = 84
height = 84
s_size = 84*84*3 # Observations are RGB frames of 84 * 84 * 3
a_size = 8 # Agent can choose 8 actions
model_path = './model'
is_training = False
load_model = True



tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

with tf.device("/cpu:0"): 
    global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    master_network = AC_Network(s_size,a_size,'global',None) # Generate global network
    num_workers = multiprocessing.cpu_count() #multiprocessing.cpu_count() # Set workers ot number of available CPU threads
    print "Starting " + str(num_workers) + " workers"
    workers = []
    # Create worker classes
    for i in range(num_workers):
        workers.append(Worker(i,s_size,a_size,trainer,model_path,global_episodes))
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
        
    if is_training:
        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate thread.
        worker_threads = []
        for worker in workers:
            worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
            t = threading.Thread(target=(worker_work))
            t.start()
            sleep(0.5)
            worker_threads.append(t)
        coord.join(worker_threads)
    else:
        #This will run the model on one thread so we can view it graphically
        workers[0].work(max_episode_length,gamma,sess,coord,saver, is_training)


