import numpy as np
import deepmind_lab
import pprint
import tensorflow as tf

# Construct and start the environment
env = deepmind_lab.Lab('seekavoid_arena_01', ['RGB_INTERLACED'], config={
	'fps': str(60),
	'width': str(360),
	'height': str(240)})
env.reset()

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


#Example
old_obs = env.observations()
reward = env.step(ACTIONS['forward'], num_steps = 4)

# Retrieve the observations of the environment in its new state
obs = env.observations() # dict of Numpy arrays
pprint.pprint(obs['RGB_INTERLACED'])
rgb_i = obs['RGB_INTERLACED']


print type(rgb_i)

#Tensorflow stuff

def turnRGBIIntoNPArray(rgbi):
    list4tf = []
    for width in rgbi:
        for colors in width:
            for color in colors:
                #And normalize
                list4tf.append(float(color)/rgbi.size)
    return np.array([list4tf])

arrayOfState = turnRGBIIntoNPArray(rgb_i)

#lstm = tf.contrib.rnn.BasicLSTMCell(128)
## Initial state of the LSTM memory.
#print [len(list4tf), lstm.state_size]
#state = tf.zeros(lstm.state_size)
#print state
#probabilities = []
#loss = 0.0

## The value of state is updated after processing each batch of words.
#output, state = lstm(list4tf, state)


#These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[1,arrayOfState.size],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([arrayOfState.size,8],0,0.01))
Qout = tf.matmul(inputs1,W)
predict = tf.argmax(Qout,1)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1,8],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

init = tf.initialize_all_variables()

# Set learning parameters
y = .99
e = 0.1
num_episodes = 2000
#create lists to contain total rewards and steps per episode
jList = []
rList = []
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
	print "EPISODE: " + str(i)
        #Reset environment and get first new observation
        env.reset()
	obs = env.observations()
        s = turnRGBIIntoNPArray(obs['RGB_INTERLACED'])
 
        rAll = 0
        d = False
        j = 0
        #The Q-Network
        while j < 100:
            print "Step: " + str(j)
            j+=1
            #Choose an action (with e chance of random action) from the Q-network
            a,allQ = sess.run([predict,Qout],feed_dict={inputs1:s})

            #Randomly sample from actions based on their Q values
            selector = np.random.random() * np.sum(allQ[0])
            run_sum = 0.0
            a[0] = 7
            for i in range(0, 8):
                run_sum += allQ[0, i]
                if selector < run_sum:
                    a[0] = i
                    break

            #Possibility of random action
            if np.random.rand(1) < e:
                a[0] = np.random.randint(0, 8)

            #Get new state and reward from environment
            print "a = " + str(a)
            print "allQ = " + str(allQ)
            r = env.step(action_list[a[0]], num_steps = 10)
            obs = env.observations()
            d = env.is_running()
            s1 = turnRGBIIntoNPArray(obs['RGB_INTERLACED'])

            #Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout,feed_dict={inputs1:s1})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            print "Q1 = " + str(Q1)
            print "maxQ1 = " + str(maxQ1)
            targetQ = allQ
            targetQ[0,a[0]] = r + y*maxQ1
            #Train our network using target and predicted Q values
            _,W1 = sess.run([updateModel,W],feed_dict={inputs1:s,nextQ:targetQ})
            rAll += r
            s = s1
            if d == False:
                #Reduce chance of random action as we train the model.
                e = 1./((i/50) + 10)
                break
        jList.append(j)
        rList.append(rAll)
print "Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%"



assert rgb_i.shape == (481, 720, 3)
