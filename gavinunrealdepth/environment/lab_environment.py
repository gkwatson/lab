# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from multiprocessing import Process, Pipe
import cv2
import numpy as np
import deepmind_lab

from environment import environment
from constants import ENV_NAME

COMMAND_RESET     = 0
COMMAND_ACTION    = 1
COMMAND_TERMINATE = 2

def worker(conn):
  level = ENV_NAME
  env = deepmind_lab.Lab(
    level,
    ['RGBD_INTERLACED'],
    config={
      'fps': str(60),
      'width': str(84),
      'height': str(84)
    })
  conn.send(0)
  
  while True:
    command, arg = conn.recv()

    if command == COMMAND_RESET:
      env.reset()
      obs = env.observations()['RGBD_INTERLACED']
      conn.send(obs)
    elif command == COMMAND_ACTION:
      reward = env.step(arg, num_steps=4)
      terminal = not env.is_running()
      if not terminal:
        obs = env.observations()['RGBD_INTERLACED']
      else:
        obs = 0
      conn.send([obs, reward, terminal])
    elif command == COMMAND_TERMINATE:
      break
    else:
      print("bad command: {}".format(command))
  env.close()      
  conn.send(0)
  conn.close()


def _action(*entries):
  return np.array(entries, dtype=np.intc)


class LabEnvironment(environment.Environment):
  ACTION_LIST = [
    _action(-20,   0,  0,  0, 0, 0, 0), # look_left
    _action( 20,   0,  0,  0, 0, 0, 0), # look_right
    #_action(  0,  10,  0,  0, 0, 0, 0), # look_up
    #_action(  0, -10,  0,  0, 0, 0, 0), # look_down
    _action(  0,   0, -1,  0, 0, 0, 0), # strafe_left
    _action(  0,   0,  1,  0, 0, 0, 0), # strafe_right
    _action(  0,   0,  0,  1, 0, 0, 0), # forward
    _action(  0,   0,  0, -1, 0, 0, 0), # backward
    #_action(  0,   0,  0,  0, 1, 0, 0), # fire
    #_action(  0,   0,  0,  0, 0, 1, 0), # jump
    #_action(  0,   0,  0,  0, 0, 0, 1)  # crouch
  ]

  @staticmethod
  def get_action_size():
    return len(LabEnvironment.ACTION_LIST)
  
  def __init__(self):
    environment.Environment.__init__(self)
    
    self.conn, child_conn = Pipe()
    self.proc = Process(target=worker, args=(child_conn,))
    self.proc.start()
    self.conn.recv()
    self.reset()

  def reset(self):
    self.conn.send([COMMAND_RESET, 0])
    obs = self.conn.recv()
    
    self.last_state, self.last_depths = self._preprocess_frame(obs)
    self.last_action = 0
    self.last_reward = 0

  def stop(self):
    self.conn.send([COMMAND_TERMINATE, 0])
    ret = self.conn.recv()
    self.conn.close()
    self.proc.join()
    print("lab environment stopped")
    
  def _preprocess_frame(self, image_withd):
    #get depth image
    depth_arr = image_withd[:,:,3:].astype(np.float32)

    #Resize and stretch upper values of distribution
    depth_arr = cv2.resize(depth_arr,(1,8)).reshape(-1)
    depth_arr = (depth_arr / 255.0) ** 10
    bins = np.array([0.0, 0.05, 0.175, 0.3, 0.425, 0.55, 0.675, 0.8, 1.0])
    inds = np.digitize(depth_arr, bins)
    for i in range(depth_arr.size):
      depth_arr[i] = bins[inds[i]-1]

    image = image_withd[:,:,:3]
    image = image.astype(np.float32)
    image = image / 255.0
    return (image, depth_arr)

  def process(self, action):
    real_action = LabEnvironment.ACTION_LIST[action]

    self.conn.send([COMMAND_ACTION, real_action])
    obs, reward, terminal = self.conn.recv()

    if not terminal:
      state, depths = self._preprocess_frame(obs)
    else:
      state = self.last_state
      depths = self.last_depths
    
    pixel_change = self._calc_pixel_change(state, self.last_state)
    self.last_state = state
    self.last_depths = depths
    self.last_action = action
    self.last_reward = reward
    return state, reward, terminal, pixel_change, depths
