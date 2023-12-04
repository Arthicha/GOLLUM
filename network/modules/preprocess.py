
# ------------------- import modules ---------------------

# standard modules
import time, sys, os
from copy import deepcopy

# math-related modules
import numpy as np # cpu array
import torch # cpu & gpu array

# modular network
from modules.torchNet import torchNet


# ------------------- configuration variables ---------------------
EPSILON = 1e-6
GAMMA = 0.05
# ------------------- class PreprocessingNetwork ---------------------

class PreprocessingNetwork(torchNet):

	# -------------------- constructor -----------------------
	# (private)

	def __init__(self,hyperparams):

		super().__init__()

		# update hyperparameter
		self.__n_state = hyperparams.n_state
		self.__n_in = hyperparams.n_in

		self.__w_r1_r1 = 1.0-hyperparams.w_time
		self.__w_s1_r1 = hyperparams.w_time

		# initialize neuron activity
		self.__states = self.zeros(1,self.__n_state) 
		self.__inputs = self.zeros(1,self.__n_state)
		self.__outputs = self.zeros(1,self.__n_state)

		# initialize connection weight
		self.__A = self.zeros(self.__n_state,self.__n_state) 
		self.__B = self.zeros(self.__n_state,self.__n_state)

		# initialize state connection proability
		self.__connection = hyperparams.connection

		# reset everything before use
		self.reset()


	# -------------------- update/handle functions -----------------------
	# (private)

	def __reset_connection(self):


		for i_from in range(self.__n_state):
			
			i_tos = torch.where(self.__connection[i_from] > GAMMA)
			
			if (i_tos[0].shape[0]) > 1:
				for i_to in i_tos:
					self.__B[i_from,i_to] = 1


	# -------------------- update/handle functions -----------------------
	# (public)

	def reset(self):

		# reset state
		self.__outputs *= 0.0

		# reset connection
		self.__reset_connection()


	def forward(self,inputs,state):
		self.__inputs = inputs
		self.__states = state


		self.__outputs = (self.__inputs) + self.__w_s1_r1*(self.__states@self.__B) + self.__w_r1_r1*self.__outputs  - 1
		#self.__outputs[self.__outputs != torch.max(self.__outputs)] *= 0
		#print('s',self.__states)
		#print('i',(self.__inputs))
		return torch.clamp(self.__outputs,0,1)








