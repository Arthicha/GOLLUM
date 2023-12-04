# ------------------- import modules ---------------------

# standard modules
import time, sys, os
from copy import deepcopy
import configparser

# math-related modules
import numpy as np # cpu array
import torch # cpu & gpu array
from torch.autograd import Variable

# modular network
from optim import Optim

# experience replay
from utils.utils import TorchReplay as Replay


# ------------------- configuration variables ---------------------
EPSILON = 1e-6
# ------------------- class GD ---------------------

class GradientDescent(Optim):

	
	# -------------------- constructor -----------------------
	# (private)

	def setup(self,config):

		self.__lr = float(config["CRITICOPTIM"]["LR"])
		self.__iteration = int(config["CRITICOPTIM"]["ITERATION"])
		self.__verbose = False
		# reset everything before use
		self.reset()

	def set_verbose(self,state):
		self.__verbose = state

	def attach_valuenet(self,vnet):
		self.vnet = vnet

	def attach_returnfunction(self,func):
		self.__function = func

	
	# ------------------------- update and learning ----------------------------
	# (public)

	
	def update(self,states,targets):
		for i in range(self.__iteration):

			self.vnet.zero_grad()

			predicted_value = self.__function(self.vnet(states))
			prepro_target = self.__function(targets)
			
			loss = torch.mean(torch.pow(prepro_target-predicted_value,2))
			loss.backward()
			
			with torch.no_grad():
				self.W -= (self.__lr*self.W.grad).detach()
			self.vnet.apply_noise(0)

			if self.__verbose:
				print("\tvalue loss:",loss.item())
		
		

		
		self.reset()
		




