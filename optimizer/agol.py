# ------------------- import modules ---------------------

# standard modules
import time, sys, os
from copy import deepcopy
import configparser

# math-related modules
import numpy as np # cpu array
import torch # cpu & gpu array
from torch.distributions import Normal

# modular network
from optim import Optim

# ------------------- configuration variables ---------------------
EPS = 1e-6
ENDINGCLIP = 5 # throw away n-last timestep

# ------------------- class AddedGradientOnlineLearning ---------------------
class AddedGradientOnlineLearning(Optim):
	# -------------------- constructor -----------------------
	# (private)

	def setup(self,config):
		self.vnet = None

		# initialize replay buffer
		self.__sigma = float(config["ACTOROPTIM"]["SIGMA"])
		self.__sigmas = self.zeros(self.W.shape[0],self.W.shape[1]) + self.__sigma
		self.__sigmamin = float(config["ACTOROPTIM"]["SIGMAMIN"])
		self.__sigmamax = float(config["ACTOROPTIM"]["SIGMAMAX"])
		self.__min_grad = float(config["ACTOROPTIM"]["MINGRAD"])
		self.__lr = float(config["ACTOROPTIM"]["LR"])

		# reset everything before use
		self.reset()

	def attach_valuenet(self,vnet):
		self.vnet = vnet

	def attach_returnfunction(self,func):
		self.compute_return = func

	# ------------------------- update and learning ----------------------------
	# (public)

	
	def update(self,states,weights,feedbacks,grads):
		
		if self.vnet is None:
			rewards = self.compute_return(feedbacks)
			advantage = (rewards-torch.mean(rewards,dim=0).unsqueeze(0))
			std_advantage =  advantage / (torch.std(rewards,dim=0).unsqueeze(0)+EPS)
		else:
			advantage = self.compute_return(feedbacks)-self.compute_return(self.vnet(states))
			std = (EPS+torch.sqrt(torch.mean(torch.pow(advantage,2)))).unsqueeze(0)
			std_advantage = advantage/std

		std_advantage[std_advantage < 0] *= 0.1
		std_advantage = torch.clamp(std_advantage,-3.0,3.0)
		
		update = (torch.abs(grads)*(weights-self.W))* std_advantage
		update = torch.sum(20*update[:,:-ENDINGCLIP] , dim=[0,1])/torch.pow(self.__sigmas,2)
		dw = torch.clamp(self.__lr*0.1*1e-4*update,-self.__min_grad,self.__min_grad)
		
		dsigma = std_advantage*torch.abs(grads)*(torch.pow(weights-self.W,2)-torch.pow(self.__sigmas,2))/torch.pow(self.__sigmas,3)
		dsigma = self.__lr*0.1*1e-5*torch.sum(dsigma[:,:-ENDINGCLIP],dim=[0,1])

		with torch.no_grad():
			self.W += (dw).detach()
			self.__sigmas = torch.clamp(self.__sigmas + dsigma,self.__sigmamin,self.__sigmamax)
		print('w',self.W[:,2])
			
	# -------------------- apply noise -----------------------

	def wnoise(self):
		self.dist = Normal(loc=self.W.detach()*0,scale=self.__sigmas)
		noise = self.dist.rsample()
		return noise

	# -------------------- exploration handling -----------------------

	def get_sigma(self):
		return self.__sigmas.detach()


	def set_sigma(self,sigmas):
		self.__sigmas[:sigmas.shape[0]] = sigmas




	
		




