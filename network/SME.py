
# ------------------- import modules ---------------------

# standard modules
import time, sys, os
from copy import deepcopy
import configparser

# math-related modules
import numpy as np # cpu array
import torch # cpu & gpu array

# modular network
from modules.utils import HyperParams
from modules.torchNet import torchNet 
from modules.preprocess import PreprocessingNetwork
from modules.centralpattern import SequentialCentralPatternGenerator
from modules.basis import BasisNetwork 
from modules.motor import MotorNetwork 

#plot
import matplotlib.pyplot as plt

# ------------------- configuration variables ---------------------

EPSILON = 1e-6 # a very small value

# ------------------- class SDN ---------------------

class SequentialMotionExecutor(torchNet):
	'''
	Sequential Motion Executor : Actor Network
	Parameters:
		connection/transition matrix from 'connection' 
		hyperparameter from a .init file at 'configfile'
	'''

	# ---------------------- constructor ------------------------ 
	def __init__(self,configfile, connection):

		super().__init__()

		# initialize hyperparameter
		config = configparser.ConfigParser()
		config.read(configfile)

		self.__n_state = connection.shape[0]
		self.__n_in = int(config['HYPERPARAM']['NIN'])
		self.__n_out = int(config['HYPERPARAM']['NOUT'])
		self.__t_init = int(config['HYPERPARAM']['TINIT'])

		self.__connection = connection

		self.__hyperparams = HyperParams(self.__n_state,self.__n_in,self.__n_out)
		self.__hyperparams.w_time = float(config['C']['W_TIME']) + 0.01*((self.__n_state//4)-1)
		self.__hyperparams.connection = self.__connection

		# ---------------------- initialize modular neural network ------------------------ 
		# (update in this order)
		self.ppn = PreprocessingNetwork(self.__hyperparams)
		self.zpg = SequentialCentralPatternGenerator(self.__hyperparams)
		self.bfn = BasisNetwork(self.__hyperparams)
		self.mn = MotorNetwork(self.__hyperparams,
			outputgain=[float(gain) for gain in list((config['MN']['GAIN']).split(","))],
			activation=lambda x: torch.tanh(x))

		premotorparams = deepcopy(self.__hyperparams)
		premotorparams.n_out = self.__n_state
		self.pmn = MotorNetwork(premotorparams,None,activation=lambda x: x)
		with torch.no_grad():
			self.pmn.W *= 0
			self.pmn.W[np.arange(self.__n_state),np.arange(self.__n_state)] += 1

		self.spn = torch.nn.Linear(self.__n_in,self.__n_state,bias=True).to(self.device)
		with torch.no_grad():
			self.spn.weight *= 0
			self.spn.bias *= 0

		valueparams = deepcopy(self.__hyperparams)
		valueparams.n_out = 1
		self.vn = MotorNetwork(valueparams,None,activation=lambda x: x)
		self.vbn = MotorNetwork(valueparams,None,activation=lambda x: x)
		with torch.no_grad():
			self.vbn.W *= 0 
			self.vbn.W += 1
			self.vbn.Wn *= 0 
			self.vbn.Wn += 1

		observationparams = deepcopy(self.__hyperparams)
		observationparams.n_out = self.__n_in
		self.on = MotorNetwork(observationparams,None,activation=lambda x: x)
		self.obn = MotorNetwork(observationparams,None,activation=lambda x: x)
		with torch.no_grad():
			self.obn.W *= 0 
			self.obn.W += 1
			self.obn.Wn *= 0 
			self.obn.Wn += 1

		# ---------------------- initialize neuron activity ------------------------ 
		self.__observation = self.zeros(1,self.__n_in)
		self.__prestateinput = self.zeros(1,self.__n_state)
		self.__stateinput = self.zeros(1,self.__n_state)
		self.__state = self.zeros(1,self.__n_state)
		self.__basis = self.zeros(1,self.__n_state)
		self.__pattern = self.zeros(1,self.__n_state)
		self.outputs = self.zeros(1,self.__n_out)

		# ---------------------- reset modular neural network ------------------------ 
		self.reset()

	# ---------------------- debugging   ------------------------ 

	def get_state(self,torch=False):
		if torch:
			return self.__state
		else:
			return self.__state.detach().cpu().numpy()[0]

	def get_stateinput(self,torch=False):
		if torch:
			return self.__stateinput
		else:
			return self.__stateinput.detach().cpu().numpy()[0]

	def get_prestateinput(self,torch=False):
		if torch:
			return self.__prestateinput
		else:
			return self.__prestateinput.detach().cpu().numpy()[0]

	def get_basis(self,torch=False):
		if torch:
			return self.__basis
		else:
			return self.__basis.detach().cpu().numpy()[0]

	def get_output(self,torch=False):
		if torch:
			return self.outputs
		else:
			return self.outputs.detach().cpu().numpy()[0,:3]

	def explore(self,wnoise,pwnoise):
		self.mn.Wn = self.mn.W + wnoise
		self.pmn.Wn = self.pmn.W + pwnoise

	def zero_grad(self):
		with torch.no_grad():
			self.mn.W.grad = None
			self.mn.Wn.grad = None

	# ---------------------- update   ------------------------

	def reset(self):
		self.__prestateinput = self.zeros(1,self.__n_state)
		self.__stateinput = self.zeros(1,self.__n_state)
		self.__state = self.zeros(1,self.__n_state)
		self.__basis = self.zeros(1,self.__n_state)
		self.outputs = self.zeros(1,self.__n_out)

		self.zpg.reset()
		self.bfn.reset()
		self.mn.reset()
		self.vn.reset()
		self.on.reset()
		self.vbn.reset()
		self.obn.reset()

		for i in range(self.__t_init):
			self.forward(self.__observation)

		while 1: # so that the activity starts around the same state all the time
			self.forward(self.__observation)
			imax = torch.argmax(self.__basis)
			if imax%4 == 0:
				break


	def forward(self,observation):
		self.__observation = observation
		self.__prestateinput = torch.clamp(self.spn(self.__observation),0,1) 
		self.__stateinput = self.ppn(self.__prestateinput,self.__state)
		self.__state = self.zpg(self.__stateinput,self.__basis)
		self.__basis = self.bfn(self.__state)
		self.__pattern = self.pmn(self.__basis)
		self.outputs = self.mn(self.__pattern)

		return self.outputs

	def train_sensornetwork(self,sensoryweights,niteration=50):

		adam = torch.optim.Adam(self.spn.parameters(),lr=0.2)
		criterion = torch.nn.MSELoss()
		targets = torch.eye(self.__n_state,dtype=torch.float).to(self.device)
		for i in range(0,self.__n_state,4):
			targets[i:i+4,i:i+4] = 1

		for i in range(niteration):
			self.spn.zero_grad()
			adam.zero_grad()

			prediction = torch.clamp(self.spn(sensoryweights),0,1)
			
			loss = criterion(prediction,targets)
			loss.backward()
			adam.step()


	







