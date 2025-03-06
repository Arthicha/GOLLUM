
# ------------------- import modules ---------------------

# standard modules
import time, sys, os
from copy import deepcopy
import configparser

# math-related modules
import numpy as np # cpu array
import torch # cpu & gpu array

# simulation
from interface.vrepinterfaze import VrepInterfaze

# control
from network.SME import SequentialMotionExecutor
from optimizer.agol import AddedGradientOnlineLearning
from optimizer.gd import GradientDescent
from utils.utils import TorchReplay as Replay
from utils.utils import numpy, tensor, compute_return, compute_uncertainty

# visualization
import matplotlib.pyplot as plt

# ------------------- config variables ---------------------

NREPLAY = 8
NTIMESTEP = 20
NEPISODE = 1000
RESET = False
CONNECTION = torch.FloatTensor(np.zeros((4,4))).cuda()
CONNECTION[[0,1,2,3],[1,2,3,0]] = 1
NIN = CONNECTION.shape[0]
# ------------------- auxiliary functions ---------------------


sme = None # SME network
mnagol = pmnagol = None # mnagol learning algorithm
vgd = vbgd = ogd = obgd = None # target fitting learning algorithm
reward_replay = pregrad_replay = grad_replay = preweight_replay = None # experience replay
weight_replay = basis_replay = observation_replay = None # another experience replay



def create_network():
	global sme 
	global mnagol, pmnagol 
	global vgd, vbgd, ogd, obgd
	global reward_replay, grad_replay, weight_replay, basis_replay, observation_replay
	global preweight_replay, pregrad_replay
	global CONNECTION

	# initiliaze SME network
	sme = SequentialMotionExecutor('config/network.ini',CONNECTION)
	

	# initialize mnagol learning algorithm
	mnagol = AddedGradientOnlineLearning(sme.mn.W,'config/mn_optim.ini')
	mnagol.attach_returnfunction(compute_return) # set return function
	mnagol.attach_valuenet(sme.vn) # set value network (remove this if you want to use average baseline)

	pmnagol = AddedGradientOnlineLearning(sme.pmn.W,'config/pmn_optim.ini')
	pmnagol.attach_returnfunction(compute_return) # set return function
	pmnagol.attach_valuenet(sme.vn) # set value network (remove this if you want to use average baseline)

	# initialzie GD learning algorithm for baseline estimation
	vgd = GradientDescent(sme.vn.W,'config/vn_optim.ini')
	vgd.attach_returnfunction(compute_return)  # set return function
	vgd.attach_valuenet(sme.vn)
	vgd.set_verbose(False) # True for printing the loss

	# initialzie GD learning algorithm for minimum value estimation
	vbgd = GradientDescent(sme.vbn.W,'config/vbn_optim.ini')
	vbgd.attach_returnfunction(lambda x: x)  # set return function
	vbgd.attach_valuenet(sme.vbn)
	vbgd.set_verbose(False) # True for printing the loss

	# initialzie GD learning algorithm for baseline estimation
	ogd = GradientDescent(sme.on.W,'config/on_optim.ini')
	ogd.attach_returnfunction(lambda x:x)  # set return function
	ogd.attach_valuenet(sme.on)
	ogd.set_verbose(False) # True for printing the loss

	# initialzie GD learning algorithm for minimum value estimation
	obgd = GradientDescent(sme.obn.W,'config/obn_optim.ini')
	obgd.attach_returnfunction(lambda x: x)  # set return function
	obgd.attach_valuenet(sme.obn)
	obgd.set_verbose(False) # True for printing the loss

	# initialize experience replay
	reward_replay = Replay(NREPLAY,shape=(NTIMESTEP,1,1))
	pregrad_replay = Replay(NREPLAY,shape= (NTIMESTEP,CONNECTION.shape[0],CONNECTION.shape[0]))
	grad_replay = Replay(NREPLAY,shape= (NTIMESTEP,CONNECTION.shape[0],18))
	preweight_replay = Replay(NREPLAY,shape=(1,CONNECTION.shape[0],CONNECTION.shape[0]))
	weight_replay = Replay(NREPLAY,shape=(1,CONNECTION.shape[0],18))
	basis_replay = Replay(NREPLAY,shape=(NTIMESTEP,1,CONNECTION.shape[0]))
	observation_replay = Replay(NREPLAY,shape=(NTIMESTEP,1,1))


def save(filename):
	global sme, mnagol, pmnagol
	global CONNECTION

	data = {}
	data['connection'] = CONNECTION
	data['mn_weight'] = sme.mn.W.detach()
	data['vn_weight'] = sme.vn.W.detach()
	data['on_weight'] = sme.on.W.detach()
	data['vbn_weight'] = sme.vbn.W.detach()
	data['obn_weight'] = sme.obn.W.detach()
	data['sigma'] = mnagol.get_sigma()
	data['pmn_sigma'] = pmnagol.get_sigma()
	torch.save(data,filename)

def load(filename,loadconection=True):
	global sme 
	global mnagol, pmnagol
	global CONNECTION

	data = torch.load(filename,map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu') )
	if loadconection:
		CONNECTION = data['connection']
	create_network()

	modules = [sme.mn,sme.vn,sme.on,sme.vbn,sme.obn]
	names = ['mn_weight','vn_weight','on_weight','vbn_weight','obn_weight']
	with torch.no_grad():
		for name, module in zip(names,modules):
			module.W[:data[name].shape[0]] *= 0
			module.W[:data[name].shape[0]] += data[name].detach()
		mnagol.set_sigma(data['sigma'])
		try:
			pmnagol.set_sigma(data['pmn_sigma'])
		except:
			pass

	for module in modules:
		module.apply_noise(0)



# ------------------- setup ---------------------
create_network()


# initialize simulation interface
vrep = VrepInterfaze()

observation = tensor(np.zeros((1,1)))
observation[0,0] = -1

with torch.no_grad():
	sme.on.W += observation
sme.on.apply_noise(0)
sme.train_sensornetwork(sme.on.W)
# ------------------- start locomotion learning ---------------------
for i in range(NEPISODE):
	print('episode',i)

	# ------------------- episode-wise reset ---------------------
	observation[0,0] *= -1 if i%300 == 0 else 1
	output = sme.forward(observation)

	if RESET: # reset the simulation/network
		vrep.reset()
		sme.reset()
	prepose = vrep.get_robot_pose()
	
	# ------------------- episode-wise setup ---------------------
	
	# preprocess mn exploration
	mn_noise = mnagol.wnoise()
	idx = torch.argmax(sme.get_basis(torch=True))//4
	mask = tensor(np.zeros((CONNECTION.shape[0],18)))
	mask[4*idx:4*idx+4,:] = 1
	mn_noise *= mask

	# preprocess pmn exploration
	pmn_noise = pmnagol.wnoise()
	mask = np.zeros((CONNECTION.shape[0],CONNECTION.shape[0]))+1
	for k in range(CONNECTION.shape[0]//4):
		mask[4*k:4*k+4,4*k:4*k+4] *= 0
	pmn_noise *= tensor(mask)

	# perform exploration
	sme.explore(mn_noise,pmn_noise)
	weight_replay.add(sme.mn.Wn)
	preweight_replay.add(sme.pmn.Wn)
	
	for t in range(NTIMESTEP):

		# update network
		output = sme.forward(observation)
		basis = sme.get_basis(torch=True)

		# update environment
		joint_command = output.clone()
		joint_command[:,[0,3,6,9,12,15]] = torch.clamp(joint_command[:,[0,3,6,9,12,15]],-1,1)
		vrep.set_robot_joint(numpy(joint_command))
		vrep.update()

		# compute reward
		pose = vrep.get_robot_pose()
		dx = pose[0]-prepose[0]
		dy = pose[1]-prepose[1]
		reward = observation[0,0].item()*(dx*np.cos(prepose[-1]) + dy*np.sin(prepose[-1])) 
		#reward = observation[0,0].item()*(pose[0]-prepose[0])
		prepose = deepcopy(pose)

		# backpropagate output gradient
		sme.zero_grad()
		torch.sum(output).backward() 

		# append experience replay
		reward_replay.add(tensor([reward]).unsqueeze(0))
		basis_replay.add(basis)
		pregrad_replay.add(sme.pmn.W.grad)
		grad_replay.add(sme.mn.W.grad)
		observation_replay.add(observation)

	

	print('\t episodic reward',torch.sum(reward_replay.data()[-1]).item())
	print('\t obs:', torch.mean(observation_replay.data()[-1,:,0,0]).item())

	# ------------------- new condition detection ---------------------
	v_delta = torch.mean(compute_return(reward_replay.data()[[-1]])-compute_return(sme.vn(basis_replay.data()[[-1]])))
	v_thresh = -torch.clamp(torch.mean(sme.vbn(basis_replay.data()[[-1]])),0.01,None)

	o_delta = torch.abs(torch.mean(observation_replay.data()[[-1]]-sme.on(basis_replay.data()[[-1]])))
	o_thresh = torch.clamp(torch.mean(sme.obn(basis_replay.data()[[-1]])),0.01,None)
	print(v_delta, v_thresh)
	if (v_delta > v_thresh) or torch.any(o_delta < o_thresh): 
		# ------------------- update the network if being in the same condition ---------------------
		vgd.update(basis_replay.data(),reward_replay.data())
		ogd.update(basis_replay.data(),observation_replay.data())
		vbgd.update(basis_replay.data(),compute_uncertainty(reward_replay.data(),compute_return(sme.vn(basis_replay.data()))).detach())
		obgd.update(basis_replay.data(),compute_uncertainty(observation_replay.data(),sme.on(basis_replay.data())).detach())
		mnagol.update(basis_replay.data(),weight_replay.data(),reward_replay.data(),grad_replay.data())
		pmnagol.update(basis_replay.data(),preweight_replay.data(),reward_replay.data(),pregrad_replay.data())
	else:
		# ------------------- create new subnetwork if experiencing new condition ---------------------
		save('config/temp.pt')
		sub_i = 4*(torch.argmax(basis_replay.data()[-1,-1,0])//4)
		sub_j = CONNECTION.shape[0]
		newconnection = torch.FloatTensor(np.zeros((CONNECTION.shape[0]+4,CONNECTION.shape[0]+4))).cuda()
		newconnection[:CONNECTION.shape[0],:CONNECTION.shape[0]] = CONNECTION
		newconnection[[sub_j,sub_j+1,sub_j+2,sub_j+3],[sub_j+1,sub_j+2,sub_j+3,sub_j+0]] = 1
		newconnection[[sub_i,sub_j+2],[sub_j,sub_i+2]] = 1
		CONNECTION = newconnection
		create_network()
		load('config/temp.pt',loadconection=False)
		with torch.no_grad():
			sme.on.W[CONNECTION.shape[0]-4:,:] += observation
			presensor = sme.on.W 
		sme.on.apply_noise(0)
		sme.train_sensornetwork(presensor)
		save('config/temp.pt')


