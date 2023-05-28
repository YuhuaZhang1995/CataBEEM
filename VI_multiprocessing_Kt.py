import numpy as np
import math
import scipy.optimize as opt
from scipy.optimize import minimize
from scipy.special import psi
import sys
import datetime
from multiprocessing import Pool


### Read and summarize the data ###

ct1=datetime.datetime.now()
print(ct1)

filename=sys.argv[1]
infile=open(filename)
next(infile) #skip the first line
input_net=[]
unique_sender=set()
unique_receiver=set()
sender_dict={} #used to record the connection between senders and the receivers
receiver_dict={} 
node_dict={}
topic_ind={} #topic assignment
block_ind={} #block assignment
#block_ind_s={}
#block_ind_r={}
deg_ind={} #degree assignment
interaction_index={} #node:[int1, int2,...]
interaction_sender={} #sender:[int1, int2,...]
interaction_receiver={} #receiver:[int1, int2,...]
interaction_topic={} #interact: topic indicator
line_ct=0
for line in infile:
	s1=line.strip().split("\t")[0]
	s2=line.strip().split("\t")[1]
	Topic=line.strip().split("\t")[2]
	cs1=line.strip().split("\t")[3]
	cs2=line.strip().split("\t")[4]
	
	input_net.append([int(s1),int(s2)])
	unique_sender.add(int(s1))
	unique_receiver.add(int(s2))
	if not int(s1) in sender_dict:
		sender_dict[int(s1)]=[int(s2)]
	else:
		sender_dict[int(s1)].append(int(s2))
	if not int(s2) in receiver_dict:
		receiver_dict[int(s2)]=[int(s1)]
	else:
		receiver_dict[int(s2)].append(int(s1))
	#if not int(s1) in topic_ind:
	#	topic_ind[int(s1)]=int(Topic)
	#if not int(s2) in topic_ind:
	#	topic_ind[int(s2)]=int(Topic)
	if not int(s1) in block_ind:
		block_ind[int(s1)]={int(Topic):int(cs1)}
	else:
		if not int(Topic) in block_ind[int(s1)]:
			block_ind[int(s1)][int(Topic)]=int(cs1)
	if not int(s2) in block_ind:
		block_ind[int(s2)]={int(Topic):int(cs2)}
	else:
		if not int(Topic) in block_ind[int(s2)]:
			block_ind[int(s2)][int(Topic)]=int(cs2)
	#if not int(s1) in block_ind_s:
	#	block_ind_s[int(s1)]={int(cs1)
	#if not int(s2) in block_ind_r:
	#	block_ind_r[int(s2)]=int(cs2)
	if not int(s1) in deg_ind:
		deg_ind[int(s1)]=1
	else:
		deg_ind[int(s1)]+=1
	if not int(s2) in deg_ind:
		deg_ind[int(s2)]=1
	else:
		deg_ind[int(s2)]+=1
	if not int(s1) in interaction_index:
		interaction_index[int(s1)]=[line_ct]
	else:
		interaction_index[int(s1)].append(line_ct)
	if not int(s2) in interaction_index:
		interaction_index[int(s2)]=[line_ct]
	else:
		interaction_index[int(s2)].append(line_ct)
	if not int(s1) in interaction_sender:
		interaction_sender[int(s1)]=[line_ct]
	else:
		interaction_sender[int(s1)].append(line_ct)
	if not int(s2) in interaction_receiver:
		interaction_receiver[int(s2)]=[line_ct]
	else:
		interaction_receiver[int(s2)].append(line_ct)
	interaction_topic[line_ct]=int(Topic)
	line_ct+=1

infile.close()	
topic_ind = dict(sorted(topic_ind.items()))
block_ind = dict(sorted(block_ind.items()))
#block_ind_s = dict(sorted(block_ind_s.items()))
#block_ind_r = dict(sorted(block_ind_r.items()))
deg_ind = dict(sorted(deg_ind.items()))


### Statistics of the data ###
# We need a correspondence between phi_s, phi_r and the sorted nodelist
# We assume the order in nodelist are sorted
nodelist_sort={} #nodelist, sorted
unique_node=set.union(unique_sender,unique_receiver)
unique_node=sorted(list(unique_node))
for i in range(len(unique_node)):
	nodelist_sort[unique_node[i]]=i
#need to transfer set into list so indexing will be easier later
unique_sender=sorted(list(unique_sender))
unique_receiver=sorted(list(unique_receiver))


### Initialize the parameters ###

#introduce sigma to the initialization
sigma=0.2

# Model parameters
K=[4,4] # number of underlying blocks
T=2 # number of topics
B=[np.random.dirichlet(np.ones(K[t]),size=K[t]) for t in range(T)]
for t in range(T):
	for k in range(K[t]):
		for kk in range(K[t]):
			if k==kk:
				B[t][k][kk]=0.9
			else:	
				B[t][k][kk]=0.1/(K[t]-1)

alpha_zero=[np.ones(K[t]) for t in range(T)] #assumed to be fixed,  prior for \pi
tilde_eta=[1]*len(unique_node)

###### Incorporate the LDA prior here ######
#tilde_alpha=[1]*T #prior for \tilde{\pi}
tilde_alpha = np.ones((len(input_net),T)) # dim=n * T
for i in range(len(input_net)):
	tmp_topic_ind=interaction_topic[i]
	for t in range(T):
		if t==tmp_topic_ind:
			tilde_alpha[i][t]=0.001
		else:
			tilde_alpha[i][t]=0.001
############################################

# Latent parameters
# need to make sure sum_K phi = 1
# need to make sure sum_T psi = 1
phi_s = [[np.random.dirichlet(np.ones(K[t]),size=1)[0] for t in range(T)] for i in range(len(unique_sender))] # probability of assigning the node to the first cluster, dim = number of senders * K
phi_r = [[np.random.dirichlet(np.ones(K[t]),size=1)[0] for t in range(T)] for i in range(len(unique_receiver))] # probability of assigning the node to the first cluster, dim = number of receivers * K
# Assumed truth for phi
for i in range(len(unique_sender)):
	#print(nodelist_sort[unique_sender[i]])
	#tmp_block_ind=list(block_ind_s.values())[i]
	#tmp_topic_ind=topic_ind[unique_sender[i]]
	#if unique_sender[i]%4==0 or unique_sender[i]%4==1:
	#	tmp_topic_ind=0
	#else:
	#	tmp_topic_ind=1
	#tmp_block_ind=unique_sender[i]%4
	
	sender=unique_sender[i]
	tmp_block_ind=block_ind[sender]
	for t in range(T):
		if t in tmp_block_ind:
			tmp_tb=tmp_block_ind[t]
			for k in range(K[t]):
				if k==tmp_tb:
					phi_s[i][t][k]=1-sigma
				else:
					phi_s[i][t][k]=0+sigma/(K[t]-1)
		else:
			for k in range(K[t]):
				phi_s[i][t][k]=0.5

			
for i in range(len(unique_receiver)):
	#tmp_block_ind=list(block_ind_r.values())[i]
	#tmp_topic_ind=topic_ind[unique_receiver[i]]
	#if unique_receiver[i]%4==0 or unique_receiver[i]%4==1:
	#	tmp_topic_ind=0
	#else:
	#	tmp_topic_ind=1
	#tmp_block_ind=unique_receiver[i]%4
	
	receiver=unique_receiver[i]
	tmp_block_ind=block_ind[receiver]
	for t in range(T):
		if t in tmp_block_ind:
			tmp_tb=tmp_block_ind[t]
			for k in range(K[t]):
				if k==tmp_tb:
					phi_r[i][t][k]=1-sigma
				else:
					phi_r[i][t][k]=0+sigma/(K[t]-1)
		else:
			for k in range(K[t]):
				phi_r[i][t][k]=0.5


psi_i=np.random.dirichlet(np.ones(T),size=len(input_net)) # probability of assigning the interaction to the first topic, dim = number of interaction * T
# Assumed truth for psi
for i in range(len(input_net)):
	tmp_topic_ind=interaction_topic[i]
	for t in range(T):
		if t==tmp_topic_ind:
			psi_i[i][t]=1-sigma
		else:
			psi_i[i][t]=0+sigma


eta = [np.ones(K[t]) for t in range(T)] # dim = T*K 
lambda_i = [[np.random.dirichlet(np.ones(K[t]),size=1)[0] for t in range(T)] for i in range(len(unique_node))] #dim = number of nodes * T * K
#Assumed truth for lambda
for i in range(len(input_net)):
	sender=input_net[i][0]
	receiver=input_net[i][1]
	index_s=unique_sender.index(sender)
	index_r=unique_receiver.index(receiver)
	node_index_s=unique_node.index(sender)
	node_index_r=unique_node.index(receiver)
	for t in range(T):
		for k in range(K[t]):
			lambda_i[node_index_s][t][k]+=psi_i[i][t]*phi_s[index_s][t][k]
			lambda_i[node_index_r][t][k]+=psi_i[i][t]*phi_r[index_r][t][k]

##### Incorporate LDA prior here #####
#gamma=[1]*T # dim = T
gamma= np.random.dirichlet(np.ones(T),size=len(input_net)) # dim=n * T
for i in range(len(input_net)):
	for t in range(T):
		gamma[i][t]=tilde_alpha[i][t]+psi_i[i][t]
######################################

ct2=datetime.datetime.now()
#print(ct2)

#############Variational EM algorithm############

def update_psi(ix):
	sender=input_net[ix][0]
	receiver=input_net[ix][1]
	index_s=unique_sender.index(sender)
	index_r=unique_receiver.index(receiver)
	node_index_s=unique_node.index(sender)
	node_index_r=unique_node.index(receiver)
	tmp=[]
	for t in range(T):
		tmp_t=0
		tmp_t+=psi(gamma[ix][t])-psi(sum(gamma[ix]))
		for k in range(K[t]):
			tmp_t+=phi_s[index_s][t][k]*(psi(eta[t][k])-psi(sum([eta[t][x] for x in range(K[t])])))
			tmp_t+=phi_s[index_s][t][k]*(psi(lambda_i[node_index_s][t][k])-sum_lambda_i[t][k])
			tmp_t+=phi_r[index_r][t][k]*(psi(lambda_i[node_index_r][t][k])-sum_lambda_i[t][k])
			for kk in range(K[t]):
				tmp_t+=np.log(B[t][k][kk])*phi_r[index_r][t][kk]*phi_s[index_s][t][k]
		tmp.append(tmp_t)
	#In case probability with 0
	tmp=[float(i)-max(tmp) for i in tmp]
	tmp=[math.exp(i) for i in tmp]
	tmp=[float(i)/sum(tmp) for i in tmp]
	return tmp

def update_phi_s(ix):
	#tmp_phi_s=np.random.dirichlet(np.ones(K),size=T)
	#sender=unique_sender[ix]
	#int_ct=interaction_sender[sender]
	#for t in range(T):
	#	tmp=[]
	#	for k in range(K):
	#		tmp_k=0
	#		r_ct=0 #index for receivers of sender
	#		for int_ctt in int_ct:
	#			tmp_psi=psi_i[int_ctt][t]
	#			tmp_k+=tmp_psi*(psi(eta[t][k])-psi(sum(eta[t])))
	#			tmp_lambda=lambda_i[nodelist_sort[sender]][t][k]
	#			tmp_k+=tmp_psi*(psi(tmp_lambda)-sum_lambda_i[t][k])
	#			receiver=sender_dict[sender][r_ct]
	#			r_ct+=1
	#			for kk in range(K):
	#				tmp_k+=tmp_psi*np.log(B[t][k][kk])*phi_r[unique_receiver.index(receiver)][t][kk]
	#		tmp.append(tmp_k)
	#	#In case probability with 0
	#	tmp=[float(i)-max(tmp) for i in tmp]
	#	tmp=[math.exp(i) for i in tmp]
	#	tmp=[float(i)/sum(tmp) for i in tmp]
	#	tmp_phi_s[t]=tmp
	#return tmp_phi_s
	tmp_phi_s=[np.random.dirichlet(np.ones(K[t]),size=1) for t in range(T)]
	for t in range(T):
		tmp=phi_s[ix][t]
		tmp=[float(i)-max(tmp) for i in tmp]
		tmp=[math.exp(i) for i in tmp]
		tmp=[float(i)/sum(tmp) for i in tmp]
		tmp_phi_s[t]=tmp
	return tmp_phi_s


def update_phi_r(ix):
	#tmp_phi_r=np.random.dirichlet(np.ones(K),size=T)
	#receiver=unique_receiver[ix]
	#int_ct=interaction_receiver[receiver]
	#for t in range(T):
	#	tmp=[]
	#	for k in range(K):
	#		tmp_k=0
	#		s_ct=0
	#		for int_ctt in int_ct:
	#			tmp_psi=psi_i[int_ctt][t]
	#			tmp_lambda=lambda_i[nodelist_sort[receiver]][t][k]
	#			tmp_k+=tmp_psi*(psi(tmp_lambda)-sum_lambda_i[t][k])
	#			sender=receiver_dict[receiver][s_ct]
	#			s_ct+=1
	#			for kk in range(K):
	#				tmp_k+=tmp_psi*np.log(B[t][kk][k])*phi_s[unique_sender.index(sender)][t][kk]
	#		tmp.append(tmp_k)
	#	#In case probability with 0
	#	tmp=[float(i)-max(tmp) for i in tmp]
	#	tmp=[math.exp(i) for i in tmp]
	#	tmp=[float(i)/sum(tmp) for i in tmp]
	#	tmp_phi_r[t]=tmp
	#return tmp_phi_r
	tmp_phi_r=[np.random.dirichlet(np.ones(K[t]),size=1) for t in range(T)]
	for t in range(T):
		tmp=phi_r[ix][t]
		tmp=[float(i)-max(tmp) for i in tmp]
		tmp=[math.exp(i) for i in tmp]
		tmp=[float(i)/sum(tmp) for i in tmp]
		tmp_phi_r[t]=tmp
	return tmp_phi_r
		

def update_phi(x_index_list):
	sub_phi_s=[[np.zeros(K[t],dtype=np.float32) for t in range(T)] for i in range(len(unique_sender))]
	sub_phi_r=[[np.zeros(K[t],dtype=np.float32) for t in range(T)] for i in range(len(unique_receiver))]
	for i in x_index_list:
		sender=input_net[i][0]
		receiver=input_net[i][1]
		node_index_s=nodelist_sort[sender]
		node_index_r=nodelist_sort[receiver]
		index_s=unique_sender.index(sender)
		index_r=unique_receiver.index(receiver)
		for t in range(T):
			tmp_s=[]
			tmp_r=[]
			for k in range(K[t]):
				tmp_k_s=0
				tmp_k_r=0
				tmp_psi=psi_i[i][t]
				tmp_k_s+=tmp_psi*(psi(eta[t][k])-psi(sum(eta[t])))
				tmp_lambda_s=lambda_i[nodelist_sort[sender]][t][k]
				tmp_lambda_r=lambda_i[nodelist_sort[receiver]][t][k]
				tmp_k_s+=tmp_psi*(psi(tmp_lambda_s)-sum_lambda_i[t][k])
				tmp_k_r+=tmp_psi*(psi(tmp_lambda_r)-sum_lambda_i[t][k])
				for kk in range(K[t]):
					tmp_k_s+=tmp_psi*np.log(B[t][k][kk])*phi_r[unique_receiver.index(receiver)][t][kk]
					tmp_k_r+=tmp_psi*np.log(B[t][k][kk])*phi_s[unique_sender.index(sender)][t][kk]
				tmp_s.append(tmp_k_s)
				tmp_r.append(tmp_k_r)
			#In case probability with 0
			#tmp=[float(i)-max(tmp) for i in tmp]
			#tmp=[math.exp(i) for i in tmp]
			#tmp=[float(i)/sum(tmp) for i in tmp]
			sub_phi_s[index_s][t]+=tmp_s
			sub_phi_r[index_r][t]+=tmp_r
	return [sub_phi_s,sub_phi_r]
		


def update_lambda_eta_B(x_index_list): 
	sub_lambda_i = [[np.zeros(K[t],dtype=np.float32) for t in range(T)] for i in range(len(unique_node))]
	sub_eta = [np.zeros(K[t],dtype=np.float32) for t in range(T)]
	sub_B=[np.zeros((K[t], K[t]), dtype=np.float32) for t in range(T)]
	for i in x_index_list:
		sender=input_net[i][0]
		receiver=input_net[i][1]
		node_index_s=nodelist_sort[sender]
		node_index_r=nodelist_sort[receiver]
		index_s=unique_sender.index(sender)
		index_r=unique_receiver.index(receiver)
		
		for t in range(T):
			for k in range(K[t]):
				sub_lambda_i[node_index_s][t][k]+=psi_i[i][t]*phi_s[index_s][t][k]
				sub_lambda_i[node_index_r][t][k]+=psi_i[i][t]*phi_r[index_r][t][k]
				sub_eta[t][k]+=psi_i[i][t]*phi_s[index_s][t][k]
				for kk in range(K[t]):
					tmp_psi=psi_i[i][t]
					sub_B[t][k][kk]+=tmp_psi*(phi_s[unique_sender.index(sender)][t][k])*(phi_r[unique_receiver.index(receiver)][t][kk])
	return [sub_lambda_i, sub_eta, sub_B]

def log_llk(x_index_list):
	sub_a=0
	sub_b=0
	sub_c=0
	sub_d=0
	for i in x_index_list:
		sender=input_net[i][0]
		receiver=input_net[i][1]
		for t in range(T):
			tmp_psi=psi_i[i][t]			
			sub_a+=psi_i[i][t]*(psi(gamma[i][t])-psi(sum(gamma[i])))
			for k in range(K[t]):
				sub_b+=psi_i[i][t]*phi_s[unique_sender.index(sender)][t][k]*(psi(eta[t][k])-psi(sum([eta[t][x] for x in range(K[t])])))
				for kk in range(K[t]):
					sub_c+=psi_i[i][t]*phi_s[unique_sender.index(sender)][t][k]*(psi(lambda_i[nodelist_sort[sender]][t][k])-sum_lambda_i[t][k])
					sub_c+=psi_i[i][t]*phi_r[unique_receiver.index(receiver)][t][kk]*(psi(lambda_i[nodelist_sort[receiver]][t][kk])-sum_lambda_i[t][kk])
					tmp_phi_s=phi_s[unique_sender.index(sender)][t][k]
					tmp_phi_r=phi_r[unique_receiver.index(receiver)][t][kk]
					sub_d+=psi_i[i][t]*tmp_phi_s*tmp_phi_r*math.log(B[t][k][kk])
	return sub_a+sub_b+sub_c+sub_d

################### Initialization ##################
sum_lambda_i=[np.random.dirichlet(np.ones(K[t]),size=1)[0] for t in range(T)]
sum_psi_lambda_i=[np.random.dirichlet(np.ones(K[t]),size=1)[0] for t in range(T)]
sum_lambda=[np.random.dirichlet(np.ones(K[t]),size=1)[0] for t in range(T)]
for t in range(T):
	for k in range(K[t]):
		sum_lambda_i[t][k]=psi(sum([lambda_i[i][t][k] for i in range(len(lambda_i))]))
		sum_psi_lambda_i[t][k]=sum([psi(lambda_i[i][t][k]) for i in range(len(lambda_i))])
		sum_lambda[t][k]=sum([(lambda_i[i][t][k]-1)*(psi(lambda_i[i][t][k])-sum_lambda_i[t][k]) for i in range(len(lambda_i))])

old_llk=float('inf')
new_llk=0.0
iter_ind=1.0

############## Beginning of iteration ################
while iter_ind>1e-6:
	#print(B)

	#In case elements in B exactly equal to 0
	for t in range(T):
		for k in range(K[t]):
			for kk in range(K[t]):
				if B[t][k][kk]<1e-10:
					B[t][k][kk]=1e-10
	ct1=datetime.datetime.now()
	print(ct1)
	
	# Update psi
	numprocesses=100
	pool = Pool(processes=numprocesses)
	psi_i = pool.map(update_psi, np.arange(len(psi_i)))
	pool.close()
	pool.join()
	
	ct2=datetime.datetime.now()
	print(ct2-ct1)
	
	#update phi_s and phi_r
	pool = Pool(processes=numprocesses)
	index_split = np.array_split(np.arange(len(input_net)), numprocesses)
	pool_output_list = pool.map(update_phi, index_split)
	pool.close()
	pool.join()

	for iprocess in range(numprocesses):
		if iprocess==0:
			phi_s=pool_output_list[iprocess][0]
			phi_r=pool_output_list[iprocess][1]
		else:
			phi_s=np.add(phi_s,pool_output_list[iprocess][0])
			phi_r=np.add(phi_r,pool_output_list[iprocess][1])
	
	pool = Pool(processes=numprocesses)
	phi_s = pool.map(update_phi_s, np.arange(len(phi_s)))
	pool.close()
	pool.join()

	pool = Pool(processes=numprocesses)
	phi_r = pool.map(update_phi_r, np.arange(len(phi_r)))
	pool.close()
	pool.join()
	
	
	ct4=datetime.datetime.now()
	print(ct4-ct2)

	# Update eta
	for t in range(T):	
		for k in range(K[t]):
			eta[t][k]=alpha_zero[t][k]	
	
	# Update lambda and gamma
	for t in range(T):
		for k in range(K[t]):
			for i in range(len(unique_node)):
				lambda_i[i][t][k]=tilde_eta[i]
	# Update B
	for t in range(T):
		for k in range(K[t]):
			for kk in range(K[t]):
				B[t][k][kk]=0

	#ct4=datetime.datetime.now()
	#print(ct4-ct3)
	
	#Update gamma, lambda_i, eta, B
	gamma=tilde_alpha+psi_i	
	pool = Pool(processes=numprocesses)
	index_split = np.array_split(np.arange(len(input_net)), numprocesses)
	pool_output_list = pool.map(update_lambda_eta_B, index_split)
	pool.close()
	pool.join()
	
	for iprocess in range(numprocesses):
		lambda_i = np.add(lambda_i,pool_output_list[iprocess][0])
		eta += pool_output_list[iprocess][1]
		B=np.add(B,pool_output_list[iprocess][2])
	
	for t in range(T):
		for k in range(K[t]):
			tmp=sum(B[t][k])
			B[t][k]=B[t][k]/tmp
	
	ct5=datetime.datetime.now()
	print(ct5-ct4)
	
	#calculate the summation over lambda_i
	for t in range(T):
		for k in range(K[t]):
			sum_lambda_i[t][k]=psi(sum([lambda_i[i][t][k] for i in range(len(lambda_i))]))
			sum_psi_lambda_i[t][k]=sum([psi(lambda_i[i][t][k]) for i in range(len(lambda_i))])
			sum_lambda[t][k]=sum([(lambda_i[i][t][k]-1)*(psi(lambda_i[i][t][k])-sum_lambda_i[t][k]) for i in range(len(lambda_i))])
	
	ct6=datetime.datetime.now()
	print(ct6-ct5)
	
	##### calculate log likelihood ######
	pool = Pool(processes=numprocesses)
	index_split = np.array_split(np.arange(len(input_net)), numprocesses)
	pool_output_list = pool.map(log_llk, index_split)
	pool.close()
	pool.join()
	
	llk_output=sum(pool_output_list)
	old_llk=new_llk
	new_llk=llk_output
	print(new_llk)
	iter_ind=abs((new_llk-old_llk)/new_llk)

	ct7=datetime.datetime.now()
	print(ct7-ct6)
	print(ct7-ct1)		


ct3=datetime.datetime.now()
filename_phi_s=sys.argv[2]
filename_phi_r=sys.argv[3]
filename_psi=sys.argv[4]
filename_lambda=sys.argv[5]

myfile=open(filename_phi_s,"a")
for i in range(len(unique_sender)):
	tmp="["+str(unique_sender[i])+", "+str(phi_s[i])+"]"+"\n"
	myfile.write(tmp)
myfile.close()

myfile1=open(filename_phi_r,"a")
for i in range(len(unique_receiver)):
	tmp="["+str(unique_receiver[i])+", "+str(phi_r[i])+"]"+"\n"
	myfile1.write(tmp)
myfile1.close()

myfile2=open(filename_psi,"a")
for i in range(len(input_net)):
	myfile2.write(str(i)+"\n")
	myfile2.write("["+str(psi_i[i])+"\n")
myfile2.close()

myfile3=open(filename_lambda,"a")
for i in range(len(unique_node)):
	myfile3.write(str(unique_node[i])+"\n")
	myfile3.write(str(lambda_i[i])+"\n")
myfile3.close()

print(new_llk)
print(B)





