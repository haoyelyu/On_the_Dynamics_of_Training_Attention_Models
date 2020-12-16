from math import exp as exp
from math import floor as floor
import numpy as np
import os
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from models.model import *
from auxilaries.auxilary_function import *
import configparser
import ast
from tqdm import tqdm
import heapq

def topicEmbNorm(k, non_topic_word_per_sent):
	""" calculate the theoretical SEN relationship

	Args:
	k (float): score
	non_topic_word_per_sent: num of non-topic word per sentence 
			(in SST experiement, this is the avg sentence len of a word minus one)
	
	Assumptions: keys are initialized from zero

	Return: the emb norm in terms of k
	"""
	return (max(0, 2*(k+exp(k)/non_topic_word_per_sent-1/non_topic_word_per_sent))+10e-15)**0.5



def getTopicPurityAndOcc(word, sent_list, topic_list):



	numPos = 0
	numNeg = 0
	total_occurance = 0
	for sent_idx, sent in enumerate(sent_list):
		if word in sent:
			if topic_list[sent_idx] == 0:
				numNeg+=1
			else:
				numPos+=1
			total_occurance +=1
	return abs(numPos/(numNeg+numPos)- numNeg/(numNeg+numPos)), total_occurance


def get_largest_score_word(word, sent_list, topic_list, score_list, word_to_ix):
	""" get the list of words having the largest scores in a sentence containing ``word''


	Args:
	word (string): see the method description aboce. 
	sent_list (list of string lists): the list of sentences in the training set
	topic_list (list of integers): the list of topics in the training set
	score_list (list of floats): the word scores
	word_to_ix (dictionary): word to word_idx mapping

	Return:
	largestScoreWordLstNeg (list of floats): list of words having the largest scores in a  
		negative sentence containing ``word''
	largestScoreWordLstPos (list of floats): list of words having the largest scores in a  
		positive sentence containing ``word''
	"""

	largestScoreWordLstNeg = []
	largestScoreWordLstPos = []

	for sent_idx, sent in enumerate(sent_list):
		if word in sent:
			largestWordTemp = sent[0]
			for aWord in sent:
				if (score_list[word_to_ix[largestWordTemp]] < score_list[word_to_ix[aWord]]):
					largestWordTemp = aWord
			if topic_list[sent_idx] == 0:
				largestScoreWordLstNeg.append(largestWordTemp)
			else:
				largestScoreWordLstPos.append(largestWordTemp)
	return largestScoreWordLstNeg, largestScoreWordLstPos

def get_sent_list_word(word, sent_list, topic_list):
	"""
	Get the list of sentences containing ``word'' as well as the corresponding target topic list

	Args:
	word (string): see the method description above. 
	sent_list (list of string lists): the list of sentences in the training set
	topic_list (list of integers): the list of topics in the training set
	
	Return:
	sent_list_word (list of string lists): list of sentences containing ``word''

	topic_list_word (list of ints): list of the corresponding target topics

	"""

	sent_list_word = []
	topic_list_word = []
	for sent_idx, sent in enumerate(sent_list):
		if word in sent:
			sent_list_word.append(sent)
			topic_list_word.append(topic_list[sent_idx])
	return sent_list_word, topic_list_word



def plot_SEN_curve(model_config_path, dataset_config_path, train_config_path, dataset,\
 topic_embedding_norm_log_lst, topic_score_log, path_to_store_exp_result, model_name):
	
	''' plot experiment SEN curve with the theoretical counterparts (for synthetic data experiment). 

	Args:
	model_config_path (string): the path to the model configuration
	dataset_config_path (string): the path to the dataset configuration
	train_config_path (string): the path to the training configuration
	dataset (dictionary): the dataset on which the model is trained 
	topic_embedding_norm_log_lst (numpy arr): list of the topic word embedding norms
	topic_score_log (numpy arr)): list of the topic word scores
	path_to_store_exp_result (string):  path to store plots
	model_name (string): the name of model shown in the legend.

	'''

	config = configparser.ConfigParser()
	config.read(model_config_path)
	query_var = float(config.get('Model Config', 'queryVar'))
	config.read(dataset_config_path)
	non_topic_word_per_sent = int(config.get('Synthetic Gen', 'numNonTopicWordPerSent'))
	word_to_ix = dataset['word_to_ix']
	topic_word_list = dataset['topic_word_list']
	path_to_store_exp_result = os.path.join(path_to_store_exp_result, 'SEN_curves')
	mkdir(path_to_store_exp_result)
	plt.figure(figsize=(3.5,3))
	for topic_word_idx in range(len(topic_word_list)):
		plt.clf()
		ax = plt.subplot(1, 1, 1)
		plt.ylim(ymax = 5, ymin = 0)
		plt.xlim(xmax = 5, xmin = 0)
		ax.set_xlabel(f'Score ({model_name}) $\sigma^2 = {query_var}$')
		ax.set_ylabel('Topic word embedding norm')
		ax.plot(topic_score_log[word_to_ix[topic_word_list[0]]], topic_embedding_norm_log_lst[word_to_ix[topic_word_list[0]]], linestyle='dotted', linewidth=3, label = 'Experimental')
		ax.plot(topic_score_log[word_to_ix[topic_word_list[0]]], [topicEmbNorm(k, non_topic_word_per_sent) for k in topic_score_log[word_to_ix[topic_word_list[0]]]], linestyle='dotted', linewidth=3, label = 'Theoretical')
		ax.legend()
		path_to_store_img = os.path.join(path_to_store_exp_result, 'NormVsKeyTopicWord'+str(topic_word_idx)+'.pdf')
		plt.savefig(path_to_store_img, bbox_inches='tight')
	plt.close()

def get_avg_sent_length(word, sent_list):
	''' calculate the average sentence length containing ``word''

	Args:
	word (string): see the method description above.
	sent_list (list of string list): the list of the sentences in the trainining dataset.
	

	return:
	1) (float) the avg sentence length containing ``word''
	2) (float) the std of the sentence lengths
	'''
	sentLenLst = []
	for sent in sent_list:
		if word in sent:
			sentLenLst.append(len(sent))
	sentLenLst = np.array(sentLenLst)
	return np.mean(sentLenLst), np.std(sentLenLst)


def plot_SEN_curve_SST(dataset,	path_to_store_exp_result, \
	model_name, nLargest, embedding_norm_log_numpy, score_log_numpy):
	'''	plot experiment SEN curve with the theoretical counterparts (for SST experiment).

	Args:
	dataset (dictionary): the dataset on which the model is trained 
	path_to_store_exp_result (string):  path to store plots
	model_name (string): the name of model shown in the legend.
	nLargest (int): plot SEN curves for the words contatining nLargest scores
	embedding_norm_log_numpy (numpy): word embedding norm list at various epoch iterations.
	score_log_numpy (numpy): word score list at various epoch iterations.

	'''

	configparser.ConfigParser()
	word_to_ix = dataset['word_to_ix']
	word_voc_dict = dataset['word_voc_dict']
	sent_list_train = dataset['sent_list_train']
	path_to_store_exp_result = os.path.join(path_to_store_exp_result, 'SEN_curves')

	# get words with n_largest scores to plot SEN curve
	word_to_trace = [word for word, key in heapq.nlargest(nLargest, \
		[(word, score_log_numpy[-1, word_to_ix[word]]) for word in word_voc_dict], key=lambda x: x[1])]
	
	mkdir(path_to_store_exp_result)

	plt.figure(figsize=(3.5,3))
	for word in tqdm(word_to_trace):
		# calulate the avg length of the sentences containing word
		num_non_topic_word, std = get_avg_sent_length(word, sent_list_train)
		num_non_topic_word = num_non_topic_word-1
		plt.clf()
		ax = plt.subplot(1, 1, 1)
		ax.set_xlabel(f'Score ({model_name})')
		ax.set_ylabel('Topic word embedding norm')
		ax.plot(score_log_numpy[:,word_to_ix[word]], embedding_norm_log_numpy[:,word_to_ix[word]], linestyle='dotted', linewidth=3, label = 'Experimental')
		ax.plot(score_log_numpy[:,word_to_ix[word]], [topicEmbNorm(k, num_non_topic_word) for k in score_log_numpy[:,word_to_ix[word]]], linestyle='dotted', linewidth=3, label = 'Theoretical')
		ax.legend()
		path_to_store_img = os.path.join(path_to_store_exp_result, word+'.pdf')
		plt.savefig(path_to_store_img, bbox_inches='tight')
	plt.close()


def plot_topic_purity_dyn(word_to_trace_list, dataset, \
	path_to_store_exp_result, model_name, score_log_numpy):
	'''	For each word in word_to_trace_list, let L denote the sent list containing the word. 
	For sentences in L, plot the avg purity and the occurence change of the word that has the largest score.

	Args:
	word_to_trace_list (list of strings): see the method description above.
	dataset (dictionary): the dataset on which the model is trained 
	path_to_store_exp_result (string):  path to store plots
	model_name (string): the name of model shown in the legend.
	score_log_numpy (numpy): word score list at various epoch iterations.
	'''
		
	plt.figure()
	plt.figure(figsize=(2.8*5,2))
	sent_list = dataset['sent_list_train']
	topic_list = dataset['topic_list_train']
	word_to_ix = dataset['word_to_ix']

	for word_to_trace_idx, word_to_trace in enumerate(word_to_trace_list):
		plt.style.use('seaborn-deep')
		snapIdxLst = list(range(0,len(score_log_numpy),1))
		avgTopicPurityLst = np.zeros(len(snapIdxLst))
		avgOccPurityLst = np.zeros(len(snapIdxLst))

		for i, snapIdx in enumerate(snapIdxLst):
			sent_list_word, topic_list_word = get_sent_list_word(word_to_trace, sent_list, topic_list)
			largestScoreWordLstNeg, largestScoreWordLstPos = \
				get_largest_score_word(word_to_trace, sent_list_word, topic_list_word, score_log_numpy[snapIdx], word_to_ix)

			topicPurityLst = [getTopicPurityAndOcc(word, sent_list, topic_list)[0] for word in largestScoreWordLstPos+largestScoreWordLstNeg]
			topicOccLst = [getTopicPurityAndOcc(word, sent_list, topic_list)[1]//1e3 for word in largestScoreWordLstPos+largestScoreWordLstNeg]

			avgTopicPurityLst[i] = sum(topicPurityLst)/len(topicPurityLst)
			avgOccPurityLst[i] = sum(topicOccLst)/len(topicOccLst)

		ax1 = plt.subplot(1, 5, word_to_trace_idx+1)
		color = 'tab:red'
		ax1.set_ylim(-0.1,1.1)
		ax1.plot(snapIdxLst, avgTopicPurityLst, color=color)

		if word_to_trace_idx == 0:
			ax1.set_ylabel('Avg. of $\delta(u)$', color=color)
		else:
			ax1.get_yaxis().set_ticks([])
		ax1.set_xlabel(f'x$10^3$ Epoch -- {model_name}' + ' -- $\mathcal{A}_{ '+f'{word_to_trace}'+'}(t)$')
		ax2 = ax1.twinx()
		color = 'tab:blue'
		if word_to_trace_idx == len(word_to_trace_list)-1:
			ax2.set_ylabel('Avg. of occ. (x $10^3$)', color=color)		
		else:
			ax2.get_yaxis().set_ticks([])
		ax2.plot(snapIdxLst, avgOccPurityLst, color=color)
		ax2.set_ylim(-0.1,3.05)
	plt.tight_layout()
	plt.subplots_adjust(wspace=0, hspace=0)
	plt.savefig(path_to_store_exp_result+'/distr.pdf')
	plt.close()


def softmax(scores):
	''' calculate the word weights in a sentence.
	Args:

	scores (list of floats): the word scores in a sentence
	
	return:
	(list of floats): the word weights in a sentence
	'''

	expScores = [exp(score) for score in scores]
	Z = sum(expScores)
	return [expScore/Z for expScore in expScores]


def plot_two_word_in_sent_weight_dyn(sent_idx_to_trace, word_idx_to_trace, dataset, \
	embedding_norm_log_numpy, score_log_numpy, path_to_store_exp_result):
	''' plot the weight changes of two words in a sentence 

	Args:
	sent_idx_to_trace (int): trace sent_idx_to_trace-th sent in the training set 
	word_idx_to_trace (list of ints): the list contain two ints [i, j]. The method trace the i-th and j-th
									word in a sentence.
	dataset (dictionary): the dataset on which the model is trained 
	embedding_norm_log_numpy (numpy): word embedding norm list at various epoch iterations.
	score_log_numpy (numpy): word score list at various epoch iterations. 
	path_to_store_exp_result (string):  path to store plots
	'''

	
	path_to_store_exp_result = os.path.join(path_to_store_exp_result, 'sentWordWeightTrace')
	mkdir(path_to_store_exp_result)
	sent_list = dataset['sent_list_train']
	word_to_ix = dataset['word_to_ix']
	sentence_to_trace = sent_list[sent_idx_to_trace]
	weights = np.zeros((len(score_log_numpy), len(sentence_to_trace)))
	scores = np.zeros((len(score_log_numpy), len(sentence_to_trace)))
	embedding_norms = np.zeros((len(score_log_numpy), len(sentence_to_trace)))

	for snapIdx in range(len(score_log_numpy)):
		scores[snapIdx]  = [score_log_numpy[snapIdx, word_to_ix[word]] \
			for word in sentence_to_trace]
		weights[snapIdx] = softmax(scores[snapIdx])
		embedding_norms[snapIdx] = [embedding_norm_log_numpy[snapIdx, word_to_ix[word]] for word in sentence_to_trace]

	plt.figure()
	plt.figure(figsize=(3.5,2.5))

	plt.plot(range(1,(len(score_log_numpy)+1)), weights[:,word_idx_to_trace[0]],\
	 	label=sentence_to_trace[word_idx_to_trace[0]], linewidth=3, linestyle='dotted')
	plt.plot(range(1,(len(score_log_numpy)+1)), weights[:,word_idx_to_trace[1]],\
		label=sentence_to_trace[word_idx_to_trace[1]], linewidth=3, linestyle='dotted')
	plt.ylabel('Weights')
	plt.xlabel('$x10^3$ Epoch')
	plt.legend()
	plt.savefig(os.path.join(path_to_store_exp_result, str(sent_idx_to_trace)+'_weight_'+ \
		str(sentence_to_trace[word_idx_to_trace[0]])+'_'+sentence_to_trace[word_idx_to_trace[1]]+'.pdf'), \
		bbox_inches='tight', )

	plt.clf()
	plt.plot(range(1,(len(score_log_numpy)+1)), scores[:,word_idx_to_trace[0]], \
		label=sentence_to_trace[word_idx_to_trace[0]], linewidth=3, linestyle='dotted')
	plt.plot(range(1,(len(score_log_numpy)+1)), scores[:,word_idx_to_trace[1]], \
		label=sentence_to_trace[word_idx_to_trace[1]], linewidth=3, linestyle='dotted')
	plt.ylabel('Scores')
	plt.xlabel('$x10^3$ Epoch')
	plt.legend()
	plt.savefig(os.path.join(path_to_store_exp_result, str(sent_idx_to_trace)+'_key_'+ \
		str(sentence_to_trace[word_idx_to_trace[0]])+'_'+sentence_to_trace[word_idx_to_trace[1]]+'.pdf'), \
		bbox_inches='tight', )

	plt.clf()
	plt.plot(range(1,(len(score_log_numpy)+1)), embedding_norms[:,word_idx_to_trace[0]], \
		label=sentence_to_trace[word_idx_to_trace[0]], linewidth=3, linestyle='dotted')
	plt.plot(range(1,(len(score_log_numpy)+1)), embedding_norms[:,word_idx_to_trace[1]], \
		label=sentence_to_trace[word_idx_to_trace[1]], linewidth=3, linestyle='dotted')
	plt.ylabel('Word embedding norms')
	plt.xlabel('$x10^3$ Epoch')
	plt.legend()
	plt.savefig(os.path.join(path_to_store_exp_result, str(sent_idx_to_trace)+\
		'_norm_'+str(sentence_to_trace[word_idx_to_trace[0]])+'_'+sentence_to_trace[word_idx_to_trace[1]]+'.pdf'),\
		 bbox_inches='tight', )



def plot_loss_curve(training_loss_log_lst, path_to_store_exp_result, model_name_lst):
	''' Plot the training loss change in terms of the num of epochs. 

	Args:
	training_loss_log_lst (list of numpy lists): list of numpy lists that record the training loss at various epoch
	path_to_store_exp_result (string): path to store plots
	model_name_lst  (list of strings): lst of model names for plot legends.
	'''

	plt.figure(figsize=(4,2.5))
	plt.clf()
	ax = plt.subplot(1, 1, 1)
	ax.set_xlabel(f'Epoch')
	ax.set_ylabel('Training Loss')
	for training_loss_log, model_name in zip(training_loss_log_lst, model_name_lst):
		ax.plot(np.arange(0, len(training_loss_log), 1), training_loss_log, linestyle='dotted', linewidth=3, label = f'({model_name})')
	ax.legend(loc = 1)
	path_to_store_img = os.path.join(path_to_store_exp_result, 'loss_curves.pdf')
	plt.savefig(path_to_store_img, bbox_inches='tight')
	plt.close()


def plot_emb_norm_curve(dataset, topic_embedding_norm_log_lst, path_to_store_exp_result, model_name_lst):
	''' Plot the embedding norm changes in terms of the num of epochs. 

	Args:
	dataset (dictionary): the dataset on which the model is trained 
	topic_embedding_norm_log_lst (list of numpy lists): list of numpy lists that record the topic word emb norm at various epoch
	path_to_store_exp_result (string): path to store plots	
	model_name_lst  (list of strings): lst of model names for plot legends.
	'''


	word_to_ix = dataset['word_to_ix']
	topic_word_list = dataset['topic_word_list']

	plt.figure(figsize=(4,2.5))
	plt.clf()
	ax = plt.subplot(1, 1, 1)
	ax.set_xlabel(f'Epoch')
	ax.set_ylabel('Topic word embedding norm')

	for topic_embedding_norm_log_lst, model_name in zip(topic_embedding_norm_log_lst, model_name_lst):
		ax.plot(np.arange(0, len(topic_embedding_norm_log_lst[word_to_ix[topic_word_list[0]]]), 1), \
			topic_embedding_norm_log_lst[word_to_ix[topic_word_list[0]]], linestyle='dotted', linewidth=3, label = f'({model_name})')

	ax.legend(loc = 2)
	path_to_store_img = os.path.join(path_to_store_exp_result, 'embedding_norm_epoch.pdf')
	plt.savefig(path_to_store_img, bbox_inches='tight')
	plt.close()

def plot_score_curve(dataset, topic_score_log_lst, path_to_store_exp_result, model_name_lst):
	''' Plot the score changes in terms of the num of epochs. 

	Args:
	dataset (dictionary): the dataset on which the model is trained 
	topic_score_log_lst (list of numpy lists): list of numpy lists that record the topic word score at various epoch
	path_to_store_exp_result (string): path to store plots	
	model_name_lst  (list of strings): lst of model names for plot legends.
	'''

	word_to_ix = dataset['word_to_ix']
	topic_word_list = dataset['topic_word_list']

	plt.figure(figsize=(4,2.5))
	plt.clf()
	ax = plt.subplot(1, 1, 1)
	ax.set_xlabel(f'Epoch')
	ax.set_ylabel('Topic word score')

	for topic_score_log, model_name in zip(topic_score_log_lst, model_name_lst):
		ax.plot(np.arange(0, len(topic_score_log[word_to_ix[topic_word_list[0]]]), 1), \
			topic_score_log[word_to_ix[topic_word_list[0]]], linestyle='dotted', linewidth=3, label = f'({model_name})')

	ax.legend(loc = 2)
	path_to_store_img = os.path.join(path_to_store_exp_result, 'score_epoch.pdf')
	plt.savefig(path_to_store_img, bbox_inches='tight')
	plt.close()


def getDataRange(last_epoch_log_lst, key):
	# auxilary function for plot_emb_norm_and_score_distribution
	xmin = min([min(last_epoch_log[key]) for last_epoch_log in last_epoch_log_lst])
	xmax = max([max(last_epoch_log[key]) for last_epoch_log in last_epoch_log_lst])
	return xmin, xmax

def plot_emb_norm_and_score_distribution(last_epoch_log_lst, model_name_lst, path_to_store_exp_result):
	''' Plot the embedding norm and score distributions

	Args:
	last_epoch_log_lst (dictionary): a dictionary stores the parameters of the model in the last training epoch
	path_to_store_exp_result (string): path to store plots	
	model_name_lst  (list of strings): lst of model names for plot legends.
	'''

	# create dir to save results
	path_to_store_exp_result = os.path.join(path_to_store_exp_result, 'emb_norm_score_distr')
	mkdir(path_to_store_exp_result)

	# Plot non-topic word embedding norm distribution
	plt.figure(figsize=(3.5,3))
	plt.clf()	
	ax = plt.subplot(1, 1, 1)
	ax.set_ylim((0, 1.05))

	xmin, xmax = getDataRange(last_epoch_log_lst, 'non_topic_emb_norm')

	bins = np.linspace(xmin, xmax, 10)

	x_lst = [last_epoch_log['non_topic_emb_norm'] for last_epoch_log in last_epoch_log_lst]
	weights_lst = [np.ones(len(x))/len(x) for x in x_lst]


	plt.hist(x_lst, weights = weights_lst, bins=bins, label = model_name_lst)
	plt.grid(axis='y', alpha=0.75)
	ax.set_xlabel('Non-topic word embedding norm distribution')
	ax.set_ylabel('Frequency')
	ax.legend(loc=1)
	plt.savefig(os.path.join(path_to_store_exp_result, 'NonTopicWordNormDistribution.pdf'), bbox_inches='tight')

	# Plot non-topic word embedding norm distribution
	plt.clf()
	ax = plt.subplot(1, 1, 1)
	ax.set_ylim((0, 1.05))
	xmin, xmax = getDataRange(last_epoch_log_lst, 'topic_emb_norm')
	bins = np.linspace(xmin, xmax, 10)

	x_lst = [last_epoch_log['topic_emb_norm'] for last_epoch_log in last_epoch_log_lst]
	weights_lst = [np.ones(len(x))/len(x) for x in x_lst]
	plt.hist(x_lst, weights = weights_lst, bins=bins, label = model_name_lst)

	plt.grid(axis='y', alpha=0.75)
	ax.set_xlabel('Topic word embedding norm distribution')
	ax.set_ylabel('Frequency')
	ax.legend(loc=2)
	plt.yticks(np.arange(0,1.1,0.2))
	plt.savefig(os.path.join(path_to_store_exp_result, 'TopicWordNormDistribution.pdf'), bbox_inches='tight')


	# Plot non-topic word score distribution
	plt.figure(figsize=(3.5,3))
	plt.clf()	
	ax = plt.subplot(1, 1, 1)
	ax.set_ylim((0, 1.05))

	xmin, xmax = getDataRange(last_epoch_log_lst, 'non_topic_score')
	bins = np.linspace(xmin, xmax, 10)

	x_lst = [last_epoch_log['non_topic_score'] for last_epoch_log in last_epoch_log_lst]
	weights_lst = [np.ones(len(x))/len(x) for x in x_lst]
	plt.hist(x_lst, weights = weights_lst, bins=bins, label = model_name_lst)

	plt.grid(axis='y', alpha=0.75)
	ax.set_xlabel('Non-topic word score distribution')
	ax.set_ylabel('Frequency')
	ax.legend(loc=1)
	plt.yticks(np.arange(0,1.1,0.2))
	plt.savefig(os.path.join(path_to_store_exp_result, 'NonTopicWordScoreDistribution.pdf'), bbox_inches='tight')

	# Plot topic word score distribution
	plt.figure(figsize=(3.5,3))
	plt.clf()	
	ax = plt.subplot(1, 1, 1)
	ax.set_ylim((0, 1.05))

	xmin, xmax = getDataRange(last_epoch_log_lst, 'topic_score')
	bins = np.linspace(xmin, xmax, 10)

	x_lst = [last_epoch_log['topic_score'] for last_epoch_log in last_epoch_log_lst]
	weights_lst = [np.ones(len(x))/len(x) for x in x_lst]
	plt.hist(x_lst, weights = weights_lst, bins=bins, label = model_name_lst)

	plt.grid(axis='y', alpha=0.75)
	ax.set_xlabel('Topic word score distribution')
	ax.set_ylabel('Frequency')
	ax.legend(loc=1)
	plt.yticks(np.arange(0,1.1,0.2))
	plt.savefig(os.path.join(path_to_store_exp_result, 'TopicWordScoreDistribution.pdf'), bbox_inches='tight')