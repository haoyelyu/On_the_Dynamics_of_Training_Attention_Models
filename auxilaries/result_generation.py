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
def topicEmbNorm(k, numRandomWordPerSent):
	"""Assume that keys are initialized from zero"""
	return (max(0, 2*(k+exp(k)/numRandomWordPerSent-1/numRandomWordPerSent))+10e-15)**0.5

def getTopicPurityAndOcc(word, sentList, topicList):
	numPos = 0
	numNeg = 0
	totalOccurance = 0
	for sentIdx, sent in enumerate(sentList):
		if word in sent:
			if topicList[sentIdx] == 0:
				numNeg+=1
			else:
				numPos+=1
			totalOccurance +=1
	return abs(numPos/(numNeg+numPos)-numNeg/(numNeg+numPos)), totalOccurance

def getTheLargestKeyWordLst_word(word, sentList, topicList, keyList, word_to_ix):
	largestScoreWordLstNeg = []
	largestScoreWordLstPos = []

	for sentIdx, sent in enumerate(sentList):
		if word in sent:
			largestWordTemp = sent[0]
			for aWord in sent:
				# if (aWord != word) and (keyList[word_to_ix[largestWordTemp]] < keyList[word_to_ix[aWord]]):
				if (keyList[word_to_ix[largestWordTemp]] < keyList[word_to_ix[aWord]]):
					largestWordTemp = aWord
			if topicList[sentIdx] == 0:
				largestScoreWordLstNeg.append(largestWordTemp)
			else:
				largestScoreWordLstPos.append(largestWordTemp)
	return largestScoreWordLstNeg, largestScoreWordLstPos

def getSentList_word(word, sentList, topicList):
	sentList_word = []
	topicList_word = []
	for sentIdx, sent in enumerate(sentList):
		if word in sent:
			sentList_word.append(sent)
			topicList_word.append(topicList[sentIdx])
	return sentList_word, topicList_word

def plot_SEN_curve(model_config_path, dataset_config_path, train_config_path, dataset,\
 embeddingNormTopicLog, scoreTopicLog, path_to_store_exp_result, model_name):
	config = configparser.ConfigParser()
	config.read(model_config_path)
	queryVar = float(config.get('Model Config', 'queryVar'))
	config.read(dataset_config_path)
	numRandomWordPerSent = int(config.get('Synthetic Gen', 'numRandomWordPerSent'))
	word_to_ix = dataset['word_to_ix']
	topicWordLst = dataset['topicWordLst']
	path_to_store_exp_result = os.path.join(path_to_store_exp_result, 'SEN_curves')
	mkdir(path_to_store_exp_result)
	fig = plt.figure(figsize=(3.5,3))
	for topic_word_idx in range(len(topicWordLst)):
		plt.clf()
		ax = plt.subplot(1, 1, 1)
		plt.ylim(ymax = 5, ymin = 0)
		plt.xlim(xmax = 5, xmin = 0)
		ax.set_xlabel(f'Score ({model_name}) $\sigma^2 = {queryVar}$')
		ax.set_ylabel('Topic word embedding norm')
		ax.plot(scoreTopicLog[word_to_ix[topicWordLst[0]]], embeddingNormTopicLog[word_to_ix[topicWordLst[0]]], linestyle='dotted', linewidth=3, label = 'Experimental')
		ax.plot(scoreTopicLog[word_to_ix[topicWordLst[0]]], [topicEmbNorm(k, numRandomWordPerSent) for k in scoreTopicLog[word_to_ix[topicWordLst[0]]]], linestyle='dotted', linewidth=3, label = 'Theoretical')
		ax.legend()
		path_to_store_img = os.path.join(path_to_store_exp_result, 'NormVsKeyTopicWord'+str(topic_word_idx)+'.pdf')
		plt.savefig(path_to_store_img, bbox_inches='tight')
	plt.close()

def avgSentLength(word, sentList):
	sentLenLst = []
	for sent in sentList:
		if word in sent:
			sentLenLst.append(len(sent))
	sentLenLst = np.array(sentLenLst)
	return np.mean(sentLenLst), np.std(sentLenLst)

def plot_SEN_curve_SST(dataset,	path_to_store_exp_result, \
	model_name, nLargest, embeddingNormLog_Numpy, scoreLog_Numpy):
	config = configparser.ConfigParser()
	word_to_ix = dataset['word_to_ix']
	wordVocDict = dataset['wordVocDict']
	sent_list_train = dataset['sent_list_train']
	path_to_store_exp_result = os.path.join(path_to_store_exp_result, 'SEN_curves')

	# get words with n_largest scores to plot SEN curve
	wordToTrace = [word for word, key in heapq.nlargest(nLargest, \
		[(word, scoreLog_Numpy[-1, word_to_ix[word]]) for word in wordVocDict], key=lambda x: x[1])]
	print(wordToTrace)
	mkdir(path_to_store_exp_result)

	fig = plt.figure(figsize=(3.5,3))
	for word in tqdm(wordToTrace):
		# calulate the avg length of the sentences containing word
		numRandomWord, std = avgSentLength(word, sent_list_train)
		numRandomWord = numRandomWord-1
		plt.clf()
		ax = plt.subplot(1, 1, 1)
		plt.ylim(ymax = 5, ymin = 0)
		plt.xlim(xmax = 5, xmin = 0)
		ax.set_xlabel(f'Score ({model_name})')
		ax.set_ylabel('Topic word embedding norm')
		ax.plot(scoreLog_Numpy[:,word_to_ix[word]], embeddingNormLog_Numpy[:,word_to_ix[word]], linestyle='dotted', linewidth=3, label = 'Experimental')
		ax.plot(scoreLog_Numpy[:,word_to_ix[word]], [topicEmbNorm(k, numRandomWord) for k in scoreLog_Numpy[:,word_to_ix[word]]], linestyle='dotted', linewidth=3, label = 'Theoretical')
		ax.legend()
		path_to_store_img = os.path.join(path_to_store_exp_result, word+'.pdf')
		plt.savefig(path_to_store_img, bbox_inches='tight')
	plt.close()

def plot_topic_purity_dyn(wordToTrace_Lst, dataset, \
	path_to_store_exp_result, model_name, scoreLog_Numpy):
	fig = plt.figure()
	plt.figure(figsize=(2.8*5,2))
	sentList = dataset['sent_list_train']
	topicList = dataset['topic_list_train']
	word_to_ix = dataset['word_to_ix']

	for wordToTrace_idx, wordToTrace in enumerate(wordToTrace_Lst):
		plt.style.use('seaborn-deep')
		snapIdxLst = list(range(0,len(scoreLog_Numpy),1))
		avgTopicPurityLst = np.zeros(len(snapIdxLst))
		avgOccPurityLst = np.zeros(len(snapIdxLst))

		for i, snapIdx in enumerate(snapIdxLst):
			sentList_word, topicList_word = getSentList_word(wordToTrace, sentList, topicList)
			largestScoreWordLstNeg, largestScoreWordLstPos = \
				getTheLargestKeyWordLst_word(wordToTrace, sentList_word, topicList_word, scoreLog_Numpy[snapIdx], word_to_ix)

			topicPurityLst = [getTopicPurityAndOcc(word, sentList, topicList)[0] for word in largestScoreWordLstPos+largestScoreWordLstNeg]
			topicOccLst = [getTopicPurityAndOcc(word, sentList, topicList)[1]//1e3 for word in largestScoreWordLstPos+largestScoreWordLstNeg]

			avgTopicPurityLst[i] = sum(topicPurityLst)/len(topicPurityLst)
			avgOccPurityLst[i] = sum(topicOccLst)/len(topicOccLst)

		ax1 = plt.subplot(1, 5, wordToTrace_idx+1)
		color = 'tab:red'
		ax1.set_ylim(-0.1,1.1)
		ax1.plot(snapIdxLst, avgTopicPurityLst, color=color)

		if wordToTrace_idx == 0:
			ax1.set_ylabel('Avg. of $\delta(u)$', color=color)
		else:
			ax1.get_yaxis().set_ticks([])
			# plt.subplots_adjust(wspace = -1)
		ax1.set_xlabel(f'x$10^3$ Epoch -- {model_name}' + ' -- $\mathcal{A}_{ '+f'{wordToTrace}'+'}(t)$')
		ax2 = ax1.twinx()
		color = 'tab:blue'
		if wordToTrace_idx == len(wordToTrace_Lst)-1:
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
	expScores = [exp(score) for score in scores]
	Z = sum(expScores)
	return [expScore/Z for expScore in expScores]

def plot_two_word_in_sent_weight_dyn(sentIdxForTrace, wordIndToTrace, dataset, \
	embeddingNormLog_Numpy, scoreLog_Numpy, path_to_store_exp_result):
	
	path_to_store_exp_result = os.path.join(path_to_store_exp_result, 'sentWordWeightTrace')
	mkdir(path_to_store_exp_result)
	sentList = dataset['sent_list_train']
	word_to_ix = dataset['word_to_ix']
	sentenceToTrace = sentList[sentIdxForTrace]
	weights = np.zeros((len(scoreLog_Numpy), len(sentenceToTrace)))
	scores = np.zeros((len(scoreLog_Numpy), len(sentenceToTrace)))
	embeddingNorms = np.zeros((len(scoreLog_Numpy), len(sentenceToTrace)))

	for snapIdx in range(len(scoreLog_Numpy)):
		scores[snapIdx]  = [scoreLog_Numpy[snapIdx, word_to_ix[word]] \
			for word in sentenceToTrace]
		weights[snapIdx] = softmax(scores[snapIdx])
		embeddingNorms[snapIdx] = [embeddingNormLog_Numpy[snapIdx, word_to_ix[word]] for word in sentenceToTrace]

	fig = plt.figure()
	plt.figure(figsize=(3.5,2.5))

	plt.plot(range(1,(len(scoreLog_Numpy)+1)), weights[:,wordIndToTrace[0]],\
	 	label=sentenceToTrace[wordIndToTrace[0]], linewidth=3, linestyle='dotted')
	plt.plot(range(1,(len(scoreLog_Numpy)+1)), weights[:,wordIndToTrace[1]],\
		label=sentenceToTrace[wordIndToTrace[1]], linewidth=3, linestyle='dotted')
	plt.ylabel('Weights')
	plt.xlabel('$x10^3$ Epoch')
	plt.legend()
	plt.savefig(os.path.join(path_to_store_exp_result, str(sentIdxForTrace)+'_weight_'+ \
		str(sentenceToTrace[wordIndToTrace[0]])+'_'+sentenceToTrace[wordIndToTrace[1]]+'.pdf'), \
		bbox_inches='tight', )

	plt.clf()
	plt.plot(range(1,(len(scoreLog_Numpy)+1)), scores[:,wordIndToTrace[0]], \
		label=sentenceToTrace[wordIndToTrace[0]], linewidth=3, linestyle='dotted')
	plt.plot(range(1,(len(scoreLog_Numpy)+1)), scores[:,wordIndToTrace[1]], \
		label=sentenceToTrace[wordIndToTrace[1]], linewidth=3, linestyle='dotted')
	plt.ylabel('Scores')
	plt.xlabel('$x10^3$ Epoch')
	plt.legend()
	plt.savefig(os.path.join(path_to_store_exp_result, str(sentIdxForTrace)+'_key_'+ \
		str(sentenceToTrace[wordIndToTrace[0]])+'_'+sentenceToTrace[wordIndToTrace[1]]+'.pdf'), \
		bbox_inches='tight', )

	plt.clf()
	plt.plot(range(1,(len(scoreLog_Numpy)+1)), embeddingNorms[:,wordIndToTrace[0]], \
		label=sentenceToTrace[wordIndToTrace[0]], linewidth=3, linestyle='dotted')
	plt.plot(range(1,(len(scoreLog_Numpy)+1)), embeddingNorms[:,wordIndToTrace[1]], \
		label=sentenceToTrace[wordIndToTrace[1]], linewidth=3, linestyle='dotted')
	plt.ylabel('Word embedding norms')
	plt.xlabel('$x10^3$ Epoch')
	plt.legend()
	plt.savefig(os.path.join(path_to_store_exp_result, str(sentIdxForTrace)+\
		'_norm_'+str(sentenceToTrace[wordIndToTrace[0]])+'_'+sentenceToTrace[wordIndToTrace[1]]+'.pdf'),\
		 bbox_inches='tight', )



def plot_loss_curve(training_loss_log_lst, path_to_store_exp_result, model_name_lst):
	fig = plt.figure(figsize=(4,2.5))
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


def plot_emb_norm_curve(dataset, embeddingNormTopicLog_lst, path_to_store_exp_result, model_name_lst):
	word_to_ix = dataset['word_to_ix']
	topicWordLst = dataset['topicWordLst']

	fig = plt.figure(figsize=(4,2.5))
	plt.clf()
	ax = plt.subplot(1, 1, 1)
	ax.set_xlabel(f'Epoch')
	ax.set_ylabel('Topic word embedding norm')

	for embeddingNormTopicLog, model_name in zip(embeddingNormTopicLog_lst, model_name_lst):
		ax.plot(np.arange(0, len(embeddingNormTopicLog[word_to_ix[topicWordLst[0]]]), 1), \
			embeddingNormTopicLog[word_to_ix[topicWordLst[0]]], linestyle='dotted', linewidth=3, label = f'({model_name})')

	ax.legend(loc = 2)
	path_to_store_img = os.path.join(path_to_store_exp_result, 'embedding_norm_epoch.pdf')
	plt.savefig(path_to_store_img, bbox_inches='tight')
	plt.close()

def plot_score_curve(dataset, scoreTopicLog_lst, path_to_store_exp_result, model_name_lst):
	word_to_ix = dataset['word_to_ix']
	topicWordLst = dataset['topicWordLst']

	fig = plt.figure(figsize=(4,2.5))
	plt.clf()
	ax = plt.subplot(1, 1, 1)
	ax.set_xlabel(f'Epoch')
	ax.set_ylabel('Topic word score')

	for scoreTopicLog, model_name in zip(scoreTopicLog_lst, model_name_lst):
		ax.plot(np.arange(0, len(scoreTopicLog[word_to_ix[topicWordLst[0]]]), 1), \
			scoreTopicLog[word_to_ix[topicWordLst[0]]], linestyle='dotted', linewidth=3, label = f'({model_name})')

	ax.legend(loc = 2)
	path_to_store_img = os.path.join(path_to_store_exp_result, 'score_epoch.pdf')
	plt.savefig(path_to_store_img, bbox_inches='tight')
	plt.close()




def getDataRange(last_epoch_log_lst, key):
	xmin = min([min(last_epoch_log[key]) for last_epoch_log in last_epoch_log_lst])
	xmax = max([max(last_epoch_log[key]) for last_epoch_log in last_epoch_log_lst])
	return xmin, xmax

def plot_emb_norm_and_score_distribution(last_epoch_log_lst, model_name_lst, path_to_store_exp_result):
	# create dir to save results
	path_to_store_exp_result = os.path.join(path_to_store_exp_result, 'emb_norm_score_distr')
	mkdir(path_to_store_exp_result)

	# Plot non-topic word embedding norm distribution
	fig = plt.figure(figsize=(3.5,3))
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
	fig = plt.figure(figsize=(3.5,3))
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
	fig = plt.figure(figsize=(3.5,3))
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














