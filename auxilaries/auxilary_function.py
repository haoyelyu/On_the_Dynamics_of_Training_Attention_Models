from math import exp as exp
from math import floor as floor
import numpy as np
import copy
import csv
import os
import configparser
import ast
import re
from sklearn.metrics import accuracy_score
from data.synthetic_data_generation.data_gen import *
from models.model import *
import shutil

def mkdir(dir_path):
	if os.path.exists(dir_path):
		shutil.rmtree(dir_path)
	os.makedirs(dir_path)

def eucleadan_norm(vec):
	return vec.pow(2).sum().pow(0.5).tolist()

class Para_Recorder:
	def __init__(self, num_cache, model, dataset, record_period):
		self.model = model
		self.num_cache = num_cache
		self.record_period = record_period
		self.embeddingNormTopicLog = np.zeros((len(dataset['topicWordLst']), num_cache))
		self.scoreTopicLog = np.zeros((len(dataset['topicWordLst']), num_cache))
		self.trainLossLog = np.zeros(num_cache,)
		self.record_idx = 0
		self.dataset = dataset

	def record(self, train_loss):
		embeddingWeight = None
		keyWeight = None
		queryWeight = None
		for name, parameter in self.model.named_parameters():
			if (name == 'keys.keys.weight'):
				keyWeight = copy.deepcopy(parameter.data)
			if (name == 'embeddings.embeddings.weight'):
				embeddingWeight = copy.deepcopy(parameter.data)
			if (name == 'attn.queries'):
				queryWeight = copy.deepcopy(parameter.data)


		if self.record_idx >= self.num_cache:
			print('Recorder is full!!')
		else:
			for topicInd, topic in enumerate(self.dataset['topicWordLst']):
				self.embeddingNormTopicLog[topicInd, self.record_idx] = \
					eucleadan_norm(embeddingWeight[self.dataset['word_to_ix'][topic]])*eucleadan_norm(queryWeight)
				self.scoreTopicLog[topicInd, self.record_idx] = \
					queryWeight.mm(keyWeight[self.dataset['word_to_ix'][topic]].view(-1, 1)).tolist()[0][0]
				self.trainLossLog[self.record_idx] = train_loss
			self.record_idx += 1

	def get_record(self):
		for name, parameter in self.model.named_parameters():
			if (name == 'keys.keys.weight'):
				keyWeight = copy.deepcopy(parameter.data)
			if (name == 'embeddings.embeddings.weight'):
				embeddingWeight = copy.deepcopy(parameter.data)
			if (name == 'attn.queries'):
				queryWeight = copy.deepcopy(parameter.data)

		non_topic_score_lastEpoch = np.zeros(len(self.dataset['randomWordDict']))
		non_topic_emb_norm_lastEpoch = np.zeros(len(self.dataset['randomWordDict']))
		topic_score_lastEpoch = np.zeros(len(self.dataset['topicWordLst']))
		topic_emb_norm_lastEpoch = np.zeros(len(self.dataset['topicWordLst']))

		for non_topic_word_idx, non_topic_word in enumerate(self.dataset['randomWordDict']):
			non_topic_emb_norm_lastEpoch[non_topic_word_idx] = \
				eucleadan_norm(embeddingWeight[self.dataset['word_to_ix'][non_topic_word]])*eucleadan_norm(queryWeight)
			non_topic_score_lastEpoch[non_topic_word_idx] = \
				queryWeight.mm(keyWeight[self.dataset['word_to_ix'][non_topic_word]].view(-1, 1)).tolist()[0][0]

		for topic_word_idx, topic_word in enumerate(self.dataset['topicWordLst']):
			topic_emb_norm_lastEpoch[topic_word_idx] = \
				eucleadan_norm(embeddingWeight[self.dataset['word_to_ix'][topic_word]])*eucleadan_norm(queryWeight)
			topic_score_lastEpoch[topic_word_idx] = \
				queryWeight.mm(keyWeight[self.dataset['word_to_ix'][topic_word]].view(-1, 1)).tolist()[0][0]

		last_epoch_log = {
			'non_topic_score' : non_topic_score_lastEpoch,
			'non_topic_emb_norm' : non_topic_emb_norm_lastEpoch,
			'topic_score' : topic_score_lastEpoch,
			'topic_emb_norm' : topic_emb_norm_lastEpoch
			}

		return self.embeddingNormTopicLog[:self.record_idx], \
			self.scoreTopicLog[:self.record_idx], \
			last_epoch_log, \
			self.trainLossLog[:self.record_idx]


class Para_Recorder_SST:
	def __init__(self, model, dataset):
		self.model = model
		self.embeddingNormLog = []
		self.scoreLog = []
		self.record_idx = 0
		self.dataset = dataset

	def record(self, train_loss):
		embeddingWeight = None
		keyWeight = None
		queryWeight = None

		for name, parameter in self.model.named_parameters():
			if (name == 'keys.keys.weight'):
				keyWeight = copy.deepcopy(parameter.data)
			if (name == 'embeddings.embeddings.weight'):
				embeddingWeight = copy.deepcopy(parameter.data)
			if (name == 'attn.queries'):
				queryWeight = copy.deepcopy(parameter.data)

		embeddingNorms = np.zeros(len(self.dataset['wordVocDict']))
		scores = np.zeros(len(self.dataset['wordVocDict']))
		for word_idx, word in enumerate(self.dataset['wordVocDict']):
			embeddingNorms[self.dataset['word_to_ix'][word]] = \
				eucleadan_norm(embeddingWeight[self.dataset['word_to_ix'][word]])*eucleadan_norm(queryWeight)
			scores[self.dataset['word_to_ix'][word]] = \
				queryWeight.mm(keyWeight[self.dataset['word_to_ix'][word]].view(-1, 1)).tolist()[0][0]


		self.embeddingNormLog.append(embeddingNorms)
		self.scoreLog.append(scores)


		self.record_idx += 1

	def get_record(self):
		scoreLog_Numpy = np.zeros((len(self.scoreLog), len(self.dataset['word_to_ix'])))
		embeddingNormLog_Numpy = np.zeros((len(self.embeddingNormLog), len(self.dataset['word_to_ix'])))

		for snapIdx in range(len(self.scoreLog)):
			scoreLog_Numpy[snapIdx] = self.scoreLog[snapIdx]
			embeddingNormLog_Numpy[snapIdx] = self.embeddingNormLog[snapIdx]

		return embeddingNormLog_Numpy, scoreLog_Numpy


def get_trainable_para(query_trainable, key_trainable, emb_trainable, model):
	trainable_para_lst = []
	for name, parameter in model.named_parameters():
		if ('topicPred' in name):
			trainable_para_lst += [parameter]
		if (name == 'attn.queries' and query_trainable):
			trainable_para_lst += [parameter]
		if (name == 'embeddings.embeddings.weight' and emb_trainable):
			trainable_para_lst += [parameter]
		if (name == 'keys.keys.weight' and key_trainable):
			trainable_para_lst += [parameter]

	return trainable_para_lst

def config_model(configFilePath, voc_size, numTopic, dev=None):
	config = configparser.ConfigParser()
	config.read(configFilePath)
	model_name = config.get('Model Config', 'model_name')
	dimEmb = int(config.get('Model Config', 'dimEmb'))
	dimKey = int(config.get('Model Config', 'dimKey'))
	dimQuery = int(config.get('Model Config', 'dimQuery'))
	queryVar = float(config.get('Model Config', 'queryVar'))
	embVar = float(config.get('Model Config', 'embVar'))
	model = Model(voc_size, dimKey, dimEmb, dimQuery, numTopic, queryVar, embVar, model_name).to(dev)
	return model

def produce_data(configFilePath, dev=None):
	config = configparser.ConfigParser()
	config.read(configFilePath)
	numTopic = int(config.get('Synthetic Gen', 'numTopic'))
	distr = ast.literal_eval(config.get('Synthetic Gen', 'distr'))
	numSample = int(config.get('Synthetic Gen', 'numSample'))
	wordSize = int(config.get('Synthetic Gen', 'wordSize'))
	randomWordVocSize = int(config.get('Synthetic Gen', 'randomWordVocSize'))
	numRandomWordPerSent = int(config.get('Synthetic Gen', 'numRandomWordPerSent'))
	partition = ast.literal_eval(config.get('Synthetic Gen', 'partition'))
	partitionInIndex = [floor(int((numSample-1)*rate)) for rate in partition]
	sentList, topicList, wordVocDict, topicWordLst, randomWordDict = genTrainingSentTopic(numTopic, numSample, wordSize, numRandomWordPerSent, randomWordVocSize, distr)
	sentListTrain = sentList[:partitionInIndex[0]]
	topicListTrain = topicList[:partitionInIndex[0]]
	sentListValid = sentList[partitionInIndex[0]:partitionInIndex[1]]
	topicListValid = topicList[partitionInIndex[0]:partitionInIndex[1]]
	sentListTest = sentList[partitionInIndex[1]:partitionInIndex[2]]
	topicListTest = topicList[partitionInIndex[1]:partitionInIndex[2]]

	# convert all the samples into tensor presentation. 
	word_to_ix = {word: i for i, word in enumerate(wordVocDict)}
	context_idxs_train = torch.tensor([[word_to_ix[w] for w in sent] for sent in sentListTrain], dtype=torch.long).to(dev)
	target_batch_train = torch.tensor([target for target in topicListTrain], dtype=torch.long).to(dev)
	context_idxs_valid = torch.tensor([[word_to_ix[w] for w in sent] for sent in sentListTrain], dtype=torch.long).to(dev)
	target_batch_valid = torch.tensor([target for target in topicListTrain], dtype=torch.long).to(dev)
	context_idxs_test = torch.tensor([[word_to_ix[w] for w in sent] for sent in sentListTest], dtype=torch.long).to(dev)
	target_batch_test = [target for target in topicListTest]

	return {
	'numTopic': numTopic,
	'wordVocDict': wordVocDict,
	'topicWordLst': topicWordLst,
	'randomWordDict': randomWordDict,
	'word_to_ix': word_to_ix,
	'context_idxs_train': context_idxs_train,
	'target_batch_train': target_batch_train,
	'context_idxs_valid': context_idxs_valid,
	'target_batch_valid': target_batch_valid,
	'context_idxs_test': context_idxs_test,
	'target_batch_test': target_batch_test,
	}


def load_single_data_file(dataset_path):
	num_sample = 0 
	maxSentLen = 0
	word_list, sent_list, topic_list = [], [], []

	with open(dataset_path) as tsvfile:
		tsvreader = csv.reader(tsvfile, delimiter="\t")
		for line in tsvreader:
			topicSent = line[0].split()
			topic = topicSent[0]
			sent = ' '.join(topicSent[1:])
			words = re.sub(r'[^\w\s]','',sent).split()		
			sent_list.append(words)
			maxSentLen = max(maxSentLen, len(words))
			word_list += words
			topic_list.append(int(topic))
			num_sample+=1

	return word_list, sent_list, topic_list, num_sample, maxSentLen


def padding(sent, maxSentLen, padStr):
	"""padding sent to have length maxSentLen using padStr"""
	return(sent+[padStr]*(maxSentLen-len(sent)))

def load_sst(sst2_data_set_lst, dev=None):
	padStr = '$PAD$'
	word_list_train, sent_list_train, topic_list_train, num_sample_train, maxSentLen_train = \
	 load_single_data_file(sst2_data_set_lst[0])
	word_list_valid, sent_list_valid, topic_list_valid, num_sample_valid, maxSentLen_valid = \
	 load_single_data_file(sst2_data_set_lst[1])
	word_list_test, sent_list_test, topic_list_test, num_sample_test, maxSentLen_test = \
	 load_single_data_file(sst2_data_set_lst[2])

	wordVocDict = set(word_list_train+word_list_valid+word_list_test+[padStr])
	word_to_ix = {word: i for i, word in enumerate(wordVocDict)}
	maxSentLen = max(maxSentLen_train, maxSentLen_valid, maxSentLen_test)

	context_idxs_train = torch.tensor([[word_to_ix[w] for w in padding(sent, maxSentLen, padStr)] for sent in sent_list_train], dtype=torch.long).to(dev)
	context_idxs_valid = torch.tensor([[word_to_ix[w] for w in padding(sent, maxSentLen, padStr)] for sent in sent_list_valid], dtype=torch.long).to(dev)
	context_idxs_test = torch.tensor([[word_to_ix[w] for w in padding(sent, maxSentLen, padStr)] for sent in sent_list_test], dtype=torch.long).to(dev)
	target_batch_train = torch.tensor([target for target in topic_list_train], dtype=torch.long).to(dev)
	target_batch_valid = torch.tensor([target for target in topic_list_valid], dtype=torch.long).to(dev)
	target_batch_test = [target for target in topic_list_test]


	return {
	'numTopic': 2,
	'sent_list_train': sent_list_train,
	'topic_list_train': topic_list_train,
	'wordVocDict': wordVocDict,
	'word_to_ix': word_to_ix,
	'context_idxs_train': context_idxs_train,
	'context_idxs_valid': context_idxs_valid,
	'context_idxs_test': context_idxs_test,
	'target_batch_train': target_batch_train,
	'target_batch_valid': target_batch_valid,
	'target_batch_test': target_batch_test
	}


def perform_exp(configFilePath, model, dataset, para_Recorder, dev=None, \
	record_period = 1, patience  = -1, path_to_save_model = None):
	# Load experiment configurations
	config = configparser.ConfigParser()
	config.read(configFilePath)	
	learning_rate = float(config.get('Train Config', 'learning_rate'))
	query_trainable = eval(config.get('Train Config', 'query_trainable'))
	key_trainable = eval(config.get('Train Config', 'key_trainable'))
	emb_trainable = eval(config.get('Train Config', 'emb_trainable'))
	
	
	# generate a training set
	loss_fn = torch.nn.CrossEntropyLoss(reduction='mean').to(dev)
	trainable_para_lst = get_trainable_para(query_trainable, key_trainable, emb_trainable, model)
	optimizer = torch.optim.SGD(trainable_para_lst, lr = learning_rate)
	

	if  patience < 0 :
		num_epoch = int(config.get('Train Config', 'numEpoch'))
		for epoch_idx in range(num_epoch):
			model.zero_grad()
			log_probs = model(dataset['context_idxs_train'])
			loss = loss_fn(log_probs, dataset['target_batch_train'])
			loss.backward()	
			optimizer.step()
			if epoch_idx%record_period == 0:
				para_Recorder.record(loss.item())
			with torch.no_grad():
				log_probs = model(dataset['context_idxs_valid'])
				loss_valid = loss_fn(log_probs, dataset['target_batch_valid'])
			print(epoch_idx, ' Loss: ', loss.item(), 'Valid Loss: ', loss_valid.item())
	else:
		patience_current = patience
		torch.save(model.state_dict(), os.path.join(path_to_save_model, 'paras.pt'))
		best_valid_loss = 1e8
		epoch_idx = 0
		while patience_current > 0:
			model.zero_grad()
			log_probs = model(dataset['context_idxs_train'])
			loss = loss_fn(log_probs, dataset['target_batch_train'])
			loss.backward()	
			optimizer.step()
			if epoch_idx%record_period == 0:
				para_Recorder.record(loss.item())
			with torch.no_grad():
				log_probs = model(dataset['context_idxs_valid'])
				loss_valid = loss_fn(log_probs, dataset['target_batch_valid'])
			print(epoch_idx, ' Loss: ', loss.item(), 'Valid Loss: ', loss_valid.item())
			epoch_idx = epoch_idx + 1
			# update patience and the best model
			if best_valid_loss > loss_valid:
				# reset patience
				patience_current = patience 
				# save model
				torch.save(model.state_dict(), os.path.join(path_to_save_model, 'paras.pt'))
				# update best_valid_loss
				best_valid_loss = loss_valid
			else:
				# loss patience by one
				patience_current = patience_current - 1

		print("Early Stop. Load the best model.")
		model.load_state_dict(torch.load(os.path.join(path_to_save_model, 'paras.pt')))

	with torch.no_grad():
		log_probs = model(dataset['context_idxs_test'])
	log_probs = log_probs.cpu().numpy()
	pred_target = np.argmax(log_probs, axis=1)
	print('Test accuracy:', accuracy_score(dataset['target_batch_test'], pred_target))
	return para_Recorder


def get_model_name(model_config_path, train_config_path):
	config = configparser.ConfigParser()
	config.read(model_config_path)
	model_name = config.get('Model Config', 'model_name')
	config = configparser.ConfigParser()
	config.read(train_config_path)
	query_trainable = eval(config.get('Train Config', 'query_trainable'))
	key_trainable = eval(config.get('Train Config', 'key_trainable'))
	emb_trainable = eval(config.get('Train Config', 'emb_trainable'))
	model_name = 'Attn-'+model_name
	if(not key_trainable):
		model_name+='-KF'
	if(not emb_trainable):
		model_name+='-EF'
	if(query_trainable):
		model_name+='-QT'

	return model_name

