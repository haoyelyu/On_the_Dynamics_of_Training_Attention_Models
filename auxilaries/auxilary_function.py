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
		self.topic_embedding_norm_log = np.zeros((len(dataset['topic_word_list']), num_cache))
		self.topic_score_log = np.zeros((len(dataset['topic_word_list']), num_cache))
		self.trainLossLog = np.zeros(num_cache,)
		self.record_idx = 0
		self.dataset = dataset

	def record(self, train_loss):
		embedding_weight = None
		key_weight = None
		query_weight = None
		for name, parameter in self.model.named_parameters():
			if (name == 'keys.keys.weight'):
				key_weight = copy.deepcopy(parameter.data)
			if (name == 'embeddings.embeddings.weight'):
				embedding_weight = copy.deepcopy(parameter.data)
			if (name == 'attn.queries'):
				query_weight = copy.deepcopy(parameter.data)


		if self.record_idx >= self.num_cache:
			print('Recorder is full!!')
		else:
			for topic_idx, topic in enumerate(self.dataset['topic_word_list']):
				self.topic_embedding_norm_log[topic_idx, self.record_idx] = \
					eucleadan_norm(embedding_weight[self.dataset['word_to_ix'][topic]])*eucleadan_norm(query_weight)
				self.topic_score_log[topic_idx, self.record_idx] = \
					query_weight.mm(key_weight[self.dataset['word_to_ix'][topic]].view(-1, 1)).tolist()[0][0]
				self.trainLossLog[self.record_idx] = train_loss
			self.record_idx += 1

	def get_record(self):
		for name, parameter in self.model.named_parameters():
			if (name == 'keys.keys.weight'):
				key_weight = copy.deepcopy(parameter.data)
			if (name == 'embeddings.embeddings.weight'):
				embedding_weight = copy.deepcopy(parameter.data)
			if (name == 'attn.queries'):
				query_weight = copy.deepcopy(parameter.data)

		non_topic_score_lastEpoch = np.zeros(len(self.dataset['non_topic_word_dict']))
		non_topic_emb_norm_lastEpoch = np.zeros(len(self.dataset['non_topic_word_dict']))
		topic_score_lastEpoch = np.zeros(len(self.dataset['topic_word_list']))
		topic_emb_norm_lastEpoch = np.zeros(len(self.dataset['topic_word_list']))

		for non_topic_word_idx, non_topic_word in enumerate(self.dataset['non_topic_word_dict']):
			non_topic_emb_norm_lastEpoch[non_topic_word_idx] = \
				eucleadan_norm(embedding_weight[self.dataset['word_to_ix'][non_topic_word]])*eucleadan_norm(query_weight)
			non_topic_score_lastEpoch[non_topic_word_idx] = \
				query_weight.mm(key_weight[self.dataset['word_to_ix'][non_topic_word]].view(-1, 1)).tolist()[0][0]

		for topic_word_idx, topic_word in enumerate(self.dataset['topic_word_list']):
			topic_emb_norm_lastEpoch[topic_word_idx] = \
				eucleadan_norm(embedding_weight[self.dataset['word_to_ix'][topic_word]])*eucleadan_norm(query_weight)
			topic_score_lastEpoch[topic_word_idx] = \
				query_weight.mm(key_weight[self.dataset['word_to_ix'][topic_word]].view(-1, 1)).tolist()[0][0]

		last_epoch_log = {
			'non_topic_score' : non_topic_score_lastEpoch,
			'non_topic_emb_norm' : non_topic_emb_norm_lastEpoch,
			'topic_score' : topic_score_lastEpoch,
			'topic_emb_norm' : topic_emb_norm_lastEpoch
			}

		return self.topic_embedding_norm_log[:self.record_idx], \
			self.topic_score_log[:self.record_idx], \
			last_epoch_log, \
			self.trainLossLog[:self.record_idx]


class Para_Recorder_SST:
	def __init__(self, model, dataset):
		self.model = model
		self.embedding_norm_log = []
		self.score_log = []
		self.record_idx = 0
		self.dataset = dataset

	def record(self, train_loss):
		embedding_weight = None
		key_weight = None
		query_weight = None

		for name, parameter in self.model.named_parameters():
			if (name == 'keys.keys.weight'):
				key_weight = copy.deepcopy(parameter.data)
			if (name == 'embeddings.embeddings.weight'):
				embedding_weight = copy.deepcopy(parameter.data)
			if (name == 'attn.queries'):
				query_weight = copy.deepcopy(parameter.data)

		embedding_norms = np.zeros(len(self.dataset['word_voc_dict']))
		scores = np.zeros(len(self.dataset['word_voc_dict']))
		for word_idx, word in enumerate(self.dataset['word_voc_dict']):
			embedding_norms[self.dataset['word_to_ix'][word]] = \
				eucleadan_norm(embedding_weight[self.dataset['word_to_ix'][word]])*eucleadan_norm(query_weight)
			scores[self.dataset['word_to_ix'][word]] = \
				query_weight.mm(key_weight[self.dataset['word_to_ix'][word]].view(-1, 1)).tolist()[0][0]


		self.embedding_norm_log.append(embedding_norms)
		self.score_log.append(scores)


		self.record_idx += 1

	def get_record(self):
		score_log_numpy = np.zeros((len(self.score_log), len(self.dataset['word_to_ix'])))
		embedding_norm_log_numpy = np.zeros((len(self.embedding_norm_log), len(self.dataset['word_to_ix'])))

		for snapIdx in range(len(self.score_log)):
			score_log_numpy[snapIdx] = self.score_log[snapIdx]
			embedding_norm_log_numpy[snapIdx] = self.embedding_norm_log[snapIdx]

		return embedding_norm_log_numpy, score_log_numpy


def get_trainable_para(query_trainable, key_trainable, emb_trainable, model):
	trainable_para_list = []
	for name, parameter in model.named_parameters():
		if 'topicPred' in name:
			trainable_para_list += [parameter]
		if name == 'attn.queries' and query_trainable:
			trainable_para_list += [parameter]
		if name == 'embeddings.embeddings.weight' and emb_trainable:
			trainable_para_list += [parameter]
		if name == 'keys.keys.weight' and key_trainable:
			trainable_para_list += [parameter]

	return trainable_para_list

def config_model(config_file_path, voc_size, num_topic, dev=None):
	config = configparser.ConfigParser()
	config.read(config_file_path)
	model_name = config.get('Model Config', 'model_name')
	dim_emb = int(config.get('Model Config', 'dimEmb'))
	dim_key = int(config.get('Model Config', 'dimKey'))
	dim_query = int(config.get('Model Config', 'dimQuery'))
	query_var = float(config.get('Model Config', 'queryVar'))
	emb_var = float(config.get('Model Config', 'embVar'))
	model = Model(voc_size, dim_key, dim_emb, dim_query, num_topic, query_var, emb_var, model_name).to(dev)
	return model

def produce_data(config_file_path, dev=None):
	config = configparser.ConfigParser()
	config.read(config_file_path)
	num_topic = int(config.get('Synthetic Gen', 'numTopic'))
	distr = ast.literal_eval(config.get('Synthetic Gen', 'distr'))
	num_sample = int(config.get('Synthetic Gen', 'numSample'))
	word_size = int(config.get('Synthetic Gen', 'wordSize'))
	non_topic_word_voc_size = int(config.get('Synthetic Gen', 'nonTopicWordVocSize'))
	num_non_topic_word_per_sent = int(config.get('Synthetic Gen', 'numNonTopicWordPerSent'))
	partition = ast.literal_eval(config.get('Synthetic Gen', 'partition'))
	partition_in_index = [floor(int((num_sample-1)*rate)) for rate in partition]
	sent_list, topic_list, word_voc_dict, topic_word_list, non_topic_word_dict = \
		genTrainingSentTopic(num_topic, num_sample, word_size, num_non_topic_word_per_sent, non_topic_word_voc_size, distr)
	sent_list_train = sent_list[:partition_in_index[0]]
	topic_list_train = topic_list[:partition_in_index[0]]
	sent_list_valid = sent_list[partition_in_index[0]:partition_in_index[1]]
	topic_list_valid = topic_list[partition_in_index[0]:partition_in_index[1]]
	sent_list_test = sent_list[partition_in_index[1]:partition_in_index[2]]
	topic_list_test = topic_list[partition_in_index[1]:partition_in_index[2]]

	# convert all the samples into tensor presentation. 
	word_to_ix = {word: i for i, word in enumerate(word_voc_dict)}
	context_idxs_train = torch.tensor([[word_to_ix[w] for w in sent] for sent in sent_list_train], dtype=torch.long).to(dev)
	target_batch_train = torch.tensor([target for target in topic_list_train], dtype=torch.long).to(dev)
	context_idxs_valid = torch.tensor([[word_to_ix[w] for w in sent] for sent in sent_list_valid], dtype=torch.long).to(dev)
	target_batch_valid = torch.tensor([target for target in topic_list_valid], dtype=torch.long).to(dev)
	context_idxs_test = torch.tensor([[word_to_ix[w] for w in sent] for sent in sent_list_test], dtype=torch.long).to(dev)
	target_batch_test = [target for target in topic_list_test]

	return {
	'num_topic': num_topic,
	'word_voc_dict': word_voc_dict,
	'topic_word_list': topic_word_list,
	'non_topic_word_dict': non_topic_word_dict,
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
	max_sent_len = 0
	word_list, sent_list, topic_list = [], [], []

	with open(dataset_path) as tsvfile:
		tsvreader = csv.reader(tsvfile, delimiter="\t")
		for line in tsvreader:
			topicSent = line[0].split()
			topic = topicSent[0]
			sent = ' '.join(topicSent[1:])
			words = re.sub(r'[^\w\s]','',sent).split()		
			sent_list.append(words)
			max_sent_len = max(max_sent_len, len(words))
			word_list += words
			topic_list.append(int(topic))
			num_sample+=1

	return word_list, sent_list, topic_list, num_sample, max_sent_len


def padding(sent, max_sent_len, pad_str):
	"""padding sent to have length max_sent_len using pad_str"""
	return(sent+[pad_str]*(max_sent_len-len(sent)))

def load_sst(data_set_list, dev=None):
	pad_str = '$PAD$'
	word_list_train, sent_list_train, topic_list_train, num_sample_train, max_sent_len_train = \
	 load_single_data_file(data_set_list[0])
	word_list_valid, sent_list_valid, topic_list_valid, num_sample_valid, max_sent_len_valid = \
	 load_single_data_file(data_set_list[1])
	word_list_test, sent_list_test, topic_list_test, num_sample_test, max_sent_len_test = \
	 load_single_data_file(data_set_list[2])
	word_voc_dict = set(word_list_train+word_list_valid+word_list_test+[pad_str])
	word_to_ix = {word: i for i, word in enumerate(word_voc_dict)}
	max_sent_len = max(max_sent_len_train, max_sent_len_valid, max_sent_len_test)

	context_idxs_train = torch.tensor([[word_to_ix[w] for w in padding(sent, max_sent_len, pad_str)] for sent in sent_list_train], dtype=torch.long).to(dev)
	context_idxs_valid = torch.tensor([[word_to_ix[w] for w in padding(sent, max_sent_len, pad_str)] for sent in sent_list_valid], dtype=torch.long).to(dev)
	context_idxs_test = torch.tensor([[word_to_ix[w] for w in padding(sent, max_sent_len, pad_str)] for sent in sent_list_test], dtype=torch.long).to(dev)
	target_batch_train = torch.tensor([target for target in topic_list_train], dtype=torch.long).to(dev)
	target_batch_valid = torch.tensor([target for target in topic_list_valid], dtype=torch.long).to(dev)
	target_batch_test = [target for target in topic_list_test]

	return {
	'sent_list_train': sent_list_train,
	'topic_list_train': topic_list_train,
	'word_voc_dict': word_voc_dict,
	'word_to_ix': word_to_ix,
	'context_idxs_train': context_idxs_train,
	'context_idxs_valid': context_idxs_valid,
	'context_idxs_test': context_idxs_test,
	'target_batch_train': target_batch_train,
	'target_batch_valid': target_batch_valid,
	'target_batch_test': target_batch_test
	}


def perform_exp(config_file_path, model, dataset, para_Recorder, dev=None, \
	record_period = 1, patience  = -1, path_to_save_model = None):
	# Load experiment configurations
	config = configparser.ConfigParser()
	config.read(config_file_path)	
	learning_rate = float(config.get('Train Config', 'learning_rate'))
	query_trainable = eval(config.get('Train Config', 'query_trainable'))
	key_trainable = eval(config.get('Train Config', 'key_trainable'))
	emb_trainable = eval(config.get('Train Config', 'emb_trainable'))
	
	
	# generate a training set
	loss_fn = torch.nn.CrossEntropyLoss(reduction='mean').to(dev)
	trainable_para_list = get_trainable_para(query_trainable, key_trainable, emb_trainable, model)
	optimizer = torch.optim.SGD(trainable_para_list, lr = learning_rate)
	

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

