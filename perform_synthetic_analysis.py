import torch
import configparser
import ast
import os
import numpy as np
from sklearn.metrics import accuracy_score
from auxilaries.auxilary_function import *
from auxilaries.result_generation import *

dir_to_store_exp_result = 'results/synthetic_analysis_results'
model_config_path_lst = ['exp_config/model_config_FC.txt', 'exp_config/model_config_TC.txt', 'exp_config/model_config_TL.txt']
dataset_config_path = 'exp_config/dataset_config.txt'
train_config_path = 'exp_config/train_config.txt'

last_epoch_log_lst = []
model_name_lst = []
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = produce_data(dataset_config_path, dev=dev)

for model_config_path in model_config_path_lst:
	model = config_model(model_config_path, len(dataset['word_voc_dict']), dataset['num_topic'], dev=dev)
	config = configparser.ConfigParser()
	config.read(train_config_path)	
	num_epoch = int(config.get('Train Config', 'numEpoch'))
	para_Recorder = Para_Recorder(num_epoch, model, dataset, 1)
	perform_exp(train_config_path, model, dataset, para_Recorder, dev=dev)
	topic_embedding_norm_log, topic_score_log, last_epoch_log, training_loss_log = para_Recorder.get_record()
	model_name = get_model_name(model_config_path, train_config_path)
	path_to_store_exp_result = os.path.join(dir_to_store_exp_result, model_name)
	mkdir(path_to_store_exp_result)
	plot_SEN_curve(model_config_path, dataset_config_path, train_config_path, \
		dataset, topic_embedding_norm_log, topic_score_log, path_to_store_exp_result, model_name)
	plot_loss_curve([training_loss_log], path_to_store_exp_result, [model_name])
	model_name_lst = model_name_lst + [model_name]
	last_epoch_log_lst = last_epoch_log_lst + [last_epoch_log]

plot_emb_norm_and_score_distribution(last_epoch_log_lst, model_name_lst, dir_to_store_exp_result)

