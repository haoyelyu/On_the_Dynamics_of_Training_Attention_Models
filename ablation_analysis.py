import torch
import configparser
import ast
import os
import numpy as np
from sklearn.metrics import accuracy_score
from auxilaries.auxilary_function import *
from auxilaries.result_generation import *

dir_to_store_exp_result = 'results/ablation_analysis_results'
model_config_path_lst = ['exp_config/ablation_model_config_FC.txt', 'exp_config/ablation_model_config_TC.txt']
dataset_config_path = 'exp_config/dataset_config.txt'
train_config_path_lst = ['exp_config/ablation_train_config.txt', \
						'exp_config/ablation_train_config_KF.txt',\
						'exp_config/ablation_train_config_EF.txt']


dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = produce_data(dataset_config_path, dev=dev)

for model_config_path in model_config_path_lst:
	last_epoch_log_lst = []
	model_name_lst = []
	training_loss_log_lst = []
	embeddingNormTopicLog_lst = []
	scoreTopicLog_lst = []
	config = configparser.ConfigParser()
	config.read(model_config_path)
	print(model_config_path)
	path_to_store_exp_result = os.path.join(dir_to_store_exp_result, config.get('Model Config', 'model_name'))
	mkdir(path_to_store_exp_result)

	for train_config_path in train_config_path_lst:
		model = config_model(model_config_path, len(dataset['wordVocDict']), dataset['numTopic'], dev=dev)

		config.read(train_config_path)	
		num_epoch = int(config.get('Train Config', 'numEpoch'))
		para_Recorder = Para_Recorder(num_epoch, model, dataset, 1)

		perform_exp(train_config_path, model, dataset, para_Recorder, dev=dev)
		embeddingNormTopicLog, scoreTopicLog, last_epoch_log, training_loss_log = para_Recorder.get_record()
		config.read(model_config_path)
		queryVar = float(config.get('Model Config', 'queryVar'))
		config.read(dataset_config_path)
		numRandomWordPerSent = int(config.get('Synthetic Gen', 'numRandomWordPerSent'))
		model_name = get_model_name(model_config_path, train_config_path)
	
		model_name_lst = model_name_lst + [model_name]
		last_epoch_log_lst = last_epoch_log_lst + [last_epoch_log]
		training_loss_log_lst = training_loss_log_lst + [training_loss_log]
		embeddingNormTopicLog_lst = embeddingNormTopicLog_lst + [embeddingNormTopicLog]
		scoreTopicLog_lst = scoreTopicLog_lst + [scoreTopicLog]

	plot_loss_curve(training_loss_log_lst, path_to_store_exp_result, model_name_lst)
	plot_score_curve(dataset, scoreTopicLog_lst, path_to_store_exp_result, model_name_lst)
	plot_emb_norm_curve(dataset, embeddingNormTopicLog_lst, path_to_store_exp_result, model_name_lst)


