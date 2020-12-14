import torch
import configparser
import ast
import os
import numpy as np
import sys
from sklearn.metrics import accuracy_score
from auxilaries.auxilary_function import *
from auxilaries.result_generation import *

dir_to_store_exp_result = 'results/sst2_analysis_results'
model_config_path_lst = ['exp_config/model_config_FC_SST.txt', 'exp_config/model_config_TC_SST.txt']
train_config_path = 'exp_config/train_config_SST.txt'
sst2_data_set_lst = ['data/sst_data/sst2/stsa.binary.train', 'data/sst_data/sst2/stsa.binary.dev', 'data/sst_data/sst2/stsa.binary.test']
path_to_save_model = 'sst2_best_model'
wordToTrace_Lst = ['powerful', 'stupid','solid', 'suffers', 'mess']
n_largest = 100
sentIdxForTrace = 3586
wordIndToTrace = [1, 7]

if __name__ == "__main__":
	dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	dataset = load_sst(sst2_data_set_lst, dev = dev)
	mkdir(path_to_save_model)

	# get args from the command line
	model_type = sys.argv[1]

	if model_type not in ['FC', 'TC']:
		raise Exception(f'Illegal model type {model_type}')

	if model_type == 'FC':
		model_config_path = model_config_path_lst[0]
	else:
		model_config_path = model_config_path_lst[1]

	model = config_model(model_config_path, len(dataset['wordVocDict']), dataset['numTopic'], dev=dev)
	config = configparser.ConfigParser()
	config.read(model_config_path)
	model_name = get_model_name(model_config_path, train_config_path)
	print(f'Experiment is performed on Model {model_name}')
	path_to_store_exp_result = os.path.join(dir_to_store_exp_result, model_name)
	mkdir(path_to_store_exp_result)
	para_Recorder = Para_Recorder_SST(model, dataset)

	perform_exp(train_config_path, model, dataset, para_Recorder, dev=dev, \
		record_period=1000, patience  = 2000, path_to_save_model = path_to_save_model)

	embeddingNormLog_Numpy, scoreLog_Numpy = para_Recorder.get_record()

	plot_SEN_curve_SST(dataset,	path_to_store_exp_result, \
		model_name, n_largest, embeddingNormLog_Numpy, scoreLog_Numpy)
	plot_topic_purity_dyn(wordToTrace_Lst, dataset, \
		path_to_store_exp_result, model_name, scoreLog_Numpy)
	plot_two_word_in_sent_weight_dyn(sentIdxForTrace, wordIndToTrace, dataset, \
		embeddingNormLog_Numpy, scoreLog_Numpy, path_to_store_exp_result)