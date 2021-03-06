import torch
import configparser
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
path_to_save_model = '.sst2_best_model'
word_to_trace_list = ['powerful', 'stupid','solid', 'suffers', 'mess']
n_largest = 100
sent_idx_to_trace = 3586
word_idx_to_trace = [1, 7]

if __name__ == "__main__":
	dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	dataset = load_sst(sst2_data_set_lst, dev = dev)
	dataset['num_topic'] = 2
	mkdir(path_to_save_model)

	# get args from the command line
	model_type = sys.argv[1]

	if model_type not in ['FC', 'TC']:
		raise Exception(f'Illegal model type {model_type}')

	if model_type == 'FC':
		model_config_path = model_config_path_lst[0]
	else:
		model_config_path = model_config_path_lst[1]

	model = config_model(model_config_path, len(dataset['word_voc_dict']), dataset['num_topic'], dev=dev)
	config = configparser.ConfigParser()
	config.read(model_config_path)
	model_name = get_model_name(model_config_path, train_config_path)
	print(f'Experiment is performed on Model {model_name}')
	path_to_store_exp_result = os.path.join(dir_to_store_exp_result, model_name)
	mkdir(path_to_store_exp_result)
	para_Recorder = Para_Recorder_SST(model, dataset)

	perform_exp(train_config_path, model, dataset, para_Recorder, dev=dev, \
		record_period=1000, patience  = 2000, path_to_save_model = path_to_save_model)

	embedding_norm_log_numpy, score_log_numpy = para_Recorder.get_record()

	plot_SEN_curve_SST(dataset,	path_to_store_exp_result, \
		model_name, n_largest, embedding_norm_log_numpy, score_log_numpy)
	plot_topic_purity_dyn(word_to_trace_list, dataset, \
		path_to_store_exp_result, model_name, score_log_numpy)
	plot_two_word_in_sent_weight_dyn(sent_idx_to_trace, word_idx_to_trace, dataset, \
		embedding_norm_log_numpy, score_log_numpy, path_to_store_exp_result)