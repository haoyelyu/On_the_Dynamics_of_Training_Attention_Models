import torch
from models.model_parts import *

class Model(torch.nn.Module):
	def __init__(self, word_voc_dict_size, dim_key, dim_emb, dim_query, num_topic, query_var, emb_var, classifier_type):
		super().__init__()
		self.keys = Keys(word_voc_dict_size, dim_key)
		self.embeddings = Embeddings(word_voc_dict_size, dim_emb, emb_var)
		self.attn = Attn(dim_query, query_var)
		if classifier_type == 'FC':
			self.topicPred = ClassifyTopic_FC(dim_emb, num_topic)
		elif classifier_type == 'TC':
			self.topicPred = ClassifyTopic_TC(dim_emb, num_topic)
		elif classifier_type == 'TL':
			self.topicPred = ClassifyTopic_TL(dim_emb, num_topic)
		else:
			raise Exception(f"Illegal classifier_type: {classifier_type}")

	def forward(self, context_idxs):
		return self.topicPred(self.attn(self.embeddings(context_idxs), self.keys(context_idxs)))