import torch
from models.model_parts import *

class Model(torch.nn.Module):
	def __init__(self, wordVocDict_size, dimKey, dimEmb, dimQuery, numTopic, queryVar, embVar, classfier_type):
		super().__init__()
		self.keys = Keys(wordVocDict_size, dimKey)
		self.embeddings = Embeddings(wordVocDict_size, dimEmb, embVar)
		self.attn = Attn(dimQuery, queryVar)
		if classfier_type == 'FC':
			self.topicPred = ClassifyTopic_FC(dimEmb, numTopic)
		elif classfier_type == 'TC':
			self.topicPred = ClassifyTopic_TC(dimEmb, numTopic)
		elif classfier_type == 'TL':
			self.topicPred = ClassifyTopic_TL(dimEmb, numTopic)
		else:
			raise Exception(f"Illegal classfier_type: {classfier_type}")

	def forward(self, context_idxs):
		return self.topicPred(self.attn(self.embeddings(context_idxs), self.keys(context_idxs)))