import torch

class Attn(torch.nn.Module):
	def __init__(self, dimQuery, queryVar):
		super().__init__()
		self.softmax = torch.nn.Softmax(dim = 2)
		self.queries = torch.nn.Parameter(torch.randn(1, dimQuery))
		self.queries.data.normal_(0, queryVar)


	def forward(self, embeds, keys):
		self.queries = self.queries.to(embeds.device)
		dotProdKQ = torch.matmul(self.queries, torch.transpose(keys, 1, 2))
		weights = self.softmax(dotProdKQ)
		contextV = torch.matmul(weights, embeds)
		contextV = torch.squeeze(contextV, 1)
		return contextV

class Keys(torch.nn.Module):
	def __init__(self, vocab_size, dimKey):
		super().__init__()
		torch.manual_seed(3)
		self.keys = torch.nn.Embedding(vocab_size, dimKey)
		self.keys.weight.data.uniform_(0, 0)

	def forward(self, inputs):
		keys = self.keys(inputs)
		return keys


class Embeddings(torch.nn.Module):
	def __init__(self, vocab_size, dimEmb, embVar = 0.0001):
		super().__init__()
		torch.manual_seed(3)
		self.embeddings = torch.nn.Embedding(vocab_size, dimEmb)
		self.embeddings.weight.data.normal_(0, embVar)

	def forward(self, inputs):
		embeds = self.embeddings(inputs)
		return embeds


class ClassifyTopic_FC(torch.nn.Module):
	def __init__(self, dimEmb, numTopic):
		super().__init__()
		self.w = torch.randn(dimEmb, numTopic, requires_grad=False).fill_diagonal_(1)

	def forward(self, contextV):
		self.w = self.w.to(contextV.device)
		log_probs = contextV.mm(self.w)
		return log_probs


class ClassifyTopic_TC(torch.nn.Module):
	def __init__(self, dimEmb, numTopic):
		super().__init__()
		self.linear = torch.nn.Linear(dimEmb, numTopic, bias=False)

	def forward(self, contextV):
		log_probs = self.linear(contextV)
		return log_probs


class ClassifyTopic_TL(torch.nn.Module):
	def __init__(self, dimEmb, numTopic):
		super().__init__()
		self.front = torch.nn.Linear(dimEmb, 10)
		self.end = torch.nn.Linear(10, numTopic)

	def forward(self, contextV):
		log_probs = self.end(self.front(contextV).clamp(min=0))
		return log_probs