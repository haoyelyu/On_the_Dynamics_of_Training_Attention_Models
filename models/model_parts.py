import torch

class Attn(torch.nn.Module):
	def __init__(self, dim_query, query_var):
		super().__init__()
		self.softmax = torch.nn.Softmax(dim = 2)
		self.queries = torch.nn.Parameter(torch.randn(1, dim_query))
		self.queries.data.normal_(0, query_var)


	def forward(self, embeds, keys):
		self.queries = self.queries.to(embeds.device)
		dotProdKQ = torch.matmul(self.queries, torch.transpose(keys, 1, 2))
		weights = self.softmax(dotProdKQ)
		contextV = torch.matmul(weights, embeds)
		contextV = torch.squeeze(contextV, 1)
		return contextV

class Keys(torch.nn.Module):
	def __init__(self, vocab_size, dim_key):
		super().__init__()
		torch.manual_seed(3)
		self.keys = torch.nn.Embedding(vocab_size, dim_key)
		self.keys.weight.data.uniform_(0, 0)

	def forward(self, inputs):
		keys = self.keys(inputs)
		return keys


class Embeddings(torch.nn.Module):
	def __init__(self, vocab_size, dim_emb, emb_var = 0.0001):
		super().__init__()
		torch.manual_seed(3)
		self.embeddings = torch.nn.Embedding(vocab_size, dim_emb)
		self.embeddings.weight.data.normal_(0, emb_var)

	def forward(self, inputs):
		embeds = self.embeddings(inputs)
		return embeds


class ClassifyTopic_FC(torch.nn.Module):
	def __init__(self, dim_emb, num_topic):
		super().__init__()
		self.w = torch.randn(dim_emb, num_topic, requires_grad=False).fill_diagonal_(1)

	def forward(self, contextV):
		self.w = self.w.to(contextV.device)
		log_probs = contextV.mm(self.w)
		return log_probs


class ClassifyTopic_TC(torch.nn.Module):
	def __init__(self, dim_emb, num_topic):
		super().__init__()
		self.linear = torch.nn.Linear(dim_emb, num_topic, bias=False)

	def forward(self, contextV):
		log_probs = self.linear(contextV)
		return log_probs


class ClassifyTopic_TL(torch.nn.Module):
	def __init__(self, dim_emb, num_topic):
		super().__init__()
		self.front = torch.nn.Linear(dim_emb, 10)
		self.end = torch.nn.Linear(10, num_topic)

	def forward(self, contextV):
		log_probs = self.end(self.front(contextV).clamp(min=0))
		return log_probs