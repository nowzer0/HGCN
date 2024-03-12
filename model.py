import torch
from torch import nn
import torch.nn.functional as f
from torch.nn.modules.dropout import Dropout

class CLSPooler(nn.Module):
	def __init__(self,bert):
		super(CLSPooler,self).__init__()
		self.bert = bert

	def forward(self,inputs):
		outputs = self.bert(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask'])
		pooled_output = outputs.pooler_output.unsqueeze(1)

		return pooled_output


class HiddenLayerPooler(nn.Module):
	def __init__(self,bert,batch_size):
		super(HiddenLayerPooler,self).__init__()
		self.bert = bert
		self.batch_size=batch_size
		
	def forward(self,inputs):
		outputs = self.bert(input_ids = inputs['input_ids'],
			attention_mask = inputs['attention_mask'])

		hidden_states = outputs[0]
		pooled_output = torch.cat(tuple([hidden_states[i][j] for i in range(self.batch_size) for j in [-4, -3, -2, -1]]), dim=-1).reshape(self.batch_size,-1).unsqueeze(1)

		return pooled_output

class AveragePooler(nn.Module):
	def __init__(self,bert):
		super(AveragePooler,self).__init__()
		self.bert = bert
		
	def forward(self,inputs):
		outputs = self.bert(input_ids = inputs['input_ids'],
			attention_mask = inputs['attention_mask'])

		hidden_states = outputs[0]
		m = nn.AdaptiveAvgPool2d((1,768)) 
		pooled_output = m(hidden_states) 

		return pooled_output

# 2 Entitites > 1 Entity (ex. Title&Subtitle > Headline)
class Extractor(nn.Module):
	def __init__(self, input_dim, output_dim, drop_rate):
		super(Extractor,self).__init__()
		self.intermediate = nn.Linear(input_dim, output_dim, bias=True)
		self.layer_norm_inter =  nn.LayerNorm(output_dim)
		self.output = nn.Linear(output_dim, output_dim, bias=True)
		self.layer_norm_out =  nn.LayerNorm(output_dim)
		self.activation = nn.ELU()
		self.drop_layer = nn.Dropout(p=drop_rate)

	def forward(self,x): 
		x = self.intermediate(x)
		x = self.layer_norm_inter(x)
		x = self.activation(x)
		x = self.drop_layer(x)
		x = self.output(x)
		x = self.layer_norm_out(x)
		x = self.activation(x)
		x = self.drop_layer(x)
		return x

class BertWrapper(nn.Module):
	def __init__(self , bert, batch_first=True):
		super(BertWrapper,self).__init__()
		self.bert = bert
		self.batch_first = batch_first

	def forward(self, inputs):
		# extract input_ids and attention_mask from inputs
		input_ids = inputs['input_ids']
		attention_mask = inputs['attention_mask']

		# Squash samples and timesteps into a single axis
		input_ids = input_ids.contiguous().view(-1, input_ids.size(-1)) 
		attention_mask = attention_mask.contiguous().view(-1, attention_mask.size(-1))  
		reshaped_inputs = dict(input_ids=input_ids,attention_mask=attention_mask)
		outputs = self.bert(reshaped_inputs)

		# We have to reshape Y
		if self.batch_first:
			outputs = outputs.contiguous().view(inputs['input_ids'].size(0), -1, outputs.size(-1))  
		else:
			outputs = outputs.view(-1, inputs['input_ids'].size(1), outputs.size(-1)) 

		return outputs

# Headline & Bodytext Attention
class Headline_Bodytext_Attention(nn.Module):
	def __init__(self, bsz:int, seq_len:int, alpha:float):
		super(Headline_Bodytext_Attention,self).__init__()
		self.uniform = nn.Parameter(torch.Tensor(torch.ones((bsz,1,seq_len)).new_full((bsz,1,seq_len), 1/seq_len)),requires_grad=False) 
		self.alpha = alpha

	def forward(self,headline,bodytext):
		bodytext = torch.transpose(bodytext,1,2) 
		att_score = torch.bmm(headline, bodytext) 
		att_weight = f.softmax(att_score, dim=2) 
		bodytext = torch.transpose(bodytext,1,2) 
		b,_,s = att_weight.shape
		distribution = self.alpha * att_weight + (1-self.alpha) * self.uniform[:b,:,:]
		body = torch.bmm(distribution, bodytext) 
		return body, att_weight

class Body_Linear(nn.Module):
	def __init__(self, input_dim, output_dim=1):
		super(Body_Linear,self).__init__()
		self.linear = nn.Linear(input_dim, output_dim, bias=True) 
		self.activation = nn.ReLU()
		
	def forward(self,body_mat):
		body_mat = torch.transpose(body_mat,1,2) 
		body_vec = self.linear(body_mat)
		body_vec = self.activation(body_vec)
		body_vec = torch.transpose(body_vec,1,2)
		return body_vec

# Body Self-Attention /w headline
class Body_SelfAttention(nn.Module):
	def __init__(self, dim, heads, drop_rate, batch=True):
		super(Body_SelfAttention, self).__init__()
		self.multihead = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=drop_rate)

	def forward(self, bodytext_sents):
		attn_output, attn_output_weights = self.multihead(bodytext_sents, bodytext_sents, bodytext_sents)
		return attn_output, attn_output_weights


class GCN_layer(nn.Module):
	def __init__(self, input_dim, output_dim, drop_rate):
		super(GCN_layer, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.intermediate = nn.Linear(input_dim, output_dim, bias=True) 
		self.layer_norm =  nn.LayerNorm(output_dim)
		self.activation = nn.ELU()
		self.dropout = nn.Dropout(p=drop_rate) 
		self.Normalized_A = nn.Parameter(f.normalize(torch.Tensor(
			[[1., 1., 1., 0., 0., 0., 0.],
                       [1., 1., 0., 1., 1., 0., 0.],
                       [1., 0., 1., 0., 0., 1., 1.],
                       [0., 1., 0., 1., 0., 0., 0.],
                       [0., 1., 0., 0., 1., 0., 0.],
                       [0., 0., 1., 0., 0., 1., 0.],
                       [0., 0., 1., 0., 0., 0., 1.]
                      ]), p=1, dim=1),requires_grad=False) 

		self.register_buffer("adjacency", self.Normalized_A)
	def forward(self, input_x):
		x = self.intermediate(input_x) 
		x = self.layer_norm(x)
		x = self.activation(x)
		x = self.dropout(x)
		x = torch.matmul(self.Normalized_A, x) 
		return x +input_x 

# Hierarchial Graph Convolution Network
class HGCN(nn.Module):
	def __init__(self, vertex_dimension, gcn_layer, drop_rate):
		super(HGCN,self).__init__()
		self.GCN = nn.Sequential(*[GCN_layer(vertex_dimension, vertex_dimension,  drop_rate) for i in range(gcn_layer)])
		
	def forward(self,x):
		x = self.GCN(x)
		return x

'''
This is Our Adjacency Matrix
A = torch.FloatTensor([[1., 1., 1., 0., 0., 0., 0.],
                       [1., 1., 0., 1., 1., 0., 0.],
                       [1., 0., 1., 0., 0., 1., 1.],
                       [0., 1., 0., 1., 0., 0., 0.],
                       [0., 1., 0., 0., 1., 0., 0.],
                       [0., 0., 1., 0., 0., 1., 0.],
                       [0., 0., 1., 0., 0., 0., 1.]
                      ])
'''

# MLP Layer between GCN Layers
class MLP_Intermediate(nn.Module):
	def __init__(self, input_dim, hidden_dim, drop_rate):
		super(MLP_Intermediate,self).__init__()
		self.dense = nn.Linear(input_dim, hidden_dim, bias=True)
		self.layer_norm =  nn.LayerNorm(hidden_dim)
		self.activation = nn.ELU()
		self.dropout = nn.Dropout(p=drop_rate)


	def forward(self, hidden_states):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.layer_norm(hidden_states)
		hidden_states = self.activation(hidden_states)
		return hidden_states

class MLP_Output(nn.Module):
	def __init__(self, hidden_dim, output_dim):
		super(MLP_Output,self).__init__()
		self.dense = nn.Linear(hidden_dim, output_dim, bias=True)
		self.layer_norm =  nn.LayerNorm(output_dim)
		self.activation = nn.ELU()

	def forward(self, hidden_states):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.layer_norm(hidden_states)
		hidden_states = self.activation(hidden_states)
		return hidden_states

class MLP_Layer(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, drop_rate):
		super(MLP_Layer,self).__init__()
		self.intermediate = MLP_Intermediate(input_dim, hidden_dim, drop_rate)
		self.output = MLP_Output(hidden_dim, output_dim)

	def forward(self, input_states):
		hidden_states = self.intermediate(input_states)
		hidden_states = self.output(hidden_states)
		return hidden_states+input_states

# MLPClassifier for news vector
class MLPClassifier(nn.Module):
	def __init__(self,input_dim,hidden_dim,num_label,drop_rate):
		super(MLPClassifier,self).__init__()
		self.intermediate = MLP_Intermediate(input_dim, hidden_dim, drop_rate)
		self.to_label = nn.Linear(hidden_dim,num_label, bias=True)	

	def forward(self,x):
		x = self.intermediate(x)
		x = self.to_label(x)  
		x = torch.sigmoid(x) 
		return x
