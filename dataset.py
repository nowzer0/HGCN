''' This is where we create customized Dataset which inherits from pytorch's abstarct class 'Dataset' '''

import torch
import ast
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from nltk.tokenize import sent_tokenize

# customizer dataset

str2list = lambda x : ast.literal_eval(x)

class InconsistencyNewsDataset(Dataset):
	def __init__(self, file_name, tokenizer, max_len, max_sent):
		super().__init__()
		self.sent_len = max_sent
		self.max_len = max_len
		self.data =  self.remove_nan(pd.read_csv(file_name))
		self.sent_tokenizer = sent_tokenize
		self.tokenizer = tokenizer
		self.len = self.data.shape[0]

	@staticmethod
	def remove_nan(data):
		data = data.replace(np.nan, '', regex=True)
		return data
	
	@staticmethod
	def split_to_sentence(sentence_list, max_len):
		if len(sentence_list) >= max_len:
			return sentence_list[0:max_len]
		else:
			return np.pad(sentence_list,(0,max_len-len(sentence_list)),'constant', constant_values=('', '')).tolist()

	def __len__(self):
		return self.len

	def __getitem__(self,idx):
		
		instance = self.data.iloc[idx]

		encoded_title = self.tokenizer.encode_plus(instance['title'],max_length=self.max_len, truncation=True, padding='max_length', return_tensors='pt')
		title_input_ids = encoded_title['input_ids'].flatten()
		title_attention_mask = encoded_title['attention_mask'].flatten()

		encoded_subtitle = self.tokenizer.encode_plus(instance['subtitle'],max_length=self.max_len, truncation=True, padding='max_length', return_tensors='pt')
		subtitle_input_ids = encoded_subtitle['input_ids'].flatten()
		subtitle_attention_mask = encoded_subtitle['attention_mask'].flatten()

		sent_list = self.sent_tokenizer(instance['body'])
		padded_sent_list = self.split_to_sentence(sent_list, max_len=self.sent_len)
		encoded_body = self.tokenizer.batch_encode_plus(padded_sent_list, max_length=self.max_len, truncation=True, padding='max_length', return_tensors='pt')
		body_input_ids = encoded_body['input_ids']
		body_attention_mask = encoded_body['attention_mask']

		encoded_caption = self.tokenizer.encode_plus(instance['caption'],max_length=self.max_len, truncation=True, padding='max_length', return_tensors='pt')
		caption_input_ids = encoded_caption['input_ids'].flatten()
		caption_attention_mask = encoded_caption['attention_mask'].flatten()
		label = torch.FloatTensor(instance['label'].flatten())


		return dict(input=dict(title=dict(input_ids=title_input_ids, attention_mask=title_attention_mask), 
				subtitle=dict(input_ids=subtitle_input_ids, attention_mask=subtitle_attention_mask),
				body = dict(input_ids=body_input_ids, attention_mask=body_attention_mask),
				caption = dict(input_ids=caption_input_ids, attention_mask=caption_attention_mask)),
				label=label
				)
