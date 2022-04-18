# from summarizer import Summarizer
# from pprint import pprint
import spacy
from transformers import AutoTokenizer, AutoModel
from rouge_score import rouge_scorer
import os
import csv
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# tokenize text as required by BERT based models
def get_tokens(text, tokenizer):
  inputs = tokenizer.batch_encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True
        )
  ids = inputs['input_ids']
  mask = inputs['attention_mask']
  return ids, mask

def format_document(document):
	document = document.read().replace('\n', '')
	document = document.replace('\ufeff', '')
	document = document.replace('`', '')
	document = document.replace('\'\'', '')
	return document

def cat_documents(path, files):
	raw = ''
	for curr in files:
		file = os.path.join(path, curr)
		data = open(file, 'r')
		data = format_document(data)
		raw += data

	cat = []
	for sent in nlp(raw).sents:
		cat.append(str(sent))
	return cat

def printg(text):
	print('\033[1;32;40m', end='')
	print(text)
	print('\033[1;37;40m', end='')

class SentScore:
	def __init__(self, sent_id, sent, ref_id, ref, score):
		self.sent_id = sent_id
		self.sent = sent
		self.ref_id = ref_id
		self.ref = ref
		self.score = score


class DocumentDataset(Dataset):
	def __init__(self, csv_file, threshold):
		self.docs = pd.read_csv(csv_file)
		self.threshold = threshold
		self.sentence_model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
		self.tokenizer = AutoTokenizer.from_pretrained(self.sentence_model_name)
	
	def __len__(self):
		return (len(self.csv_file) - 1)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		doc = open(self.docs.iloc[idx, 0], 'r')
		ref = open(self.docs.iloc[idx, 1], 'r')
		doc = doc.read()
		ref = ref.read()
		doc_sents = get_sents(doc)
		ref_sents = get_sents(ref)
		doc_ids, doc_mask = get_tokens([doc], self.tokenizer)
		sent_ids, sent_mask = get_tokens(doc_sents, self.tokenizer)
		targets = get_targets(doc_sents, ref_sents, self.threshold)
		print(targets)
		data = {'sent_ids': sent_ids, 'doc_ids': doc_ids, 'sent_mask': sent_mask, 'doc_mask': doc_mask, 'targets': targets}
		return data

def get_targets(doc, ref, threshold):
	targets = []
	scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
	metric = 'rougeL'
	for sent in doc:
		target = 0
		max_score = 0
		for r in ref:
			score = scorer.score(sent, r)[metric].fmeasure
			if score > max_score:
				max_score = score
		if max_score >= threshold:
			target = 1
		targets.append(target)
	return np.asarray(targets, dtype=np.float32)

def get_sents(text):
	nlp = spacy.load('en_core_web_lg')
	sents = []
	for sent in nlp(text).sents:
		sents.append(str(sent))
	return sents


# sentence_model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
# tokenizer = AutoTokenizer.from_pretrained(sentence_model_name)

# Define GPU
use_gpu = 1
device = 'cpu'
if(use_gpu):
    device = 'cuda'
print(torch.cuda.get_device_name(torch.cuda.current_device()))

csv_name = './duc2004_dataset/DUC_dataset.csv'

training_dataset = DocumentDataset(csv_file=csv_name, threshold=0.35)
sent_ids, doc_ids, sent_mask, doc_mask, targets = training_dataset[0]
print(sent_ids)

# sentence_model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
# tokenizer = AutoTokenizer.from_pretrained(sentence_model_name)

# doc = open('./duc2004_dataset/docs/1001_Document.txt', 'r')
# doc = get_sents(doc.read())
# doc_ids, doc_mask = get_tokens(doc, tokenizer)
# print(doc_mask)



# scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
# metric = 'rougeL'
# sent_id = 0

# scores = []
# best_ref = ''
# for sent in doc:
# 	max_score = 0
# 	ref_id = 0
# 	best_ref_id = 0
# 	for r in ref:
# 		score = scorer.score(sent, r)[metric].fmeasure
# 		# 
# 		if best_ref == '':
# 			best_ref_id = ref_id
# 			best_ref = ref
# 		if score > max_score:
# 			max_score = score
# 			best_ref_id = ref_id
# 			best_ref = r
# 		ref_id += 1
# 	Sent = SentScore(sent_id, sent, best_ref_id, best_ref, round(max_score, 4))
# 	scores.append(Sent)
# 	sent_id += 1

# scores.sort(key=lambda x: x.score, reverse=True)
# for s in scores[0:5]:
# 	print(f'{s.sent_id}: {s.sent}')
# 	print(f'{s.ref_id}: {s.ref}')
# 	print('------------------------')
# 	printg(s.score)


# model = 'The quick brown fox jumps over the lazy dog.'
# ref = 'The fox'

# print(rouge.get_scores(model, ref, avg=True))





# sent_id, sent_mask = get_tokens(sentences, tokenizer)

# print(sent_id)
# print(sent_mask)
# print('---')

# doc_id, doc_mask = get_tokens([data], tokenizer)
# print(doc_id)
# print(doc_mask)
# print('---')

# # Match same doc array to each sentence in document
# doc_id, doc_mask = doc_id * len(sentences), doc_mask* len(sentences)
# print(doc_id)
# print(doc_mask)
# print('---')