# from summarizer import Summarizer
# from pprint import pprint
import spacy
# from transformers import AutoTokenizer, AutoModel
from rouge_score import rouge_scorer
import os

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

# sentence_model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
# tokenizer = AutoTokenizer.from_pretrained(sentence_model_name)
nlp = spacy.load('en_core_web_lg')

data_path = './datasets/DUC2004_Summarization_Documents/duc2004_testdata/tasks1and2/duc2004_tasks1and2_docs/docs/1001/'
data_files = ['D1.txt','D2.txt','D3.txt','D4.txt','D5.txt','D6.txt','D7.txt','D8.txt','D9.txt','D10.txt']

ref_path = './datasets/reference'
ref_files = ['Task1001_reference1.txt','Task1001_reference2.txt','Task1001_reference3.txt','Task1001_reference4.txt']


# summary1 = 'E:/ts/datasets/reference/Task1_reference2.txt'
# data = open(data_path, 'r')
# summary = open(summary1, 'r')
raw_doc = ''
for curr in data_files:
	file = os.path.join(data_path, curr)
	data = open(file, 'r')
	data = format_document(data)
	raw_doc += data

doc = cat_documents(data_path, data_files)
ref = cat_documents(ref_path, ref_files)

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
metric = 'rougeL'
sent_id = 0

scores = []
best_ref = ''
for sent in doc:
	max_score = 0
	ref_id = 0
	best_ref_id = 0
	for r in ref:
		score = scorer.score(sent, r)[metric].fmeasure
		# 
		if best_ref == '':
			best_ref_id = ref_id
			best_ref = ref
		if score > max_score:
			max_score = score
			best_ref_id = ref_id
			best_ref = r
		ref_id += 1
	Sent = SentScore(sent_id, sent, best_ref_id, best_ref, round(max_score, 4))
	scores.append(Sent)
	sent_id += 1

scores.sort(key=lambda x: x.score, reverse=True)
for s in scores[0:5]:
	print(f'{s.sent_id}: {s.sent}')
	print(f'{s.ref_id}: {s.ref}')
	print('------------------------')
	printg(s.score)


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