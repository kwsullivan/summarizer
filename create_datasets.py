import spacy
from rouge_score import rouge_scorer
import os
import sys
import matplotlib.pyplot as plt
import statistics

spacy.prefer_gpu()

def cat_documents(path, files):
    nlp = spacy.load('en_core_web_lg')
    raw = ''
    for curr in files:
        file = os.path.join(path, curr).replace("\\","/")
        data = open(file, 'r')
        data = format_document(data)
        raw += data

    cat = []
    for sent in nlp(raw).sents:
        cat.append(str(sent))
    return cat

def format_document(document):
    document = document.read().replace('\n', '')
    document = document.replace('\ufeff', '')
    document = document.replace('`', '')
    document = document.replace('\'\'', '')
    return document

def sentencify(data):
    nlp = spacy.load('en_core_web_lg')
    cat = []
    for sent in nlp(data).sents:
       cat.append(str(sent))
    return cat

def printg(text):
    print('\033[1;32;40m', end='')
    print(text)
    print('\033[1;37;40m', end='')

class SentScore:
    def __init__(self, doc_id, sent_id, sent, ref, score):
        self.doc_id = doc_id
        self.sent = sent
        self.sent_id = sent_id
        self.ref = ref
        self.score = score


test = './datasets/DUC2004_Summarization_Documents/duc2004_testdata/tasks1and2/duc2004_tasks1and2_docs/docs'
data_path = './datasets/DUC2004_Summarization_Documents/duc2004_testdata/tasks1and2/duc2004_tasks1and2_docs/docs'
data_folders = ['1001', '1008', '1009', '1013', '1022', '1026', '1031', '1032', '1033', '1038', '1043', '1050']
data_files = ['D1.txt','D2.txt','D3.txt','D4.txt','D5.txt','D6.txt','D7.txt','D8.txt','D9.txt','D10.txt']

ref_path = './datasets/reference'
ref_files_raw = ['reference1.txt','reference2.txt','reference3.txt','reference4.txt']

CURRENT_DOC = sys.argv[1]

doc = cat_documents(os.path.join(data_path, CURRENT_DOC).replace("\\","/"), data_files)

ref_files = []
for r in ref_files_raw:
    ref_name = 'Task' + CURRENT_DOC + '_' + r
    ref_files.append(ref_name)

ref = cat_documents(ref_path, ref_files)

# docs = []
# for doc in data_folders:
#     file = os.path.join(data_path, doc).replace("\\","/")
#     for darr in data_files:
#         dfile = open(os.path.join(file, darr).replace("\\","/"), 'r')
#         data = format_document(dfile)
#     docs.append(sentencify(data))

# refs = []
# for doc in data_folders:
#     for ref in ref_files:
#         ref_name = 'Task' + doc + '_' + ref
#         rfile = open(os.path.join(ref_path, ref_name).replace("\\","/"), 'r')
#         data = format_document(rfile)
#     refs.append(sentencify(data))


scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
metric = 'rougeL'

scores = []
sent_id = 0
for sent in doc:
    max_score = 0
    best_ref = ''
    for r in ref:
        score = scorer.score(sent, r)[metric].fmeasure
        if best_ref == '':
            best_ref = ref
        if score > max_score:
            max_score = score
            best_ref = r
    Sent = SentScore(CURRENT_DOC, sent_id, sent, best_ref, round(max_score, 4))
    scores.append(Sent)
    sent_id += 1

_ids = [s.sent_id for s in scores]
_scores = [s.score for s in scores]
avg_score = sum(_scores) / len(_scores)
print(f'Document {CURRENT_DOC}\n---------')
print(f'Average:\t\t{avg_score}')
print(f'Median:\t\t\t{statistics.median(_scores)}')
fig = plt.figure(figsize =(10, 7))
plt.bar(_ids, _scores, color='maroon', width=1)
plt.title(f'DUC Document {CURRENT_DOC}')
plt.xlabel('Sentence Number')
plt.ylabel('Rouge Score')
plt.savefig(f'./figures/{CURRENT_DOC}_bar.png', bbox_inches='tight')

scores.sort(key=lambda x: x.score, reverse=True)
print(f'Range of Top 10:\t{scores[9].score}-{scores[0].score}')

# for s in scores:
#     print(f'{s.doc_id}')
#     print(f'{s.sent}')
#     print(f'{s.ref}')
#     printg(s.score)
#     print('------------------------')


# scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
# metric = 'rougeL'
# scores = []
# doc_id = 0
# for doc in docs:
#     sent_scores = []
#     for sent in doc:
#         max_score = 0
#         ref_id = 0
#         for ref in refs:
#             best_ref_id = 0
#             best_ref = ''
#             for r in ref:
#                 score = scorer.score(sent, r)[metric].fmeasure
#                 if best_ref == '':
#                     best_ref = ref
#                 if score > max_score:
#                     max_score = score
#                     best_ref = r
#             Sent = SentScore(doc_id, sent, best_ref, round(max_score, 4))
#             sent_scores.append(Sent)
#     scores.append(sent_scores)
#     doc_id += 1

# for sent_scores in scores:
#     for s in sent_scores:
#         print(f'{s.sent} - {s.score}')
# for sent_scores in scores[0]:
#     sent_scores.sort(key=lambda x: x.score, reverse=True)
#     for s in sent_scores:
#         print(f'{s.sent}')
#         print(f'{s.ref}')
#         print('------------------------')
#         printg(s.score)

# iter = 0
# for sent_scores in scores:
#     sent_scores.sort(key=lambda x: x.score, reverse=True)
#     printg(f'DOC ID:: {iter}')
#     for s in sent_scores:
#         print(f'{s.sent_id}: {s.sent}')
#         print(f'{s.ref_id}: {s.ref}')
#         print('------------------------')
#         printg(s.score)
#     iter += 1