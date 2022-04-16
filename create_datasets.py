import spacy
from rouge_score import rouge_scorer
import os
import sys
import matplotlib.pyplot as plt
import statistics
import csv

spacy.prefer_gpu()

def cat_documents(path, files):
    nlp = spacy.load('en_core_web_lg')
    raw = ''
    for curr in files:
        file = os.path.join(path, curr).replace("\\","/")
        data_file = open(file, 'r')
        data = format_document(data_file)
        raw += data

    cat = []
    for sent in nlp(raw).sents:
        cat.append(str(sent))
    data_file.close()
    return cat

def format_document(document):
    document = document.read().replace('\n', '')
    document = document.replace('\ufeff', '')
    document = document.replace('`', '')
    document = document.replace('\'\'', '')
    return document

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

def printg(text):
    print('\033[1;32;40m', end='')
    print(text)
    print('\033[1;37;40m', end='')

class SentScore:
    def __init__(self, doc_id, sent_id, sent, ref_id, ref, score):
        self.doc_id = doc_id
        self.sent_id = sent_id
        self.sent = sent
        self.ref_id = ref_id
        self.ref = ref
        self.score = score
        self.valid = 1
graph = 0
metrics = 0
spreadsheet = 0
CURRENT_DOC = ''
for param in sys.argv:
    if param == '-g':
        graph = 1
    if '-' not in param:
        CURRENT_DOC = param
    if param == '-m':
        metrics = 1
    if param == '-s':
        spreadsheet = 1

if CURRENT_DOC == '':
    print('Error: Enter valid data directory.')
    sys.exit(1)

# Adjust for number of top ranking sentences
TOP_RANGE = 10

data_path = './datasets/DUC-2004-Dataset/DUC2004_Summarization_Documents/duc2004_testdata/tasks1and2/duc2004_tasks1and2_docs/docs'
data_folders = ['1001', '1008', '1009', '1013', '1022', '1026', '1031', '1032', '1033', '1038', '1043', '1050']
data_files = ['D1.txt','D2.txt','D3.txt','D4.txt','D5.txt','D6.txt','D7.txt','D8.txt','D9.txt','D10.txt']
document_name = f'./datasets/DUC-2004-Dataset/docs/{CURRENT_DOC}_Document.txt'
reference_name = f'./datasets/DUC-2004-Dataset/refs/{CURRENT_DOC}_Reference.txt'
ref_path = './datasets/DUC-2004-Dataset/reference'
ref_files_raw = ['reference1.txt','reference2.txt','reference3.txt','reference4.txt']

doc = cat_documents(os.path.join(data_path, CURRENT_DOC).replace("\\","/"), data_files)

ref_files = []
for r in ref_files_raw:
    ref_name = 'Task' + CURRENT_DOC + '_' + r
    ref_files.append(ref_name)

ref = cat_documents(ref_path, ref_files)

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
metric = 'rougeL'

scores = []
sent_id = 0
for sent in doc:
    max_score = 0
    best_ref = ''
    ref_id = 0
    best_ref_id = 0
    for r in ref:
        score = scorer.score(sent, r)[metric].fmeasure
        if score > max_score:
            max_score = score
            best_ref_id = ref_id
            best_ref = r
        ref_id += 1
    Sent = SentScore(CURRENT_DOC, sent_id, sent, best_ref_id, best_ref, round(max_score, 4))
    scores.append(Sent)
    sent_id += 1


_ids = [s.sent_id for s in scores]
_scores = [s.score for s in scores]
sorted_scores = scores
avg_score = sum(_scores) / len(_scores)
print(f'Document {CURRENT_DOC}\n')
print(f'Average:\t\t{round(avg_score, 4)}')
print(f'Median:\t\t\t{statistics.median(_scores)}\n')

sorted_scores.sort(key=lambda x: x.score, reverse=True)

print(f'Range of Top {TOP_RANGE}:\t{sorted_scores[(TOP_RANGE - 1)].score:.4f}-{sorted_scores[0].score:.4f}')
top_scores = [s.score for s in sorted_scores[0:TOP_RANGE]]
print(list(reversed(top_scores)))
top_avg = sum(top_scores) / len(top_scores)
top_med = statistics.median(top_scores)
print(f'Top {TOP_RANGE} Average:\t\t{round(top_avg, 4):.4f}')
print(f'Top {TOP_RANGE} Median:\t\t{top_med:.4f}')
print('-----------')
scores.sort(key=lambda x: x.sent_id)

if graph:
    fig = plt.figure(figsize =(10, 7))
    plt.bar(_ids, _scores, color='maroon', width=1)
    plt.axhline(y=top_avg,linewidth= 3, color='blue', linestyle= 'solid')
    plt.axhline(y=top_med,linewidth= 3, color='green', linestyle= 'solid')
    plt.title(f'DUC Document {CURRENT_DOC}')
    plt.xlabel('Sentence Number')
    plt.ylabel('Rouge Score')
    plt.savefig(f'./figures/{CURRENT_DOC}_bar.png', bbox_inches='tight')

if spreadsheet:
    threshold = top_med
    csv_name = './datasets/DUC_dataset.csv'
    with open(csv_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        targets = []
        for s in scores:
            target = 0
            if s.score >= threshold:
                target = 1
            targets.append(target)
        writer.writerow({document_name, str(targets)})


# scores.sort(key=lambda x: x.score, reverse=True)
# top_scores = [s for s in scores[0:(TOP_RANGE - 1)]]

# top_scores.sort(key=lambda x: x.ref_id)

# final = ''
# for s in top_scores:
#     final += s.sent + ' '
#     print(f'- Sentence {s.sent_id} -')
#     print(s.sent)
#     print(f'- Reference {s.ref_id} -')
#     print(s.ref)
#     print(f'Score: {s.score}')

# print(final)
# for s in scores:
#     print(f'{s.doc_id}')
#     print(f'{s.sent}')
#     print(f'{s.ref}')
#     printg(s.score)
#     print('------------------------')

# Only use unique best references ----------------------------------------------------- MAYBE?
# remove_list =[]
# for r in range(len(ref)):
#     sent_refs = []
#     for s in scores:
#         if s.ref_id == r:
#             sent_refs.append(s.sent_id)
#     if len(sent_refs) > 1:
#         best_score = 0
#         best_sent = 0
#         max_score = 0
#         for i in sent_refs:
#             score = scores[i].score
#             if score >= max_score:
#                 max_score = score
#                 best_sent = scores[i].sent_id
#         # print(f'Reference {r}: {scores[best_sent].sent} - {scores[best_sent].score}')

#         # Remove best score for reference and delete the rest
#         sent_refs.remove(best_sent)
#         remove_list.append(sent_refs)
#         for r in sent_refs:
#             scores[r].valid = 0

# _ids = [s.sent_id for s in filter(lambda v: v.valid, scores)]
# _scores = [s.score for s in filter(lambda v: v.valid, scores)]