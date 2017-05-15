# -*- coding: utf-8 -*-
import gensim
import os
from os import listdir
import collections
import smart_open
import random
import pdb
import glob
import numpy as np


def read_corpus(doc_list, labels_list, tokens_only=False):

    for idx, doc in enumerate(doc_list):

        if tokens_only:
            yield gensim.utils.simple_preprocess(doc.read())
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(words=gensim.utils.simple_preprocess(doc.read()), tags=[labels_list[idx]])

# pdb.set_trace()

txt_glob = './Documents/**/*.txt'
txt_files = glob.glob(txt_glob)

data = []
docLabels = []
for f in range(len(txt_files)):
    data.append(open(txt_files[f], "r"))

    lab = txt_files[f].split("/")[3].split(".")[0]
    docLabels.append(lab)

    # topic = txt_files[f].split("/")[2].lower()
    # if topic == "arts":
    #     labels.append(0)
    # elif topic == "food":
    #     labels.append(1)
    # elif topic == "politics":
    #     labels.append(2)
    # elif topic == "tech":
    #     labels.append(3)

a = ['anneImhof']
labelToIndx = dict()
for i, label in enumerate(docLabels):
        labelToIndx[label] = i

print('anneImhof has index ', labelToIndx[a[0]])
# print labelToIndx
# print(docLabels)

# pdb.set_trace()

# docLabels = []
# docLabels = [f for f in listdir("Documents") if f.endswith('.txt')]


# data = []
# for doc in docLabels:
#     data.append(open("Documents/" + doc, 'r'))

train_corpus = list(read_corpus(data, docLabels))
# test_corpus = list(read_corpus(data, docLabels, tokens_only=True))

# pdb.set_trace()
# print(train_corpus[:2])
# print(test_corpus[:2])

model = gensim.models.doc2vec.Doc2Vec(size=50, min_count=2, iter=550)

model.build_vocab(train_corpus)

print model.wv.vocab['and'].count

# print model.wv.vocab
print

model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)

print model.infer_vector(['only', 'you', 'can', 'prevent', 'forrest', 'fires'])

# print model.infer_vector(['computer', 'supercomputer', 'quantum', 'program', 'this', 'that'])


inferredVectors = np.empty(shape=(len(docLabels), model.vector_size))

ranks = []
second_ranks = []
# for doc_id in range(len(train_corpus)):
for doc_idx, doc_label in enumerate(docLabels):
    inferred_vector = model.infer_vector(train_corpus[doc_idx].words)

    inferredVectors[doc_idx] = inferred_vector

    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    rank = [docid for docid, sim in sims].index(doc_label)
    ranks.append(rank)

    second_ranks.append(sims[1])

print inferredVectors

print collections.Counter(ranks)  # Results vary due to random seeding and very small corpus

pdb.set_trace()

print sims[0][0]
# print train_corpus

print sims[0][0]
# for doc in train_corpus:
#     if doc[1] == ['vanGogh.txt']:
#         print doc[0]
#         break

print(u'Document ({}): «{}»\n'.format(doc_idx, ' '.join(train_corpus[doc_idx].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[labelToIndx[sims[index][0]]].words)))




# Pick a random document from the test corpus and infer a vector from the model
# doc_id = random.randint(0, len(train_corpus))
#
# # Compare and print the most/median/least similar documents from the train corpus
# print('Train Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
# sim_id = second_ranks[doc_id]
# print('Similar Document {}: «{}»\n'.format(sim_id, ' '.join(train_corpus[sim_id[0]].words)))



# Pick a random document from the test corpus and infer a vector from the model
# doc_id = random.randint(0, len(test_corpus))
# inferred_vector = model.infer_vector(test_corpus[doc_id])
# sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
#
# # Compare and print the most/median/least similar documents from the train corpus
# print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_corpus[doc_id])))
# print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
# for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
#     print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))
