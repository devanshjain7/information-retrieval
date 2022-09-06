from util import *

# Add your import statements here
import numpy as np
import pandas as pd
import itertools
from collections import Counter
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api

class InformationRetrieval():

	def __init__(self):
		self.index = None

	def buildIndex(self, docs, docIDs):
		"""
		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""
		index = {}
		docForW2V = []
		print("Building index started...")
		#Fill in code here
		for doc, docID in zip(docs, docIDs):
			for sent in doc:
				docForW2V.append(sent)
				for word in sent:
					if word in index.keys():
						index[word].append(docID)
					else:
						index[word] = [docID]

		print("Index building completed")
		self.index = index
		# self.word2vec = api.load('glove-wiki-gigaword-100')
		# self.word2vec = Word2Vec(docForW2V, min_count=1, window=5, sg=1)
		self.word2vec = api.load('word2vec-google-news-300')
		self.docIDs = docIDs

	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query


		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered = []

		#Fill in code here
		word2vec = self.word2vec
		docIDs = self.docIDs
		index = self.index

		data = [Counter(index[key]) for key in index.keys()]
		dataTable = pd.DataFrame(data, index=index.keys(), columns=docIDs)
		dataTable.fillna(0, inplace=True)
		dfSeries = dataTable.sum(axis=1)
		dataTable['IDF'] = np.log10(len(docIDs) / (dfSeries + 1)) + 1

		docVec = {}
		normDict = {}

		for id in docIDs:
			vec = None
			for term in dataTable.index:
				if dataTable[id][term] != 0:
					if term in word2vec.key_to_index.keys():
						if vec is not None:
							vec += (dataTable['IDF'][term] * dataTable[id][term]) * word2vec[term]
						else:
							vec = (dataTable['IDF'][term] * dataTable[id][term]) * word2vec[term]
			
			docVec[id] = vec
			if vec is not None:
				normDict[id] = np.linalg.norm(vec)
			else:
				normDict[id] = 0

		print("Calculating similarities for ranking")

		count = 1
		for query in queries:
			flat_query = list(itertools.chain(*query))
			counter = Counter(flat_query)
			qCol = pd.Series(counter, name='q')
			tableQ = pd.concat([dataTable, qCol], axis=1)
			tableQ.fillna(0, inplace=True)
			tableQ = tableQ.loc[index.keys()]
			vec = None
			for term in tableQ.index:
				if tableQ['q'][term] != 0:
					if term in word2vec.key_to_index.keys():
						if vec is not None:
							vec += (tableQ['IDF'][term] * tableQ['q'][term]) * word2vec[term]
						else:
							vec = (tableQ['IDF'][term] * tableQ['q'][term]) * word2vec[term]
			normQ = np.linalg.norm(vec)

			cosineSimList = []
			for id in docIDs:
				if normDict[id] == 0:
					continue
				sim = np.dot(docVec[id], vec) / (normDict[id] * normQ)
				cosineSimList.append((id, sim))

			cosineSimList.sort(key=lambda x: x[1], reverse=True)
			QdocIDs = [x[0] for x in cosineSimList]
			doc_IDs_ordered.append(QdocIDs)
			count += 1
			if count % 15 == 0:
				print(f"{count} queries processed")

		return doc_IDs_ordered
