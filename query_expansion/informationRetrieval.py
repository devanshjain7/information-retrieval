from util import *

# Add your import statements here
import numpy as np
import pandas as pd
import itertools
from collections import Counter
from nltk.corpus import wordnet

class InformationRetrieval():

	def __init__(self):
		self.index = None

	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

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

		print("Building index started...")
		#Fill in code here
		for doc, docID in zip(docs, docIDs):
			for sent in doc:
				for word in sent:
					if word in index.keys():
						index[word].append(docID)
					else:
						index[word] = [docID]

		print("Index building completed")
		self.index = index
		self.docIDs = docIDs
	
	def queryExpansion(self, counter):
		
		alpha = 0.85
		expandedCounter = counter.copy()
		for k in counter.keys():
			synonymns = []
			count = 0
			for syn in wordnet.synsets(k):
				syn_name = syn.lemmas()[0].name().split('_')
				added = False
				for i in syn_name:
					if i not in synonymns:
						synonymns.append(i)
						added = True
						expandedCounter[i] = alpha*counter[k]
				if added:
					count += 1
				if count > 3:
					break

		return expandedCounter

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
		index = self.index
		docIDs = self.docIDs

		print("Creating data matrix...")

		data = [Counter(index[key]) for key in index.keys()]
		dataTable = pd.DataFrame(data, index=index.keys(), columns=docIDs)
		dataTable.fillna(0, inplace=True)
		dfSeries = dataTable.sum(axis=1)
		dataTable['IDF'] = np.log10(len(docIDs) / (dfSeries + 1)) + 1
		
		normDict = {}

		for id in docIDs:
			dataTable['w' + str(id)] = dataTable[id] * dataTable['IDF']
			normDict[id] = np.linalg.norm(dataTable['w' + str(id)])

		print("Data matrix created")

		print("Calculating similarities for ranking")

		count = 1
		for query in queries:
			flat_query = list(itertools.chain(*query))
			counter = Counter(flat_query)
			expandedCounter = self.queryExpansion(counter)
			qCol = pd.Series(expandedCounter, name='q')
			tableQ = pd.concat([dataTable, qCol], axis=1)
			tableQ.fillna(0, inplace=True)

			tableQ['wq'] = tableQ['q'] * tableQ['IDF']
			normQ = np.linalg.norm(tableQ['wq'])
			cosineSimList = []
			for id in docIDs:
				if normDict[id] == 0:
					continue
				sim = np.dot(tableQ['w' + str(id)], tableQ['wq']) / (normDict[id] * normQ)
				cosineSimList.append((id, sim))

			cosineSimList.sort(key=lambda x: x[1], reverse=True)
			QdocIDs = [x[0] for x in cosineSimList]
			doc_IDs_ordered.append(QdocIDs)
			count += 1
			if count % 15 == 0:
				print(f"{count} queries processed")

		return doc_IDs_ordered




