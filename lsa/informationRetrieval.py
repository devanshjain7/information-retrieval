from util import *

# Add your import statements here
import numpy as np
import pandas as pd
import itertools
from collections import Counter

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


	def svd(self, mat_orig):
		u, s, vh = np.linalg.svd(mat_orig)
		return u, s, vh

	def findBestApprx(self, mat_orig, u, s, vh):
		k = 200
		u_apprx = u[:, :k]
		s_apprx = np.diag(s[:k])
		vh_apprx = vh[:k, :]

		return u_apprx, s_apprx, vh_apprx
				
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
		wCols = []
		for id in docIDs:
			dataTable['w' + str(id)] = dataTable[id] * dataTable['IDF']
			wCols.append('w' + str(id))

		print("Data matrix created")

		mat_orig = np.array(dataTable[wCols])
		u, s, vh = self.svd(mat_orig)
		u_apprx, s_apprx, vh_apprx = self.findBestApprx(mat_orig, u, s, vh)
		vhDF = pd.DataFrame(vh_apprx, columns=wCols)

		for id in docIDs:
			normDict[id] = np.linalg.norm(vhDF['w' + str(id)])

		print("Calculating similarities for ranking")

		count = 1
		for query in queries:
			flat_query = list(itertools.chain(*query))

			counter = Counter(flat_query)
			qCol = pd.Series(counter, name='q')
			qDF = pd.DataFrame(qCol, index=index.keys())
			qDF.fillna(0, inplace=True)

			qDF['wq'] = qDF['q'] * dataTable['IDF']
			qVec = np.array(qDF['wq'])
			qTrans = np.matmul(np.linalg.inv(s_apprx), np.matmul(u_apprx.T, qVec))
			normQ = np.linalg.norm(qTrans)
			cosineSimList = []
			for id in docIDs:
				if normDict[id] == 0:
					continue
				sim = np.dot(vhDF['w' + str(id)], qTrans) / (normDict[id] * normQ)
				cosineSimList.append((id, sim))

			cosineSimList.sort(key=lambda x: x[1], reverse=True)
			QdocIDs = [x[0] for x in cosineSimList]
			doc_IDs_ordered.append(QdocIDs)
			count += 1
			if count % 15 == 0:
				print(f"{count} queries processed")

		return doc_IDs_ordered