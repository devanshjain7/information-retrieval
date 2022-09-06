from util import *
import numpy as np
# Add your import statements here
import math


class Evaluation():

	def __init__(self):
		self.NDCG = []

	def relDocList(self, qrels):
		relDoc = {}
		for dict_ in qrels:
			if dict_["query_num"] in relDoc.keys():
				relDoc[dict_["query_num"]].append((int(dict_["id"]), dict_["position"]))
			else:
				relDoc[dict_["query_num"]] = [(int(dict_["id"]), dict_["position"])]

		return relDoc

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		precision = -1
		n_rel = 0
		#Fill in code here

		for i in range(k):
			if query_doc_IDs_ordered[i] in true_doc_IDs:
				n_rel += 1
		
		precision = n_rel / k
		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		meanPrecision = -1
		sumPrecision = 0
		#Fill in code here

		relDoc = self.relDocList(qrels)

		for query_id in query_ids:
			true_doc_IDs = [tup[0] for tup in relDoc[str(query_id)]]
			queryPrec = self.queryPrecision(doc_IDs_ordered[query_id - 1], query_id, true_doc_IDs, k)
			sumPrecision += queryPrec

		meanPrecision = sumPrecision / len(query_ids)

		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		recall = -1
		n_rel = 0
		#Fill in code here

		for i in range(k):
			if query_doc_IDs_ordered[i] in true_doc_IDs:
				n_rel += 1
		
		recall = n_rel / len(true_doc_IDs)
		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		meanRecall = -1
		sumRecall = 0
		#Fill in code here
		relDoc = self.relDocList(qrels)

		for query_id in query_ids:
			true_doc_IDs = [tup[0] for tup in relDoc[str(query_id)]]
			queryRec = self.queryRecall(doc_IDs_ordered[query_id - 1], query_id, true_doc_IDs, k)
			sumRecall += queryRec

		meanRecall = sumRecall / len(query_ids)
		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		fscore = -1

		#Fill in code here
		queryPrec = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		queryRec = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)

		if queryPrec == 0 and queryRec == 0:
			fscore = 0
		else:
			fscore = (2*queryPrec*queryRec) / (queryPrec + queryRec)
		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		meanFscore = -1
		sumFscore = 0
		#Fill in code here
		relDoc = self.relDocList(qrels)

		for query_id in query_ids:
			true_doc_IDs = [tup[0] for tup in relDoc[str(query_id)]]
			queryF = self.queryFscore(doc_IDs_ordered[query_id - 1], query_id, true_doc_IDs, k)
			sumFscore += queryF

		meanFscore = sumFscore / len(query_ids)
		return meanFscore
	
	def position2score(self, position):
		if position == 1:
			return 4
		elif position == 2:
			return 3
		elif position == 3:
			return 2
		elif position == 4:
			return 1
		else:
			return None

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""

		nDCG = -1
		#Fill in code here
		
		DCG = 0
		for i in range(k):
			for j in true_doc_IDs:
				if query_doc_IDs_ordered[i] == j[0]:
					DCG += self.position2score(j[1]) / math.log2(i + 2)
					break

		sorted_tup = sorted(true_doc_IDs, key=lambda x: x[1])
		IDCG = 0
		for i in range(min(k, len(true_doc_IDs))):
			IDCG += self.position2score(sorted_tup[i][1]) / math.log2(i + 2)

		nDCG = DCG / IDCG
		self.NDCG.append(nDCG)
		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		meanNDCG = -1
		sumNDCG = 0

		#Fill in code here
		relDoc = self.relDocList(qrels)

		for query_id in query_ids:
			querynCDG = self.queryNDCG(doc_IDs_ordered[query_id - 1], query_id, relDoc[str(query_id)], k)
			sumNDCG += querynCDG

		meanNDCG = sumNDCG / len(query_ids)
		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

		avgPrecision = -1
		n_rel = 0
		sumPrec = 0
		#Fill in code here
		for i in range(k):
			if query_doc_IDs_ordered[i] in true_doc_IDs:
				n_rel += 1
				sumPrec += n_rel / (i + 1)

		if n_rel == 0:
			avgPrecision = 0
		else:
			avgPrecision = sumPrec / n_rel  

		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		meanAveragePrecision = -1
		sumAvgPrecision = 0
		#Fill in code here
		relDoc = self.relDocList(q_rels)

		for query_id in query_ids:
			true_doc_IDs = [tup[0] for tup in relDoc[str(query_id)]]
			queryAP = self.queryAveragePrecision(doc_IDs_ordered[query_id - 1], query_id, true_doc_IDs, k)
			sumAvgPrecision += queryAP

		meanAveragePrecision = sumAvgPrecision / len(query_ids)

		return meanAveragePrecision

