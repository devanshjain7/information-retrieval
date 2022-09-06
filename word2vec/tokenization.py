from util import *

# Add your import statements here
from nltk.tokenize import RegexpTokenizer


class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""
		#Fill in code here
		tokenizedText = []
		for s in text:
			tokenizedText.append(s.strip().split())

		return tokenizedText



	def regExp(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		#Fill in code here
		tokenizer = RegexpTokenizer('\w+')
		tokenizedText = []
		for s in text:
			tokenizedText.append(tokenizer.tokenize(s))
		
		return tokenizedText