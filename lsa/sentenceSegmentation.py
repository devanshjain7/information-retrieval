from util import *

# Add your import statements here
import nltk.data
nltk.download('punkt')
import re



class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""
		#Fill in code here
		segmentedText = re.split('[.?!:;—]', text.strip())


		return segmentedText





	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each strin is a single sentence
		"""

		#Fill in code here
		sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
		segmentedText = sent_tokenizer.tokenize(text.strip().lower())

		
		return segmentedText