from util import *

# Add your import statements here
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords



class StopwordRemoval():

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""
		#Fill in code here
		stop_words = set(stopwords.words('english'))
		stopwordRemovedText = []

		for list_ in text:
			filtered = [w for w in list_ if not w in stop_words]
			stopwordRemovedText.append(filtered)

		

		return stopwordRemovedText




	