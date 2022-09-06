from util import *

# Add your import statements here
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

class InflectionReduction:

	def nltkTag2wordnetTag(self, nltk_tag):
		if nltk_tag.startswith('J'):
			return wordnet.ADJ
		elif nltk_tag.startswith('V'):
			return wordnet.VERB
		elif nltk_tag.startswith('N'):
			return wordnet.NOUN
		elif nltk_tag.startswith('R'):
			return wordnet.ADV
		else:          
			return None

	def reduce(self, text):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""

		#Fill in code here
		reducedText = []
		lemmatizer = WordNetLemmatizer()

		for list_ in text:
			nltk_tagged = nltk.pos_tag(list_)  

			wordnet_tagged = map(lambda x: (x[0], self.nltkTag2wordnetTag(x[1])), nltk_tagged)
			lemmatized_sent = []
			for word, tag in wordnet_tagged:
				if tag is None:
					lemmatized_sent.append(word)
				else:        
					lemmatized_sent.append(lemmatizer.lemmatize(word, tag))

			reducedText.append(lemmatized_sent)

		
		return reducedText


