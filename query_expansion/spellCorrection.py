from util import *

from textblob import TextBlob

class SpellCorrection():

    def spellCorrect(self, text):

        spellCorrectText = []

        for sent in text:
            spellCorrectSent = []
            for word in sent:
                textBlb = TextBlob(word)
                correctWord = str(textBlb.correct())
                spellCorrectSent.append(correctWord)
            spellCorrectText.append(spellCorrectSent)

        return spellCorrectText