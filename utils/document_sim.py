from utils.preprocessing import Preprocessing
import math

class DocumentSim(object):
	def __init__(self):
		self.text = Preprocessing()

	def TFs(self, sentences):

		tfs = {}
		for sent in sentences:
			wordFreqs = sent.getWordFreqs()

			for word in wordFreqs.keys():
				if tfs.get(word, 0) != 0:
					tfs[word] = tfs[word] + wordFreqs[word]
				else:
					tfs[word] = wordFreqs[word]
		return tfs

	def TFw(self, word, sentence):
		return sentence.getWordFreqs().get(word, 0)

	def IDFs(self, sentences):

		N = len(sentences)
		idfs = {}
		words = {}
		w2 = []

		for sent in sentences:
			for word in sent.getStemmedWords():
				if sent.getWordFreqs().get(word, 0) != 0:
					words[word] = words.get(word, 0) + 1

		for word in words:
			n = words[word]
			try:
				w2.append(n)
				idf = math.log10(float(N) / n)
			except ZeroDivisionError:
				idf = 0

			idfs[word] = idf

		return idfs

	def IDF(self, word, idfs):
		return idfs[word]

	def sim(self, sentence1, sentence2, idfs):

		numerator = 0
		denom1 = 0
		denom2 = 0

		for word in sentence2.getStemmedWords():
			numerator += self.TFw(word, sentence2) * self.TFw(word, sentence1) * self.IDF(word, idfs) ** 2

		for word in sentence1.getStemmedWords():
			denom2 += (self.TFw(word, sentence1) * self.IDF(word, idfs)) ** 2

		for word in sentence2.getStemmedWords():
			denom1 += (self.TFw(word, sentence2) * self.IDF(word, idfs)) ** 2

		try:
			return numerator / (math.sqrt(denom1) * math.sqrt(denom2))

		except ZeroDivisionError:
			return float("-inf")