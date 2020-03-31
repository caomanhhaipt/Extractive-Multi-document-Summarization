import os
import re
import nltk
porter = nltk.PorterStemmer()
from utils.sentence import Sentence

class Preprocessing(object):
	def __init__(self):
		pass

	def processFile(self, file_path_and_name):
		try:

			f = open(file_path_and_name, 'r')
			text_0 = f.read()

			text_1 = re.search(r"<TEXT>.*</TEXT>", text_0, re.DOTALL)
			text_1 = re.sub("<TEXT>\n", "", text_1.group(0))
			text_1 = re.sub("\n</TEXT>", "", text_1)

			text_1 = re.sub("<P>", "", text_1)
			text_1 = re.sub("</P>", "", text_1)
			text_1 = re.sub("\n", " ", text_1)
			text_1 = re.sub("\"", "\"", text_1)
			text_1 = re.sub("''", "\"", text_1)
			text_1 = re.sub("``", "\"", text_1)
			text_1 = re.sub(" +", " ", text_1)
			text_1 = re.sub(" _ ", "", text_1)

			text_1 = re.sub(r"\(AP\) _", " ", text_1)
			text_1 = re.sub("&\w+;", " ", text_1)

			sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
			lines = sent_tokenizer.tokenize(text_1.strip())

			index = lines[0].find("--")
			if index != -1:
				lines[0] = lines[0][index + 2:]
			index = lines[0].find(" _ ")
			if index != -1:
				lines[0] = lines[0][index + 3:]
			sentences = []

			for sent in lines:
				sent = sent.strip()
				OG_sent = sent[:]
				sent = sent.lower()
				line = nltk.word_tokenize(sent)

				stemmed_sentence = [porter.stem(word) for word in line]
				stemmed_sentence = list(filter(lambda x: x != '.' and x != '`' and x != ',' and x != '_' and x != ';'
														 and x != '(' and x != ')' and x.find('&') == -1
														 and x != '?' and x != "'" and x != '!' and x != '''"'''
														 and x != '``' and x != '--' and x != ':'
														 and x != "''" and x != "'s", stemmed_sentence))

				# stemmed_sentence = [word for word in stemmed_sentence if word not in stopwords.words('english')]

				if (len(stemmed_sentence) <= 4):
					continue

				if stemmed_sentence:
					sentences.append(Sentence(file_path_and_name, stemmed_sentence, OG_sent))

			return sentences


		except IOError:
			print('Oops! File not found', file_path_and_name)
			return [Sentence(file_path_and_name, [], [])]

	def get_file_path(self, file_name):
		for root, dirs, files in os.walk(os.getcwd()):
			for name in files:
				if name == file_name:
					return os.path.join(root, name)
		print("Error! file was not found!!")
		return ""

	def get_all_files(self, path=None):
		retval = []

		if path == None:
			path = os.getcwd()

		for root, dirs, files in os.walk(path):
			for name in files:
				retval.append(os.path.join(root, name))
		return retval

	def openDirectory(self, path=None):
		file_paths = self.get_all_files(path)
		sentences = []
		last_indexs = []
		for file_path in file_paths:
			last_indexs.append(len(sentences))
			sentences = sentences + self.processFile(file_path)

		return sentences, last_indexs