import math
from utils.sentence import Sentence

class MMR(object):
    def __init__(self):
        pass

    @staticmethod
    def TFs(sentences):
        tfs = {}

        for sent in sentences:
            wordFreqs = sent.getWordFreqs()

            for word in wordFreqs.keys():
                if tfs.get(word, 0) != 0:
                    tfs[word] = tfs[word] + wordFreqs[word]
                else:
                    tfs[word] = wordFreqs[word]
        return tfs

    @staticmethod
    def IDFs(sentences):
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

    @staticmethod
    def TF_IDF(sentences):
        tfs = MMR.TFs(sentences)
        idfs = MMR.IDFs(sentences)
        retval = {}

        for word in tfs:
            tf_idfs = tfs[word] * idfs[word]

            if retval.get(tf_idfs, None) == None:
                retval[tf_idfs] = [word]
            else:
                retval[tf_idfs].append(word)

        return retval

    @staticmethod
    def sentenceSim(sentence1, sentence2, IDF_w):
        numerator = 0
        denominator = 0

        for word in sentence2.getStemmedWords():
            numerator += sentence1.getWordFreqs().get(word, 0) * sentence2.getWordFreqs().get(word, 0) * IDF_w.get(word,
                                                                                                                 0) ** 2

        for word in sentence1.getStemmedWords():
            denominator += (sentence1.getWordFreqs().get(word, 0) * IDF_w.get(word, 0)) ** 2

        try:
            return numerator / math.sqrt(denominator)
        except ZeroDivisionError:
            return float("-inf")

    def buildQuery(self, sentences, TF_IDF_w, n):
        scores = list(TF_IDF_w.keys())
        scores.sort(reverse=True)

        i = 0
        j = 0
        queryWords = []

        while (i < n):
            words = TF_IDF_w[scores[j]]
            for word in words:
                queryWords.append(word)
                i = i + 1
                if (i > n):
                    break
            j = j + 1

        return Sentence("query", queryWords, queryWords)

    @staticmethod
    def bestSentence(sentences, query, IDF):
        best_sentence = None
        maxVal = float("-inf")

        for sent in sentences:
            similarity = MMR.sentenceSim(sent, query, IDF)

            if similarity > maxVal:
                best_sentence = sent
                maxVal = similarity
        sentences.remove(best_sentence)

        return best_sentence

    def makeSummary(self, sentences, best_sentence, query, summary_length, lambta, IDF):
        summary = [best_sentence]
        sum_len = len(best_sentence.getStemmedWords())

        while (sum_len < summary_length):
            MMRval = {}

            for sent in sentences:
                MMRval[sent] = self.MMRScore(sent, query, summary, lambta, IDF)

            maxxer = max(MMRval, key=MMRval.get)
            summary.append(maxxer)
            sentences.remove(maxxer)
            sum_len += len(maxxer.getStemmedWords())

        return summary

    def MMRScore(self, Si, query, Sj, lambta, IDF):
        Sim1 = MMR.sentenceSim(Si, query, IDF)
        l_expr = lambta * Sim1
        value = [float("-inf")]

        for sent in Sj:
            Sim2 = MMR.sentenceSim(Si, sent, IDF)
            value.append(Sim2)

        r_expr = (1 - lambta) * max(value)
        MMR_SCORE = l_expr - r_expr

        return MMR_SCORE