import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cosine

class CentroidBow(object):
    def __init__(self, limit=250, topic_threshold=0.3, sim_threshold=0.95, limit_type='word'):
        self.limit = limit
        self.topic_threshold = topic_threshold
        self.sim_threshold = sim_threshold
        self.limit_type = limit_type

    @staticmethod
    def similarity(v1, v2):
        score = 0.0
        if np.count_nonzero(v1) != 0 and np.count_nonzero(v2) != 0:
            score = ((1 - cosine(v1, v2)) + 1) / 2
        return score

    def sumarize(self, raw_sentences, clean_sentences, max_len):
        vectorizer = CountVectorizer(stop_words='english')
        sent_word_matrix = vectorizer.fit_transform(clean_sentences)

        transformer = TfidfTransformer(norm=None, sublinear_tf=False, smooth_idf=False)
        tfidf = transformer.fit_transform(sent_word_matrix)
        tfidf = tfidf.toarray()

        centroid_vector = tfidf.sum(0)
        centroid_vector = np.divide(centroid_vector, centroid_vector.max())
        for i in range(centroid_vector.shape[0]):
            if centroid_vector[i] <= self.topic_threshold:
                centroid_vector[i] = 0

        sentences_scores = []
        for i in range(tfidf.shape[0]):
            score = self.similarity(tfidf[i, :], centroid_vector)
            sentences_scores.append((i, raw_sentences[i], score, tfidf[i, :]))

        sentence_scores_sort = sorted(sentences_scores, key=lambda el: el[2], reverse=True)

        sentences_summary = []

        for s in sentence_scores_sort:
            if len(sentences_summary) == max_len:
                break
            include_flag = True
            for ps in sentences_summary:
                sim = CentroidBow().similarity(s[3], ps[3])

                if sim > CentroidBow().sim_threshold:
                    include_flag = False
            if include_flag:
                sentences_summary.append(s)

        results = []
        for item in sentences_summary:
            results.append(item[1])

        return results