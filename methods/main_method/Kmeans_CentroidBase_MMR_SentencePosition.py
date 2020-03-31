import os
import numpy as np
from definitions import ROOT_DIR
from utils.preprocessing import Preprocessing
from gensim.models import Word2Vec
import operator
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from methods.maximal_marginal_relevance.mmr_summarizer import MMR
from methods.centroid.centroid_summarizer import CentroidBow
import argparse

def makeSummary(sentences, best_sentence, query, summary_length, lambta, IDF):
    summary = [best_sentence]
    sum_len = 0

    while (sum_len < summary_length-1):
        MMRval = {}

        for sent in sentences:
            MMRval[sent] = MMR().MMRScore(sent, query, summary, lambta, IDF)

        maxxer = max(MMRval, key=MMRval.get)
        summary.append(maxxer)
        sentences.remove(maxxer)

        if len(sentences) == 0:
            break
        sum_len += 1

    return summary

class Summarizer(object):
    def __init__(self, n_clusters=35, len_summary=20):
        self.n_clusters = n_clusters
        self.len_summary = len_summary

    @staticmethod
    def find_position(index_currents, last_indexs):
        ranking = {}
        for index_current in index_currents:
            for index, item in enumerate(last_indexs):
                if index_current < item:
                    ranking[index_current] = index_current - last_indexs[index - 1]
                    break
            if index_current not in ranking:
                ranking[index_current] = index_current - max(last_indexs)

        sorted_ranking = sorted(ranking.items(), key=operator.itemgetter(1))

        indexs = []
        for item in sorted_ranking:
            indexs.append(item[0])

        return indexs

    def kmean_summarizer(self, sentences, text_sents, org_sents, last_indexs, n_clusters=30):
        model = Word2Vec(text_sents, min_count=1, size=256, cbow_mean=1)

        X = []
        for sent in text_sents:
            tmp = []
            count_word = 0
            for word in sent:
                count_word += 1
                if tmp == []:
                    tmp = np.array(list(model[word]))
                else:
                    tmp += np.array(list(model[word]))

            X.append(tmp)

        kmeans = KMeans(n_clusters=n_clusters)
        kmeans = kmeans.fit(X)

        avg = []
        for j in range(n_clusters):
            idx = np.where(kmeans.labels_ == j)[0]
            avg.append(np.mean(idx))
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
        ordering = sorted(range(n_clusters), key=lambda k: avg[k])

        index_currents = [closest[idx] for idx in ordering]
        position_index = Summarizer.find_position(index_currents, last_indexs)

        new_sents = [org_sents[idx] for idx in position_index]
        object_sents = [sentences[idx] for idx in position_index]

        return new_sents, object_sents

    def mmr_summarizer(self, object_sents, index_sents, mode="train", length_sentence=16):
        new_sents_mmr = []

        for item in index_sents:
            new_sents_mmr.append(object_sents[item])

        best_sent = new_sents_mmr[0]
        remaining_sents = new_sents_mmr[1:]

        IDF_w = MMR.IDFs(new_sents_mmr)

        if mode == "train":
            summary = makeSummary(remaining_sents, best_sent, best_sent, len(new_sents_mmr) - 4, 0.6, IDF_w)
        else:
            summary = makeSummary(remaining_sents, best_sent, best_sent, length_sentence, 0.6, IDF_w)

        mmr_summaries = []
        for item in summary:
            mmr_summaries.append(item.getOGwords())

        return mmr_summaries

    def add_position(self, new_sents, mmr_summaries, len_sent, index_sents, n_clusters):
        final_index = []
        for index, item in enumerate(new_sents):
            if item in mmr_summaries:
                final_index.append(index)

        number_add = len_sent - len(final_index)

        index_plus = []
        for item in range(n_clusters):
            if len(index_plus) == number_add:
                break

            if item not in index_sents:
                index_plus.append(item)

        final_index = sorted(final_index + index_plus)

        return final_index

    def centroid_summarizer(self, raw_sentences, new_sents, len_sent=20):
        centroid = CentroidBow()

        centroid_sents = centroid.sumarize(raw_sentences, new_sents, len_sent)

        new_indexs = []
        for item in centroid_sents:
            for index, item2 in enumerate(raw_sentences):
                if item == item2:
                    new_indexs.append(index)

        return new_indexs

    def summary(self, sentences, text_sents, org_sents, last_indexs, len_centroid=20, mode="train"):
        n_clusters = self.n_clusters
        len_sent = self.len_summary

        new_sents, object_sents = self.kmean_summarizer(sentences, text_sents, org_sents, last_indexs, n_clusters)

        raw_sentences = []

        for item in object_sents:
            raw_sentences.append(item.getOGwords())

        index_sents = self.centroid_summarizer(raw_sentences, new_sents, len_centroid)

        mmr_summaries = self.mmr_summarizer(object_sents, index_sents, mode)

        final_index = self.add_position(new_sents, mmr_summaries, len_sent, index_sents, n_clusters)

        final_summaries = []
        for item in final_index:
            final_summaries.append(new_sents[item])

        return final_summaries

if __name__ == "__main__":
    root_directory = ROOT_DIR + "/"

    doc_folders = os.listdir(root_directory + "Data/DUC_2007/Documents")

    summarizer = Summarizer(n_clusters=50
                            , len_summary=16)

    parser = argparse.ArgumentParser()

    parser.add_argument('--folder_to_save', help='Folder to save summaries')
    args = parser.parse_args()

    folder_to_save = args.folder_to_save
    path_to_save = root_directory + "Data/DUC_2007/" + folder_to_save + "/"

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    for folder in doc_folders:
        path = os.path.join(root_directory + "Data/DUC_2007/Documents/", '') + folder
        print (path)

        sentences, last_indexs = Preprocessing().openDirectory(path)
        text_sents = []
        for item in sentences:
            text_sents.append(item.getStemmedWords())

        clean_sents = []
        org_sents = []
        for item in sentences:
            org_sents.append(item.getOGwords())

            tmp = ""
            for word in item.getStemmedWords():
                tmp += word + " "

            if tmp[-1] not in clean_sents:
                clean_sents.append(tmp[:-1])

        summary = summarizer.summary(sentences, text_sents, org_sents, last_indexs)

        with open(path_to_save + folder + ".me", 'w') as fileOut:
            fileOut.write("\n".join(summary))