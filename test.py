from methods.main_method.Kmeans_CentroidBase_MMR_SentencePosition import Summarizer
import argparse
from utils.preprocessing import Preprocessing

if __name__ =="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--cluster', help='cluster for Kmeans')
    parser.add_argument('--number_sentence_with_centroid', help='number sentence with centroid')
    parser.add_argument('--number_sentence_with_mmr', help='number sentence with mmr')
    parser.add_argument('--path_to_data', help='path to data')

    args = parser.parse_args()

    cluster = int(args.cluster)
    number_sentence_with_centroid = int(args.number_sentence_with_centroid)
    number_sentence_with_mmr = int(args.number_sentence_with_mmr)

    path = args.path_to_data

    sentences, last_indexs = Preprocessing().openDirectory(path, "test")

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

    summarizer = Summarizer(n_clusters=cluster
                            , len_summary=number_sentence_with_mmr)
    summary = summarizer.summary(sentences, text_sents, org_sents, last_indexs, number_sentence_with_centroid,
                                 mode="test")

    print (summary)