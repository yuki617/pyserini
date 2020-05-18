import sys
sys.path.insert(0, './')

import argparse
from sklearn.naive_bayes import MultinomialNB
import helper
from pyserini.vectorizer import TfidfVectorizer
from pyserini.search import pysearch
import os


# path variables
train_txt_path = 'nist/data/qrels_train_dev.txt'
test_txt_path = 'nist/data/qrels_test.txt'
lucene_index_path = 'nist/data/lucene-index-cord19-abstract-2020-05-01'

# get round 1 topics & detail
topics_dict = pysearch.get_topics('covid_round1')


def run(k, R, df):
    R_str = ''.join([str(i) for i in R])
    run_path = f'runs/tfidf.k{k}.R{R_str}.df{df}.txt'
    print('Outputing:', run_path)

    topics = helper.get_qrels_topics(train_txt_path)
    vectorizer = TfidfVectorizer(lucene_index_path, min_df=df)
    searcher = pysearch.SimpleSearcher(lucene_index_path) if k > 0 else None

    f = open(run_path, 'w+')
    for topic in topics:
        train_docs, train_labels = helper.get_X_Y_from_qrels_by_topic(
            train_txt_path, topic, R)
        print(f'[topic][{topic}] eligible train docs {len(train_docs)}')
        train_vectors = vectorizer.get_vectors(train_docs)

        # classifier training
        clf = MultinomialNB()
        clf.fit(train_vectors, train_labels)

        # search topic question
        test_docs = []
        if k > 0:
            question = topics_dict[topic]['question']
            hits = searcher.search(question, k=k)
            eligible_test_docs = helper.get_doc_ids_from_qrels_by_topic(
                test_txt_path, topic)
            test_docs = [
                hit.docid for hit in hits if hit.docid in eligible_test_docs]
        else:
            test_docs = helper.get_doc_ids_from_qrels_by_topic(
                test_txt_path, topic)
        print(f'[topic][{topic}] eligible test docs {len(test_docs)}')
        test_vectors = vectorizer.get_vectors(test_docs)

        # classifier inference
        probs = clf.predict_proba(test_vectors)
        # Extract prob of label 0
        probs = [row[0] for row in probs]
        # sort by increase order based on prob of label 0
        preds, docs = helper.sort_dual_list(probs, test_docs)

        for index, (score, doc_id) in enumerate(zip(preds, docs)):
            rank = index + 1
            score = 1 - score
            f.write(f'{topic} Q0 {doc_id} {rank} {score} tfidf\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='use tfidf vectorizer on cord-19 dataset with ccrf technique')
    parser.add_argument('--k', type=int, required=True, help='depth of search')
    parser.add_argument('--R', type=int, required=True,
                        nargs='+', help='Use docs with relevance 1s and/or 2s')
    parser.add_argument('--df', type=int, required=True,
                        help='min document frequency for tfidf')
    args = parser.parse_args()

    R = sorted([r for r in args.R if r == 1 or r == 2])
    run(args.k, R, args.df)
