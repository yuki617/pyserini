def read_qrels(path):
    qrels = []

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            tokens = line.split(' ')
            topic = int(tokens[0])
            doc_id = tokens[-2]
            relevance = int(tokens[-1])
            qrels.append({
                'topic': topic,
                'doc_id': doc_id,
                'relevance': relevance
            })

    return qrels


def get_qrels_topics(path):
    qrels = read_qrels(path)
    topics = set()
    for pack in qrels:
        topics.add(pack['topic'])

    return topics


def get_X_Y_from_qrels_by_topic(path, topic, R):
    # always include topic 0
    R.append(0)
    qrels = [qrel for qrel in read_qrels(path) if qrel['topic'] == topic]
    qrels = [qrel for qrel in read_qrels(path) if qrel['relevance'] in R]
    x, y = [], []
    for pack in qrels:
        if pack['topic'] == topic:
            x.append(pack['doc_id'])
            label = 0 if pack['relevance'] == 0 else 1
            y.append(label)

    return x, y


def get_doc_ids_from_qrels_by_topic(path, topic):
    qrels = read_qrels(path)
    return [pack['doc_id'] for pack in qrels if pack['topic'] == topic]


def sort_dual_list(pred, docs):
    zipped_lists = zip(pred, docs)
    sorted_pairs = sorted(zipped_lists)

    tuples = zip(*sorted_pairs)
    pred, docs = [list(tuple) for tuple in tuples]

    return pred, docs
