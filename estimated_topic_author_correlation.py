import argparse
import numpy as np
import sys
import timeit

from collections import Counter
from gensim.models import LdaModel
from scipy.sparse import lil_matrix
from scipy.special import xlogy

from downsample_corpus import get_vocab
from topic_author_correlation import get_author_info
from topic_author_correlation import evaluate_correlation


# Get document-topic distributions (theta) for a gensim LDA model.
def get_gensim_thetas(in_tsv, vocab_index, gensim_ldamodel):
    n_topics = gensim_ldamodel.num_topics

    thetas = []
    reader = open(in_tsv, mode='r', encoding='utf-8')
    for doc_id, line in enumerate(reader):
        fields = line.strip().split('\t')
        token_counter = Counter(fields[2].split())
        doc_bow = [(vocab_index[term], count) for term, count in
                   token_counter.items()]
        doc_thetas = np.zeros(n_topics)
        for topic, prob in gensim_ldamodel.get_document_topics(doc_bow):
            doc_thetas[doc_id, topic] = prob
        thetas.append(doc_thetas)
    reader.close()
    thetas = np.vstack(thetas)
    return thetas


# Get document-topic distributions (theta) for MALLET LDA model.
def get_mallet_thetas(doc_topics_fn):
    thetas = []
    for line in open(doc_topics_fn, mode='r'):
        fields = line.strip().split()
        row = [float(f) for f in fields[2:]]
        thetas.append(row)
    thetas = np.array(thetas)
    thetas = thetas / thetas.sum(axis=1, keepdims=True)
    return thetas


# Get topic-term distributions (phi) for MALLET LDA model.
def get_mallet_phis(word_weights_fn, vocab_index, n_topics):
    n_vocab = len(vocab_index)
    phis = np.zeros((n_topics, n_vocab))
    for line in open(word_weights_fn, mode='r', encoding='utf-8'):
        fields = line.strip().split('\t')
        topic = int(fields[0])
        term = fields[1]
        weight = float(fields[2])
        phis[topic, vocab_index[term]] = weight
    phis = phis / phis.sum(axis=1, keepdims=True)
    return phis


# Estimate topic counts by performing a round of Gibbs sampling using the
# inferred thetas and phis to generate topic-token assignments.
def estimate_topic_counts(in_tsv, vocab_index, author_index, thetas, phis,
                          verbose=False):
    n_authors = len(author_index)
    n_docs, n_topics = thetas.shape
    n_topics, n_vocab = phis.shape
    topic_term_counts = np.zeros((n_topics, n_vocab))
    topic_author_term_counts = [lil_matrix((n_authors, n_vocab))
                                for t in range(n_topics)]
    nz_phis = phis > 0
    log_phis = xlogy(nz_phis, phis)
    for doc_id, line in enumerate(open(in_tsv, mode='r', encoding='utf-8')):
        if verbose and doc_id and doc_id % 1000 == 0:
            print('{}/{}'.format(doc_id, n_docs), file=sys.stderr)
        fields = line.strip().split('\t')
        author = fields[1]
        author_id = author_index[author]
        tokens = np.array(fields[2].split())
        theta_d = thetas[doc_id]
        nz_theta_d = theta_d > 0
        log_theta_d = xlogy(nz_theta_d, theta_d)

        for term, count in Counter(tokens).items():
            term_id = vocab_index[term]
            topic_dist = np.where(nz_phis.T[term_id] * nz_theta_d != 0,
                                  np.exp(log_phis.T[term_id] + log_theta_d),
                                  0.0).ravel()
            topic_dist = topic_dist / topic_dist.sum()
            topics = np.random.choice(n_topics, size=count, p=topic_dist)
            for topic in topics:
                topic_term_counts[topic, term_id] += 1
                topic_author_term_counts[topic][author_id, term_id] += 1
    topic_author_term_counts = [x.tocsr() for x in topic_author_term_counts]
    return topic_term_counts, topic_author_term_counts


if __name__ == '__main__':
    # Top-level parser
    parser = argparse.ArgumentParser(
        description='Estimate topic-author correlation for a MALLET or ' +
                    'gensim trained topic model.',
        add_help=False)
    subparsers = parser.add_subparsers(help='Tool used to train topic model.')

    # gensim sub-parser
    gensim_parser = subparsers.add_parser('gensim', add_help=False)
    gensim_parser.add_argument('-h', '--help', action='help',
                               help='Show this help message and exit.')
    required_gensim = gensim_parser.add_argument_group('required arguments')
    required_gensim.add_argument('--input', metavar='FILE',
                                 dest='in_tsv', required=True,
                                 help='The file of the collection which the ' +
                                      'topic model was trained on.')
    required_gensim.add_argument('--lda-model', dest='ldamodel_fn',
                                 metavar='FILE', required=True,
                                 help='The file containing the saved gensim ' +
                                      'ldamodel.')
    required_gensim.add_argument('--output', metavar='FILE',
                                 dest='out_tsv', required=True,
                                 help='Write the estimated topic-author ' +
                                      'correlation results to this file.')
    gensim_parser.add_argument('-v', '--verbose', action='store_true',
                               dest='verbose')

    # MALLET sub-parser
    mallet_parser = subparsers.add_parser('mallet', add_help=False)
    mallet_parser.add_argument('-h', '--help', action='help',
                               help='Show this help message and exit.')
    required_mallet = mallet_parser.add_argument_group('required arguments')
    required_mallet.add_argument('--input', metavar='FILE',
                                 dest='in_tsv', required=True,
                                 help='The file of the collection which the ' +
                                      'topic model was trained on.')
    required_mallet.add_argument('--topic-word-weights', metavar='FILE',
                                 dest='word_weights_fn', required=True,
                                 help='MALLET-generated topic word weights ' +
                                      'file produced by --topic-word-' +
                                      'weights-file option.')
    required_mallet.add_argument('--doc-topics', metavar='FILE',
                                 dest='doc_topics_fn', required=True,
                                 help='MALLET-generated doc topics file ' +
                                      'produced by --output-doc-topics ' +
                                      'option.')
    required_mallet.add_argument('--output', metavar='FILE',
                                 dest='out_tsv', required=True,
                                 help='Write the estimated topic-author ' +
                                      'correlation results to this file.')
    required_mallet.add_argument('--vocab', metavar='FILE',
                                 dest='vocab_fn', required=True,
                                 help='File of the working vocabulary, ' +
                                      'one word per line.')
    mallet_parser.add_argument('-v', '--verbose', action='store_true',
                               dest='verbose')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit()

    tool = sys.argv[1]
    if len(sys.argv) == 2:
        if tool == 'gensim':
            gensim_parser.print_help(sys.stderr)
            sys.exit()
        elif tool == 'mallet':
            mallet_parser.print_help(sys.stderr)
            sys.exit()

    args = parser.parse_args()

    authors, author_index, doc_author_ids = get_author_info(args.in_tsv)

    vocab_index, phis, thetas = [None, None, None]
    print('Building phi and theta matrices')
    if tool == 'gensim':
        lda_model = LdaModel.load(args.ldamodel_fn)
        vocab_index = {term: i for i, term in lda_model.id2word.items()}
        thetas = get_gensim_thetas(args.in_tsv, vocab_index, lda_model)
        phis = lda_model.get_topics()
        phis = phis / phis.sum(axis=1, keepdims=True)
    elif tool == 'mallet':
        vocab, vocab_index = get_vocab(args.vocab_fn)
        thetas = get_mallet_thetas(args.doc_topics_fn)
        n_topics = thetas.shape[1]
        phis = get_mallet_phis(args.word_weights_fn, vocab_index, n_topics)
    print('Building topic-author-term counts')
    (topic_term_counts,
     topic_author_term_counts) = estimate_topic_counts(args.in_tsv,
                                                       vocab_index,
                                                       author_index,
                                                       thetas,
                                                       phis,
                                                       verbose=args.verbose)
    print('Evaluating topic-author correlation')
    evaluate_correlation(topic_term_counts, topic_author_term_counts,
                         args.out_tsv)
