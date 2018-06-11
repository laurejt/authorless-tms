import argparse
import gzip
import numpy as np
import sys

from scipy.sparse import lil_matrix
from scipy.stats import entropy

from downsample_corpus import get_vocab


# Read in author information from input corpus.
def get_author_info(in_tsv):
    doc_authors = []
    for line in open(in_tsv, mode='r', encoding='utf-8'):
        if not line.strip():
            continue
        fields = line.strip().split('\t')
        doc_authors.append(fields[1])
    authors = sorted(list(set(doc_authors)))
    author_index = {author: i for i, author in enumerate(authors)}
    doc_author_ids = [author_index[a] for a in doc_authors]
    return authors, author_index, doc_author_ids


# Build topic-author-term counts from a MALLET state file and
# its corresponding vocabulary index and document-author mapping.
def process_state_file(state_fn, vocab_index, doc_author_ids, verbose=False):
    n_vocab = len(vocab_index)
    n_authors = max(doc_author_ids) + 1

    reader = gzip.open(state_fn, mode='rt', encoding='utf-8')
    reader.readline()  # header
    alpha_text = reader.readline().strip()
    alphas = [float(a) for a in alpha_text.split(' : ')[1].split()]
    n_topics = len(alphas)
    beta_text = reader.readline().strip()
    beta = float(beta_text.split(' : ')[1])

    topic_term_counts = np.zeros((n_topics, n_vocab))
    topic_author_term_counts = [lil_matrix((n_authors, n_vocab))
                                for t in range(n_topics)]
    current_doc_id = -1
    for i, line in enumerate(reader):
        fields = line.strip().split()
        doc_id = int(fields[0])
        if verbose and doc_id and doc_id % 1000==0:
            if doc_id != current_doc_id:
                print(doc_id, file=sys.stderr)
                current_doc_id = doc_id
        term = fields[4]
        topic = int(fields[5])
        author_id = doc_author_ids[doc_id]
        term_idx = vocab_index[term]

        topic_term_counts[topic, term_idx] += 1
        topic_author_term_counts[topic][author_id, term_idx] += 1
    reader.close()
    topic_author_term_counts = [x.tocsr() for x in topic_author_term_counts]
    return topic_term_counts, topic_author_term_counts


# Calculate Author Entropy given a topic's author-term counts.
def author_entropy(author_term_counts):
    author_counts = author_term_counts.sum(axis=1).getA1()
    author_dists = author_counts / author_counts.sum()
    return entropy(author_dists)


# Calculate the Jensen-Shannon divergence (base 2) of two probablity
# distributions p and q.
def jsd(p, q):
    pq_sum = 0.5*(p + q)
    return 0.5*(entropy(p, pq_sum, base=2) +
                entropy(q, pq_sum, base=2))


# Calculate Minus Major Author given a topic's term probability
# distribution and its author-term counts.
def minus_major_author(topic_term_dist, author_term_counts):
    max_author = author_term_counts.sum(axis=1).argmax()
    minus_major_counts = (author_term_counts.sum(axis=0) -
                          author_term_counts[max_author])
    minus_major_dist = minus_major_counts / minus_major_counts.sum()
    return jsd(topic_term_dist, minus_major_dist.getA1())


# Calculate Balance Authors given a topic's term probability
# distribution and its author-term counts.
def balanced_authors(topic_term_dist, author_term_counts):
    nz_rows = author_term_counts[author_term_counts.getnnz(axis=1) > 0]
    nz_author_dists = nz_rows / nz_rows.sum(axis=1)

    balanced_authors_dist = nz_author_dists.sum(axis=0) / nz_author_dists.sum()
    return jsd(topic_term_dist, balanced_authors_dist.getA1())


# Measure and record topic-author correlation measures (author entropy,
# minus major author, and balanced authors) for each topic.
def evaluate_correlation(topic_term_counts, topic_author_term_counts, out_tsv):
    n_topics, n_vocab = topic_term_counts.shape
    topic_counts = topic_term_counts.sum(axis=1, keepdims=True)
    topic_term_dists = topic_term_counts / topic_counts

    n_topics, n_vocab = topic_term_dists.shape
    writer = open(out_tsv, mode='w', encoding='utf-8')
    headings = ['Topic', 'Author Entropy', 'Minus Major Author',
                'Balanced Authors']
    writer.write('{}\n'.format('\t'.join(headings)))
    for topic in range(n_topics):
        topic_term_dist = topic_term_dists[topic].ravel()
        author_term_counts = topic_author_term_counts[topic]

        ae = author_entropy(author_term_counts)
        mma = minus_major_author(topic_term_dist, author_term_counts)
        ba = balanced_authors(topic_term_dist, author_term_counts)
        writer.write('{}\t{}\t{}\t{}\n'.format(topic, ae, mma, ba))
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate topic-author correlation for a MALLET ' +
                    'trained topic model.',
        add_help=False)
    required = parser.add_argument_group('required arguments')
    parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit.')
    required.add_argument('--input', dest='in_tsv',
                          metavar='FILE', required=True,
                          help='The file of the collection which the topic ' +
                               'model was trained on.')
    required.add_argument('--input-state', dest='state_fn',
                          metavar='FILE', required=True,
                          help='The MALLET state file containing the ' +
                               'gzipped Gibbs sampling state.')
    required.add_argument('--output', dest='out_tsv',
                          metavar='FILE', required=True,
                          help='Write the topic-author correlation ' +
                               'results to this file.')
    required.add_argument('--vocab', dest='vocab_fn',
                          metavar='FILE', required=True,
                          help='File of the working vocabulary, ' +
                               'one word per line.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        dest='verbose')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit()
    args = parser.parse_args()

    vocab, vocab_index = get_vocab(args.vocab_fn)
    print('Building document-author index')
    authors, author_index, doc_author_ids = get_author_info(args.in_tsv)
    print('Building topic-author-term counts')
    (topic_term_counts,
     topic_author_term_counts) = process_state_file(args.state_fn,
                                                    vocab_index,
                                                    doc_author_ids,
                                                    verbose=args.verbose)
    print('Evaulating topic-author correlation')
    evaluate_correlation(topic_term_counts, topic_author_term_counts,
                         args.out_tsv)
