import argparse
import numpy as np
import sys

from collections import Counter
from scipy.sparse import csr_matrix, lil_matrix, vstack
from scipy.stats import gamma


# Read in vocabulary from file.
def get_vocab(vocab_fn):
    vocab = []
    vocab_index = {}
    for i, line in enumerate(open(vocab_fn, mode='r', encoding='utf-8')):
        term = line.strip()
        vocab.append(term)
        vocab_index[term] = i
    return vocab, vocab_index


# From input corpus in_tsv and the index of working vocabulary vocab_index
# construct:
#   authors: working list of authors
#   author_doc_ids: mapping of authors to document ids
#   doc_term_matrix: document-term matrix
def process_corpus(in_tsv, vocab_index, verbose=False):
    vocab_size = len(vocab_index)
    authors_by_doc = []
    doc_vectors = []
    reader = open(in_tsv, mode='r', encoding='utf-8')
    for i, line in enumerate(reader):
        if verbose and i and i % 1000 == 0:
            print(i, file=sys.stderr)
        fields = line.strip().split('\t')
        authors_by_doc.append(fields[1])
        vector = lil_matrix((1, vocab_size))
        term_counts = Counter(fields[2].split())
        for term in term_counts:
            if term in vocab_index:
                col = vocab_index[term]
                vector[0, col] = term_counts[term]
        doc_vectors.append(vector)
    doc_term_matrix = vstack(doc_vectors, format='csr')
    authors = sorted(list(set(authors_by_doc)))
    author_index = {author: i for i, author in enumerate(authors)}
    author_doc_ids = {author: [] for author in authors}
    for i, a in enumerate(authors_by_doc):
        author_doc_ids[a].append(i)
    return authors, author_index, author_doc_ids, doc_term_matrix


# Construct author-term matrix from document-term matrix.
def get_author_term_matrix(authors, author_doc_ids, doc_term_matrix):
    author_vectors = [csr_matrix(doc_term_matrix[doc_ids].sum(axis=0)) for
                      doc_ids in author_doc_ids.values()]
    author_term_matrix = vstack(author_vectors, format='csc')
    return author_term_matrix


# Estimate gamma parameters k, theta using method of moments
def get_gamma_parameters(author_term_freqs):
    term_means = np.mean(author_term_freqs, axis=0).getA1()
    term_vars = np.var(author_term_freqs, axis=0, ddof=1).getA1()
    ks = np.divide(np.square(term_means), term_vars)
    thetas = np.divide(term_vars, term_means)
    return ks, thetas


# Build author-term stop weight matrix for given author-term matrix and
# probability threshold.
def get_stop_weights(author_term_matrix, threshold):
    n_authors, n_terms = author_term_matrix.shape
    weights = lil_matrix((n_authors, n_terms))
    author_term_freqs = author_term_matrix / author_term_matrix.sum(axis=1)
    ks, thetas = get_gamma_parameters(author_term_freqs)

    # Assign probabilities of deletion for each partition-feature pair
    for term_id in range(n_terms):
        author_freqs = author_term_freqs[:, term_id].getA1()
        freq_threshold = gamma.ppf(1-threshold, ks[term_id],
                                   scale=thetas[term_id])
        for author_id, freq in enumerate(author_freqs):
            if freq_threshold < freq:
                weights[author_id, term_id] = 1 - freq_threshold / freq
    weights = weights.tocsr()
    return weights


# Downsample input according to the author-term stop weights matrix.
def downsample(in_tsv, vocab_index, document_term_matrix, author_index,
               author_term_weights, out_tsv, min_tokens=20, verbose=False):
    writer = open(out_tsv, mode='w', encoding='utf-8')
    for doc_id, line in enumerate(open(in_tsv, mode='r', encoding='utf-8')):
        if verbose and doc_id and doc_id % 1000 == 0:
            print(doc_id, file=sys.stderr)
        fields = line.strip().split('\t')
        author = fields[1]
        author_id = author_index[author]
        tokens = np.array(fields[2].split())
        term_ids = np.array([vocab_index[t] for t in tokens if t in vocab_index])

        term_stop_rates = author_term_weights.getrow(author_id)
        term_stop_rates = term_stop_rates.toarray().ravel()
        token_keep_rates = [1-term_stop_rates[i] for i in term_ids]
        token_mask = np.random.binomial(1, token_keep_rates)
        n_kept = token_mask.sum()

        # Write document if it has at least min_tokens tokens
        if n_kept >= min_tokens:
            token_mask = token_mask.astype(bool)
            stopped_text = ' '.join(tokens[token_mask])
            writer.write('{}\t{}\t{}\n'.format(fields[0], fields[1],
                                               stopped_text))
    writer.close()


if __name__ == '__main__':
    # Process command line arguments
    parser = argparse.ArgumentParser(description='Downsample collection.',
                                     add_help=False)
    # Remove initial argument group so that required arguments will appear
    # before optional ones
    required = parser.add_argument_group('required arguments')
    parser.add_argument('-h', '--helper', action='help',
                        help='Show this help message and exit.')
    required.add_argument('--input', dest='in_tsv',
                          metavar='FILE', required=True,
                          help='The file of the collection to be ' +
                               'downsampled, one instance per line.')
    required.add_argument('--output', dest='out_tsv',
                          metavar='FILE', required=True,
                          help='Write the downsampled collection to this ' +
                               'file.')
    required.add_argument('--vocab', dest='vocab_fn',
                          metavar='FILE', required=True,
                          help='File of the working vocabulary, ' +
                               'one word per line.')
    parser.add_argument('--threshold', dest='threshold',
                        metavar='NUMBER', default=0.05, type=float,
                        help='Probability threshold for downsampling ' +
                             'method. Default is 0.05.')
    parser.add_argument('--min-doc-length', dest='min_doc_len',
                        metavar='N', default=20, type=int,
                        help='Remove downsampled documents with lengths ' +
                             'less than this value. Default is 20.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        dest='verbose')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    vocab, vocab_index = get_vocab(args.vocab_fn)
    print('Building doc-term matrix')
    (authors, author_index, author_doc_ids,
     doc_term_counts) = process_corpus(args.in_tsv, vocab_index,
                                       verbose=args.verbose)
    print('Building author-term matrix')
    author_term_counts = get_author_term_matrix(authors, author_doc_ids,
                                                doc_term_counts)
    print('Building stop weights')
    stop_weights = get_stop_weights(author_term_counts, args.threshold)
    print('Downsampling file')
    downsample(args.in_tsv, vocab_index, doc_term_counts, author_index,
               stop_weights, args.out_tsv, min_tokens=args.min_doc_len,
               verbose=args.verbose)
