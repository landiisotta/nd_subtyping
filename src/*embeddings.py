import utils as ut
import glove
import argparse
import sys
from time import time


"""
Embedding functions
"""


def run_emb(datadir, level=None):
    outdir = datadir + '/level-' + level
    
    # load vocabulary and behrs (ID_SUBJ:[terms]; ID_SUBJ:Fn:[terms])
    bt_to_idx, idx_to_bt = _load_vocab(outdir, ut.file_names['vocab'])
    behr, behr_tf = _load_data(outdir, ut.file_names['behr'])

    terms = []
    for vec in behrs.values():
        terms.extend(vec)

    count = 0
    list_count = {}
    for idx, lab in idx_to_bt.items():
        co = terms.count(str(idx))
        list_count[lab] = co
        if co > 1:
            count += 1
    print("Number of repeated terms: {0} -- Terms with one occurrence: {1}\n".format(count, 
          len(bt_to_idx)-count))

    print('Most frequent terms (TF>20)')
    x = []
    y = []
    for lab, co in list_count.items():
        if co > 20:
            x.append(lab)
            y.append(co)
            print('%s, %d' % (lab, co))
        else:
            x.append('TF<20')
            y.append(co)
    
    # save plot term distribution
    plt.figure(figsize=(30, 20))
    plt.bar(x, y)
    plt.tick_params(axis='x', rotation=90, labelsize=10)
    plt.savefig(os.path.join(outdir, 'term20-distribution.png'))

    plt.figure(figsize=(20, 10))
    plt.bar(range(len(list_count.values())), list(list_count.values()))
    plt.tick_params(axis='x', rotation=90, labelsize=10)
    plt.savefig(os.path.join(outdir, 'term-distribution.png'))

    print('\n')

    # TF-IDF
    print('Computing TF-IDF matrix...')
    doc_list = list(map(lambda x: ' '.join(x), list(behrs.values())))
    id_subj = [id_lab for id_lab in behrs]

    vectorizer = TfidfVectorizer(norm='l2')
    tfidf_mtx = vectorizer.fit_transform(doc_list)

    print('Performing SVD on the TF-IDF matrix...')
    reducer = TruncatedSVD(n_components=ut.n_dim, random_state=123)
    svd_mtx = reducer.fit_transform(tfidf_mtx)

    # save SVD mtx
    with open(os.path.join(outdir, 'svd-mtx.csv'), 'w') as f:
        wr = csv.writerow(f)
        for idx, lab in enumerate(id_subj):
            wr.writerow([lab] + svd_mtx[idx])
    print('\n\n')

    # GloVe embeddings
    print('Starting computing GloVe embeddings for {0} epochs'.format(ut.n_epoch))
    corpus = _build_corpus(behrs_tf)
    coocc_dict = build_cooccur(idx_to_bt, corpus, window_size=20)

    model = glove.Glove(out, alpha=0.75, x_max=100.0, d=ut.n_dim)
    for epoch in range(ut.n_epoch):
        err = model.train(batch_size=ut.batch_size)
        print("epoch %d, error %.3f" % (epoch, err), flush=True)

    Wemb = model.W + model.ContextW # as suggested in Pennington et al.
    p_emb = []
    id_list = []
    for id_subj, term in corpus.items():
        if len(term)!=0:
            id_list.append(id_subj)
            p_emb.append(np.mean([Wemb[int(t)].tolist() for t in term], 
                                 axis=0).tolist())
    # save subject embeddings
    with open(os.path.join(outdir, 'glove-mtx.csv'), 'w') as f:
        wr = csv.writer(f)
        for id_p, pe in zip(id_list, p_emb):
            wr.writerow([id_p] + list(pe))


'''
Functions
'''


def build_cooccur(vocab, corpus, window_size=10, min_count=None):
    """
    Build a word co-occurrence dictionary for the given corpus.
    This function is a dictionary generator, where each element
    is of the form
        {i_main: {i_context: cooccurrence}}
    where `i_main` is the ID of the main word in the cooccurrence and
    `i_context` is the ID of the context word, and `cooccurrence` is the
    `X_{ij}` cooccurrence value as described in Pennington et al.
    (2014).
    If `min_count` is not `None`, cooccurrence pairs where either word
    occurs in the corpus fewer than `min_count` times are ignored.
    """

    vocab_size = len(vocab)

    # Collect cooccurrences internally as a sparse matrix for passable
    # indexing speed; we'll convert into a list later
    cooccurrences = {k: {} for k in vocab}

    for id_subj, line in corpus.items():

        for center_i, center_id in enumerate(line):
            # Collect all word IDs in left window of center word
            context_ids = line[max(0, center_i - window_size) : center_i]
            contexts_len = len(context_ids)

            for left_i, left_id in enumerate(context_ids):
                # Distance from center word
                distance = contexts_len - left_i

                # Weight by inverse of distance between words
                increment = 1.0 / float(distance)

                # Build co-occurrence matrix symmetrically (pretend we
                # are calculating right contexts as well)
                if left_id in cooccurrences[center_id]:
                    cooccurrences[center_id][left_id] += increment
                    cooccurrences[left_id][center_id] += increment
                else:
                    cooccurrences[center_id][left_id] = increment
                    cooccurrences[left_id][center_id] = increment

    return cooccurrences


"""
Private Functions
"""


def _load_vocab(indir, filename):
    with open(os.path.join(indir, filename)) as f:
        rd = csv.reader(f)
        next(rd)
        bt_to_idx = {}
        idx_to_bt = {}
        for r in rd:
            bt_to_idx[str(r[0])] = int(r[1])
            idx_to_bt[int(r[1])] = str(r[0])
    return bt_to_idx, idx_to_bt


def _age_p(age):
    if 0 < age <= 2.5:
        return 'F1'
    elif 2.5 < age <= 6.0:
        return 'F2'
    elif 6.0 < age <= 13.0:
        return 'F3'
    elif 13.0 < age < 17.0:
        return 'F4'
    else:
        return 'F5'


def _load_data(indir, filename):
    with open(os.path.join(indir, filename)) as f:
        rd = csv.reader(f)
        next(rd)
        behr = {}
        behr_tf = {}
        for r in rd:
             if r[0] not in behr_tf:
                behr_tf[r[0]] = {_age_p(float(r[1])): r[2::]}
            else:
                behr_tf[r[0]].setdefault(_age_p(float(r[1])), 
                                         list()).extend(r[2::]) 
            behr.setdefault(r[0], 
                            list()).extend(list(map(lambda x: int(x), 
                                                    r[2::])))
    return behr, behr_tf


# random shuffle terms in time slots
def _build_corpus(behr):
    corpus = {}
    for id_subj, p in behr.items():
        for k in sorted(p.keys()):
            np.random.shuffle(behr[id_subj][k])
            corpus.setdefault(id_subj, 
                              list()).extend(behr[id_subj][k])
    return corpus
    

'''
Main function
'''


def _process_args():
    parser = argparse.ArgumentParser(
    description = 'Compute behavioral term embeddings')
    parser.add_argument(dest='datadir',
                        help='data directory') 
    parser.add_argument(dest='level',
                        help='depth level of terms',
                        default=None)
    return parser.parse_args(sys.argv[1:])


if __name__ == "__main__":
    args = _process_args()
    print('')

    start = time()
    run_emb(args.datadir, args.level)
    print('Processing time: %d' % round(time() - start, 2))

 
