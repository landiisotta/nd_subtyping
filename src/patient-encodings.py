import os
import csv
import numpy as np
import utils as ut
import argparse
import sys
from time import time
from datetime import datetime as dt
from datetime import date
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, matthews_corrcoef
from statsmodels.stats.proportion import proportions_ztest
# from sklearn.manifold import TSNE
import umap
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd
from bokeh.plotting import figure, output_file, save, ColumnDataSource
from bokeh.models import CategoricalColorMapper, HoverTool


def clustering_tfidf(indir, level=None):

    datadir = indir + '/level-' + level

    lab_to_idx, idx_to_lab = _load_vocab(datadir, ut.file_names['vocab'])
    _, behrs = _load_data(datadir, ut.file_names['behr'])

    terms = []
    for vec in behrs.values():
        terms.extend(vec)

    count = 0
    list_count = {}
    for idx, lab in idx_to_lab.items():
        co = terms.count(str(idx))
        list_count[lab] = co
        if co > 1:
            count += 1
    print("Number of repeated terms: {0} -- Terms with one occurrence: {1}\n".format(count, len(lab_to_idx)-count))

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

    plt.figure(figsize=(30, 20))
    plt.bar(x, y)
    plt.tick_params(axis='x', rotation=90, labelsize=10)
    plt.savefig(os.path.join(datadir, 'term20-distribution.png'))

    plt.figure(figsize=(20, 10))
    plt.bar(range(len(list_count.values())), list(list_count.values()))
    plt.tick_params(axis='x', rotation=90, labelsize=10)
    plt.savefig(os.path.join(datadir, 'term-distribution.png'))

    print('\n')

    # TF-IDF
    print('Computing TF-IDF matrix...')
    doc_list = list(map(lambda x: ' '.join(x), list(behrs.values())))
    id_subj = [id_lab for id_lab in behrs]

    vectorizer = TfidfVectorizer(norm='l2')
    tfidf_mtx = vectorizer.fit_transform(doc_list)

    print('Performing SVD on the TF-IDF matrix...')
    reducer = TruncatedSVD(n_components=ut.n_dim, random_state=123)
    encoded_dt = reducer.fit_transform(tfidf_mtx)

    # Internal clustering validation
    rf = RandomForestClassifier(criterion='entropy', random_state=42)

    n_cl_selected = []
    for it in range(100):
        idx = np.random.randint(0, len(encoded_dt), int(len(encoded_dt)*0.7))
        tmp_encodeddt = [encoded_dt[i] for i in idx]
        # tmp_idsubj = [id_subj[i] for i in idx]
        best = 0
        for n_clu in range(ut.min_cl, ut.max_cl):
            hclu = AgglomerativeClustering(n_clusters=n_clu)
            lab_cl = hclu.fit_predict(tmp_encodeddt)
            tmp_silh = silhouette_score(tmp_encodeddt, lab_cl)
            # print('(*) Number of clusters %d -- Silhouette score %.2f' % (n_clu, tmp_silh))
            try:
                enc_tr, enc_ts, lab_tr, lab_ts = train_test_split(tmp_encodeddt, lab_cl,
                                                                  stratify=lab_cl,
                                                                  test_size=0.25,
                                                                  random_state=42)
                rf.fit(enc_tr, lab_tr)
                rf_predict = rf.predict(enc_ts)
                tmp_mcc = matthews_corrcoef(lab_ts, rf_predict)
                # print('    MCC RF classifier: %.2f' % tmp_mcc)
            except ValueError:
                pass
        # mu = np.mean([tmp_mcc, tmp_silh])
        # print('    mean(MCC, silhouette): %.2f' % mu)
            if tmp_silh > best:
                best_mcc = tmp_mcc
                best_silh = tmp_silh
                best_lab_cl = lab_cl
                best_n_clu = n_clu
                best = tmp_silh
        print("(*) Iter {0} -- N clusters {1}".format(it, best_n_clu))
        n_cl_selected.append(best_n_clu)
    unique, counts = np.unique(n_cl_selected, return_counts=True)
    print("Counts of N clusters:")
    print("N clusters -- Count")
    for a, b in dict(zip(unique, counts)).items():
        print(a, b)
    print("\nBest N cluster:{0}".format(unique[np.argmax(counts)]))

    best_n_clu = unique[np.argmax(counts)]
    hclu = AgglomerativeClustering(n_clusters=best_n_clu)
    lab_cl = hclu.fit_predict(encoded_dt)
    tmp_silh = silhouette_score(encoded_dt, lab_cl)
    print('(*) Number of clusters %d -- Silhouette score %.2f' % (best_n_clu, tmp_silh))
    enc_tr, enc_ts, lab_tr, lab_ts = train_test_split(encoded_dt, lab_cl,
                                                      stratify=lab_cl,
                                                      test_size=0.25,
                                                      random_state=42)
    rf.fit(enc_tr, lab_tr)
    rf_predict = rf.predict(enc_ts)
    tmp_mcc = matthews_corrcoef(lab_ts, rf_predict)
    print('    MCC RF classifier: %.2f' % tmp_mcc)
    # mu = np.mean([tmp_mcc, tmp_silh])
    # print('    mean(MCC, silhouette): %.2f' % mcc)

    num_count = np.unique(lab_cl, return_counts=True)[1]
    for idx, nc in enumerate(num_count):
        print("Cluster {0} -- Numerosity {1}".format(idx, nc))
    print('\n')
    print('\n')
    # print("MCC: %.4f -- silhouette score: %.4f -- Number of clusters: %d\n" % (best_mcc, best_silh, best_n_clu))

    colormap = [c for c in ut.col_dict if c not in ut.c_out]
    colormap_rid = [colormap[cl] for cl in sorted(list(set(lab_cl)))]
    colors_en = [colormap_rid[v] for v in lab_cl]
    umap_mtx = umap.UMAP(random_state=42).fit_transform(encoded_dt)
    single_plot(datadir, umap_mtx, lab_cl, colors_en)

    linked = linkage(encoded_dt, 'ward')
    # Color mapping
    dflt_col = "#808080"  # Unclustered gray
    # * rows in Z correspond to "inverted U" links that connect clusters
    # * rows are ordered by increasing distance
    # * if the colors of the connected clusters match, use that color for link
    link_cols = {}
    for i, i12 in enumerate(linked[:, :2].astype(int)):
        c1, c2 = (link_cols[x] if x > len(linked) else colormap_rid[lab_cl[x]]
                  for x in i12)
        link_cols[i + 1 + len(linked)] = c1 if c1 == c2 else dflt_col

    plt.figure(figsize=(20, 10))
    # Dendrogram
    dendrogram(Z=linked, labels=lab_cl, color_threshold=None,
                   leaf_font_size=5, leaf_rotation=0, link_color_func=lambda x: link_cols[x])
    plt.savefig(os.path.join(datadir, 'dendrogram-tfidf.png'))

    with open(os.path.join(indir, 'person-demographics.csv')) as f:
        rd = csv.reader(f)
        next(rd)
        dem = {r[0]: r[1::] for r in rd}

    df_ar = []
    for id_name, coord, cl_lab in zip(id_subj, umap_mtx, lab_cl):
        df_ar.append([id_name, coord[0], coord[1], cl_lab, age(dem[id_name][0]),
                          dem[id_name][2], dem[id_name][3]])
    df_ar = np.array(df_ar)
    df = pd.DataFrame(df_ar, columns=['id_subj', 'x', 'y', 'cluster', 'age', 'sex', 'n_enc'])
    df['x'] = df['x'].astype('float64')
    df['y'] = df['y'].astype('float64')
    df['age'] = df['age'].astype('float64')
    df['n_enc'] = df['n_enc'].astype('int')

    p_clu = {}
    with open(os.path.join(datadir, 'person-cluster.txt'), 'w') as f:
        wr = csv.writer(f)
        wr.writerow(['ID_LAB', 'CLUSTER'])
        for el in df_ar:
            wr.writerow([el[0], el[3]])
            p_clu[el[0]] = el[3]


    source = ColumnDataSource(dict(
        x=df['x'].tolist(),
        y=df['y'].tolist(),
        id_subj=df['id_subj'].tolist(),
        cluster=[str(i) for i in df['cluster'].tolist()],
        age=df['age'].tolist(),
        sex=df['sex'].tolist(),
        n_enc=df['n_enc'].tolist()))

    labels = [str(i) for i in df['cluster'].tolist()]
    cmap = CategoricalColorMapper(factors=sorted(pd.unique(labels)), palette=colormap_rid)

    TOOLTIPS = [('id_subj', '@id_subj'),
                ('cluster', '@cluster'),
                ('sex', '@sex'),
                ('age', '@age'),
                ('n_enc', '@n_enc')]

    plotTools = 'box_zoom, wheel_zoom, pan,  crosshair, reset, save'

    output_file(filename=os.path.join(datadir, 'tfidf-plot-interactive.html'), mode='inline')
    p = figure(plot_width=800, plot_height=800, tools=plotTools)
    p.add_tools(HoverTool(tooltips=TOOLTIPS))
    p.circle('x', 'y', legend='cluster', source=source, color={"field": 'cluster', "transform": cmap})
    save(p)

    freq_term(lab_cl, idx_to_lab, behrs, p_clu)


"""
Functions
"""


def age(birthDate):
    days_in_year = 365.2425
    bDate = dt.strptime(birthDate, '%d/%m/%Y').date()
    currentAge = (date.today() - bDate).days / days_in_year
    return currentAge


def single_plot(outdir, data, labels, colors, leg_labels=None):
    plt.figure(figsize=(20,10))
    for cl in set(labels):
        x = [d[0] for j, d in enumerate(data) if labels[j] == cl]
        y = [d[1] for j, d in enumerate(data) if labels[j] == cl]
        cols = [c for j, c in enumerate(colors) if labels[j] == cl]
        plt.xticks([])
        plt.yticks([])
        plt.scatter(x, y, c=cols, label=cl, s=20)
    if leg_labels is not None:
        plt.legend(loc=2, labels=leg_labels, markerscale=4, fontsize=20)
    else:
        plt.legend(loc=2, markerscale=4, fontsize=20)
    plt.savefig(os.path.join(outdir, 'tfidf-plot.png'))


def FreqDict(tokens):
    freq_dict = {}
    tok = []
    for seq in tokens:
        tok.extend(seq)
    tok = set(tok)
    for t in tok:
        for seq in tokens:
            if t in seq:
                if t not in freq_dict:
                    freq_dict[t] = 1
                else:
                    freq_dict[t] += 1
    return freq_dict


def freq_term(pred_class, vocab, raw_ehr, p_clu):
    result_ztest = compare_proportion(p_clu, vocab, raw_ehr)
    data = list(raw_ehr.values())
    list_terms = []
    print("List of most frequent terms (t) in each cluster")
    print("The percentage of t in the cluster and the number of subject with t in their sequence is reported")
    print("Only terms with significant proportion compared to other clusters " 
          "are reported (see Proportion ztest tables)\n")
    for subc in range(len(set(pred_class))):
        tmp_data = {}
        for j in range(len(pred_class)):
            if pred_class[j] == subc:
                tmp_data.setdefault(subc, list()).append([rd for rd in data[j]
                                                           if rd!=0])
        print("Cluster {0} numerosity: {1}".format(subc, len(tmp_data[subc])))
        term_count = FreqDict(tmp_data[subc])
        clust_mostfreq = []
        for l in range(len(vocab)):
            try:
                MFMT = max(term_count, key=(lambda key: term_count[key]))
                num_MFMT = 0
                subc_termc = 0
                for ehr in tmp_data[subc]:
                    for e in ehr:
                        if e == MFMT:
                            subc_termc += 1
                for seq in raw_ehr.values():
                    for t in seq:
                        if t == MFMT:
                            num_MFMT += 1
                mtx_pval = result_ztest[MFMT]['pval']
                if subc != 0 and subc != len(set(pred_class))-1:
                    for pvr, pvc in zip(mtx_pval[subc], mtx_pval.transpose()[subc-1]):
                        if (pvr < 0.05 and pvr != 0) or (pvc < 0.05 and pvc != 0):
                            print("% Term:{0} "
                                  "= {1:.2f} " 
                                # "({2} out of {3} terms in the whole dataset "
                                "-- N patients in cluster {2} out of {3}".format(vocab[int(MFMT)],
                                                                          subc_termc/num_MFMT,
                                                                          #subc_termc,
                                                                          #num_MFMT,
                                                                          term_count[MFMT],
                                                                          len(tmp_data[subc])))
                            print(["ztest"] + ["cl{0}".format(idx+1) for idx in range(mtx_pval.shape[1])])
                            for idx in range(mtx_pval.shape[0]):
                                print(["cl{0}".format(idx)] + ["{0:.3f}".format(pv) for pv in mtx_pval[idx]])
                            break
                            #print(result_ztest[MFMT])
                elif subc == len(set(pred_class))-1:
                    for pv in mtx_pval.transpose()[subc-1]:
                        if pv < 0.05 and pv != 0:
                            print("% Term:{0} "
                                  "= {1:.2f} "
                                  # "({2} out of {3} terms in the whole dataset "
                                  "-- N patients in cluster {2} out of {3}".format(vocab[int(MFMT)],
                                                                                   subc_termc / num_MFMT,
                                                                                   # subc_termc,
                                                                                   # num_MFMT,
                                                                                   term_count[MFMT],
                                                                                   len(tmp_data[subc])))
                            print(["ztest"] + ["cl{0}".format(idx + 1) for idx in
                                               range(mtx_pval.shape[1])])
                            for idx in range(mtx_pval.shape[0]):
                                print(["cl{0}".format(idx)] + ["{0:.3f}".format(pv) for pv in
                                                               mtx_pval[idx]])
                            break
                else:
                    for pv in mtx_pval[subc]:
                        if pv < 0.05 and pv != 0:
                            print("% Term:{0} "
                                  "= {1:.2f} "
                                  # "({2} out of {3} terms in the whole dataset "
                                  "-- N patients in cluster {2} out of {3}".format(vocab[int(MFMT)],
                                                                                   subc_termc / num_MFMT,
                                                                                   # subc_termc,
                                                                                   # num_MFMT,
                                                                                   term_count[MFMT],
                                                                                   len(tmp_data[subc])))
                            print(["ztest"] + ["cl{0}".format(idx + 1) for idx in
                                               range(mtx_pval.shape[1])])
                            for idx in range(mtx_pval.shape[0]):
                                print(["cl{0}".format(idx)] + ["{0:.3f}".format(pv) for pv in
                                                               mtx_pval[idx]])
                            break
                term_count.pop(MFMT)
                clust_mostfreq.append(MFMT)
            except ValueError:
                pass
        print("\n")
        list_terms.append(clust_mostfreq)
    return


def compare_proportion(p_clu, vocab, behr):
    term_cl = {}
    set_terms = set(vocab.keys())
    for p_id, cl in p_clu.items():
        term_cl.setdefault(cl, list()).extend(behr[p_id])
        set_terms.update(behr[p_id])

    term_prop = {t: [None] * len(term_cl) for t in set_terms}
    for cl, term in term_cl.items():
        for t in set_terms:
            term_prop[t][int(cl)] = (term.count(t), len(term))

    # ztest of proportion with bonferroni-corrected p-values
    s = (len(term_cl.keys()) - 1, len(term_cl.keys()) - 1)
    result_ztest = {t: {'pval': np.zeros(s), 'count': term_prop[t]} for t in term_prop}
    for t, prop_count in term_prop.items():
        count_comp = 0
        for cl in range(len(prop_count)):
            idx = cl + 1
            while idx < len(prop_count):
                if prop_count[cl][0] != 0 or prop_count[idx][0] != 0:
                    count = np.array([prop_count[cl][0], prop_count[idx][0]])
                    nobs = np.array([prop_count[cl][1], prop_count[idx][1]])
                    stat, pval = proportions_ztest(count, nobs)
                    result_ztest[t]['pval'][cl][idx - 1] = pval
                    count_comp += 1
                idx += 1
        result_ztest[t]['pval'] = result_ztest[t]['pval'] * count_comp
    return result_ztest


"""
Private Functions
"""


def _load_vocab(indir, filename):
    with open(os.path.join(indir, filename)) as f:
        rd = csv.reader(f)
        next(rd)
        vocab_lab = {}
        vocab_idx = {}
        for r in rd:
            vocab_lab[str(r[0])] = int(r[1])
            vocab_idx[int(r[1])] = str(r[0])
    return vocab_lab, vocab_idx


def _load_data(indir, filename):
    with open(os.path.join(indir, filename)) as f:
        rd = csv.reader(f)
        next(rd)
        dt = {}
        dt_noage = {}
        for r in rd:
            dt.setdefault(r[0], list()).append((float(r[1]), list(map(lambda x: int(x), r[2::]))))
            dt_noage.setdefault(r[0], list()).extend(list(map(lambda x: str(x), r[2::])))
    return dt, dt_noage


"""
Main Function
"""


def _process_args():
    parser = argparse.ArgumentParser(
        description='Compute tfidf matrix and run hierarchical clustering with the desired depth level. '
                    'Perform internal clustering validation')
    parser.add_argument(dest='indir', help='data path, no level')
    parser.add_argument(dest='level', help='depth level to create data path')

    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    args = _process_args()
    print('')

    start = time()
    clustering_tfidf(args.indir, args.level)












