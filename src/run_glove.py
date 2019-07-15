# train GloVe model for word embeddings
import os
import numpy as np
import csv
import glove
import utils as ut
import umap
import matplotlib
import pandas as pd
from math import pi
from bokeh.models import LinearColorMapper, BasicTicker, PrintfTickFormatter, ColorBar, HoverTool
from bokeh.plotting import figure, output_file, save, show
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score 


# Functions


def single_plot(data, labels, colors, leg_labels=None):
    plt.figure(figsize=(20, 10))
    for cl in set(labels):
        x = [d[0] for j, d in enumerate(data) if labels[j] == cl]
        y = [d[1] for j, d in enumerate(data) if labels[j] == cl]
        cols = [c for j, c in enumerate(colors) if labels[j] == cl]
        plt.xticks([])
        plt.yticks([])
        plt.scatter(x, y, c=cols, label=cl, s=20)
    if leg_labels is not None:
        plt.legend(loc=3, labels=leg_labels, markerscale=4, fontsize=20)
    else:
        plt.legend(loc=3, markerscale=4, fontsize=20)
    plt.savefig('scatterplot-glove.png')


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


# Private functions


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


level = 4
data_folder = 'odf-data-2019-06-13-12-28-37'
DATA_PATH = os.path.expanduser('~/Documents/nd_subtyping/data/%s/level-%d' % (data_folder, level))

with open(os.path.join(os.path.expanduser('~/Documents/nd_subtyping/data/') + data_folder, 
          'person-demographics.csv')) as f:
    rd = csv.reader(f)
    next(rd)
    p_sex = {r[0]: r[3] for r in rd}

# read vocabulary
list_feat = []
with open(os.path.join(DATA_PATH, 'cohort-vocab.csv'), 'r') as f:
    rd = csv.reader(f)
    next(rd)
    idx_to_bt = {r[1]: r[0] for r in rd}
    for r in rd:
        list_feat.append('::'.join(r[0].split('::')[0:3]))
list_feat = sorted(list(set(list_feat)))

# read behr and shuffle subsequences (F1 - F5)
with open(os.path.join(DATA_PATH, 'cohort-behr.csv'), 'r') as f:
    rd = csv.reader(f)
    next(rd)
    behr = {}
    behr_age = {}
    for r in rd:
        if r[0] not in behr:
            behr[r[0]] = {_age_p(float(r[1])): r[2::]}
        else:
            behr[r[0]].setdefault(_age_p(float(r[1])), list()).extend(r[2::]) 
        behr_age.setdefault(r[0], list()).extend([(r[1], idx_to_bt[idx]) for idx in r[2::]])

feat_dict_age = {p: {} for p in behr_age.keys()}
for p, seq in behr_age.items():
    for s in seq:
        ss = s[1].split('::')
        feat_dict_age[p].setdefault('::'.join(ss[0:len(ss) - 1]), list()).append((s[0], int(ss[-1])))

corpus = {}
for id_subj, p in behr.items():
    for k in sorted(p.keys()):
        np.random.shuffle(behr[id_subj][k])
        corpus.setdefault(id_subj, 
                          list()).extend(behr[id_subj][k])

out = build_cooccur(idx_to_bt, corpus, window_size=20)

model = glove.Glove(out, alpha=0.75, x_max=100.0, d=ut.n_dim)
for epoch in range(ut.n_epoch):
    err = model.train(batch_size=ut.batch_size)
    print("epoch %d, error %.3f" % (epoch, err), flush=True)

Wemb = model.W + model.ContextW
p_emb = []
id_list = []
for id_subj, term in corpus.items():
    if len(term)!=0:
        id_list.append(id_subj)
        p_emb.append(np.mean([Wemb[int(t)].tolist() for t in term], 
                             axis=0).tolist())

# Hierarchical clustering
n_cl_selected = []
for it in range(1):
    idx = np.random.randint(0, len(p_emb), int(len(p_emb)*0.8))
    tmp_dt = [p_emb[i] for i in idx]
    best_silh = 0
    for n in range(ut.min_cl, ut.max_cl):
        hclu = AgglomerativeClustering(n_clusters=n, linkage='ward')
        lab_cl = hclu.fit_predict(tmp_dt)
        tmp_silh = silhouette_score(tmp_dt, lab_cl)
        if tmp_silh > best_silh:
            best_silh = tmp_silh
            best_lab_cl = lab_cl
            best_n_clu = n
    print("(*) Iter {0} -- N clusters {1}".format(it, best_n_clu))
    n_cl_selected.append(best_n_clu)
unique, counts = np.unique(n_cl_selected, return_counts=True)
print("Counts of N clusters:")
print("N clusters -- Count")
for a, b in dict(zip(unique, counts)).items():
    print(a, b)
print("\nBest N cluster:{0}".format(unique[np.argmax(counts)]))

best_n_clu = unique[np.argmax(counts)]
hclu = AgglomerativeClustering(n_clusters=best_n_clu, linkage='ward')
lab_cl = hclu.fit_predict(p_emb)
silh = silhouette_score(p_emb, lab_cl)
print('(*) Number of clusters %d -- Silhouette score %.2f' % (best_n_clu, silh))

num_count = np.unique(lab_cl, return_counts=True)[1]
for idx, nc in enumerate(num_count):
    print("Cluster {0} -- Numerosity {1}".format(idx, nc))
print('\n')
print('\n')

colormap = [c for c in ut.col_dict if c not in ut.c_out]
colormap_rid = [colormap[cl] for cl in sorted(list(set(lab_cl)))]
colors_en = [colormap_rid[v] for v in lab_cl]
umap_mtx = umap.UMAP(random_state=42, n_neighbors=25, min_dist=0.0).fit_transform(p_emb)
single_plot(umap_mtx, lab_cl, colors_en)

linked = linkage(p_emb, 'ward')
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

plt.figure(figsize=(20, 30))
    # Dendrogram
dendrogram(Z=linked, labels=lab_cl, color_threshold=None,
           leaf_font_size=5, leaf_rotation=0, link_color_func=lambda x: link_cols[x])
plt.savefig('dendrogram-glove.png')

cl_p = {}
for cl, pt in zip(lab_cl, id_list):
    cl_p.setdefault(cl, list()).append(pt)

df_lab_age = []
for cl, p_vec in cl_p.items():
    for f in list_feat:
        for p in p_vec:
            try:
                for el in feat_dict_age[p][f]:
                    try:
                        df_lab_age.append(['-'.join([cl, p]), f, p_sex[p]] + [float(el[0]), el[1],
                                                                              df_sc[f].loc['::'.join([p, el[0]]),
                                                                                           'val'].item()])
                    except ValueError:
                        df_lab_age.append(['-'.join([cl, p]), f, p_sex[p]] + [float(el[0]), el[1],
                                                                              np.mean(df_sc[f].loc['::'.join([p, el[0]]),
                                                                                                   'val'].tolist())])
            except KeyError:
                pass
df_lab_age = sorted(df_lab_age, key=lambda x: (x[0], x[1], x[3]))
df_allcl = pd.DataFrame(df_lab_age, columns=['cllab', 'feat', 'sex', 'age', 'score', 'scaled_score'])
print(df_lab_age)

output_file(filename='HMP-glove-level-{0}-allSubj.html'.format(level),
            mode='inline')
colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41",
          "#550b1d"]
mapper = LinearColorMapper(palette=colors, low=df_allcl.scaled_score.min(), high=df_allcl.scaled_score.max())

lab = sorted(list(set(df_allcl['cllab'])))

p = figure(title="Scores level {0}".format(level, cl),
           x_range=lab, y_range=list_feat,
           x_axis_location="above", plot_width=2500, plot_height=800,
           toolbar_location='below')

TOOLTIPS = [('age', '@age'),
            ('sex', '@sex'),
            ('score', '@score'),
            ('feat', '@feat'),
            ('cllab', '@cllab')]
p.add_tools(HoverTool(tooltips=TOOLTIPS))

p.grid.grid_line_color = None
p.axis.axis_line_color = None
p.axis.major_tick_line_color = None
p.xaxis.major_label_text_font_size = "7pt"
p.yaxis.major_label_text_font_size = "7pt"
p.axis.major_label_standoff = 0
p.xaxis.major_label_orientation = pi / 2

p.rect(x="cllab", y="feat", width=1, height=1,
       source=df_allcl,
       fill_color={'field': 'scaled_score', 'transform': mapper},
       line_color=None)

color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="8pt",
                     ticker=BasicTicker(desired_num_ticks=len(colors)),
                     formatter=PrintfTickFormatter(format="%d.2"),
                     label_standoff=6, border_line_color=None, location=(0, 0))
p.add_layout(color_bar, 'right')
show(p)
