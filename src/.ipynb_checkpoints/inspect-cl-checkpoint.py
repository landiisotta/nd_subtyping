
import csv
import os
import numpy as np
from math import pi
import pandas as pd

from scipy.stats import ttest_ind
from bokeh.models import LinearColorMapper, BasicTicker, PrintfTickFormatter, ColorBar, HoverTool
from bokeh.plotting import figure, output_file, save

level = ['level-1', 'level-2', 'level-3']
datapath = os.path.expanduser("~/Documents/nd_subtyping/data/odf-data-2019-06-03-15-37-39/")
with open(os.path.join(datapath, level[0], 'person-cluster.txt')) as f:
    rd = csv.reader(f)
    next(rd)
    cl_p = {}
    p_cl = {}
    for r in rd:
        p_cl[r[0]] = r[1]
        cl_p.setdefault(r[1], []).append(r[0])


# Inspect subclusters on different levels


def inspect_subcl(datapath, level):
    
    with open(os.path.join(datapath, 'level-{0}'.format(level), 'cohort-vocab.csv')) as f:
        rd = csv.reader(f)
        next(rd)
        idx_to_bt = {r[1]: r[0] for r in rd}

    behr = {}
    behr_age
    with open(os.path.join(datapath, 'level-{0}'.format(level), 'cohort-behr.csv')) as f:
        rd = csv.reader(f)
        next(rd)
        for r in rd:
            behr.setdefault(r[0], list()).extend([idx_to_bt[idx] for idx in r[2::]])
            behr_age.setdefault(r[0], list()).extend([(r[1], idx_to_bt[idx]) for idx in r[2::]])
            
    feat_dict = {p: {} for p in behr.keys()}
    for p, seq in behr.items():
        for s in seq:
            ss = s.split('::')
            feat_dict[p].setdefault('::'.join(ss[0:len(ss)-1]), list()).append(int(ss[-1]))
            
    cl_stat = {cl: {} for cl in set(p_cl.values())}
    for subj, cl in p_cl.items():
        for ky, ft in feat_dict[subj].items():
            if len(ft) > 1:
                mft = np.mean(ft)
                cl_stat[cl].setdefault(ky, list()).extend([mft])
            else:
                cl_stat[cl].setdefault(ky, list()).extend(ft)
    
    list_feat = set()

    for cl in cl_stat:
        tmp = set(list(cl_stat[cl].keys()))
        list_feat = list_feat.union(tmp)
    list_feat = sorted(list(list_feat))

    df_vect = []
    for cl, feat in cl_stat.items():
        for f in list_feat:
            try:
                df_vect.append([cl, f, np.mean(feat[f])])
            except KeyError:
                df_vect.append([cl, f, None])
    df_vect = sorted(df_vect, key=lambda x: x[0])

    df_lab = {}
    for cl, p_vec in cl_p.items():
        for f in list_feat:
            for p in p_vec:
                try:
                    df_lab.setdefault(cl, list()).append([p, f, np.mean(feat_dict[p][f])])
                except KeyError:
                    df_lab.setdefault(cl, list()).append([p, f, None])
        df_lab[cl] = sorted(df_lab[cl], key=lambda x: x[0])

    df_ttest = pairwise_ttest(list_feat, cl_stat)
    df_ttest.to_csv(os.path.join(datapath, 'level-{0}'.format(level), 
                           'pairwise-ttest-level{0}.csv'.format(level)), 
              index=False)
    
    df = pd.DataFrame(df_vect, columns=['cluster', 'feat', 'm_score'])
    clusters = sorted(list(set(df['cluster'])))

    output_file(filename=os.path.join(datapath, 'level-{0}'.format(level),
                                      'HMP-level-{0}.html'.format(level)), 
                mode='inline')
    colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", 
              "#550b1d"]
    mapper = LinearColorMapper(palette=colors, low=df.m_score.min(), high=df.m_score.max())

    TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"

    p = figure(title="Mean scores level {0} for {1} clusters".format(level, len(clusters)),
               x_range=clusters, y_range=list_feat,
               x_axis_location="above", plot_width=600, plot_height=700,
               toolbar_location='below')

    TOOLTIPS = [('m_score', '@m_score')]
    p.add_tools(HoverTool(tooltips=TOOLTIPS))

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "8pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = pi / 3

    p.rect(x="cluster", y="feat", width=1, height=1,
           source=df,
           fill_color={'field': 'm_score', 'transform': mapper},
           line_color=None)

    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="8pt",
                         ticker=BasicTicker(desired_num_ticks=len(colors)),
                         formatter=PrintfTickFormatter(format="%d.2"),
                         label_standoff=6, border_line_color=None, location=(0, 0))
    p.add_layout(color_bar, 'right')
# show(p)
    save(p)
    
    for c, st in cl_stat.items():
        print("Cluster {0} -- Level-{1}\n".format(c, 2))
        for s, val in st.items():
            print("Term {0} -- Mean score: {1:.2f} ({2:.2f}, {3:.2f} -- SD: {4:.2f})".format(s, np.mean(val),
                                                                          np.min(val), np.max(val), np.std(val)))
        print("\n")
    print("\n\n")


def pairwise_ttest(list_feat, score_dict):
    pval_dict = {ky:{} for ky in list_feat}
    for f in sorted(list_feat):
        for cl in sorted(list(score_dict.keys())):
            cl_comp = int(cl) + 1
            while cl_comp < len(sorted(list(score_dict.keys()))):
                try:
                    stat, pval = ttest_ind(score_dict[cl][f], score_dict[str(cl_comp)][f])
                    pval_dict[f]['-'.join([str(cl), str(cl_comp)])] = pval
                except KeyError:
                    pval_dict[f]['-'.join([str(cl), str(cl_comp)])] = None
                cl_comp += 1
    df_vect = []
    for f, comp in pval_dict.items():
        n_comp = len([p for p in comp.values() if p is not None])
        for c in comp:
            if comp[c] is not None:
                comp[c] = comp[c] * n_comp
        df_vect.append([f] + [comp[c] for c in sorted(list(comp.keys()))])
        col_names = [c for c in sorted(list(comp.keys()))]
    df = pd.DataFrame(df_vect, columns=['feat'] + col_names)
    return df


# inspect_subcl(datapath, 2)


# inspect_subcl(datapath, 1)


inspect_subcl(datapath, 3)


    

