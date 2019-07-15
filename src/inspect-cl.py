import csv
import os
import numpy as np
from math import pi
import pandas as pd
from sklearn.preprocessing import StandardScaler

from scipy.stats import ttest_ind
from bokeh.models import LinearColorMapper, BasicTicker, PrintfTickFormatter, ColorBar, HoverTool, ColumnDataSource
from bokeh.plotting import figure, output_file, save

level = ['level-1', 'level-2', 'level-3', 'level-4']
datapath = os.path.expanduser("~/Documents/nd_subtyping/data/odf-data-2019-06-13-12-28-37/")
with open(os.path.join(datapath, level[3], 'person-cluster.txt')) as f:
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
    behr_age = {}
    with open(os.path.join(datapath, 'level-{0}'.format(level), 'cohort-behr.csv')) as f:
        rd = csv.reader(f)
        next(rd)
        for r in rd:
            behr.setdefault(r[0], list()).extend([idx_to_bt[idx] for idx in r[2::]])
            behr_age.setdefault(r[0], list()).extend([(r[1], idx_to_bt[idx]) for idx in r[2::]])

    with open(os.path.join(datapath, 'person-demographics.csv')) as f:
        rd = csv.reader(f)
        next(rd)
        p_sex = {r[0]: r[3] for r in rd}

    feat_dict = {p: {} for p in behr.keys()}
    for p, seq in behr.items():
        for s in seq:
            ss = s.split('::')
            feat_dict[p].setdefault('::'.join(ss[0:len(ss) - 1]), list()).append(int(ss[-1]))

    feat_dict_age = {p: {} for p in behr_age.keys()}
    for p, seq in behr_age.items():
        for s in seq:
            ss = s[1].split('::')
            feat_dict_age[p].setdefault('::'.join(ss[0:len(ss) - 1]), list()).append((s[0], int(ss[-1])))

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

    scaler = StandardScaler(copy=False)
    ds_sc = {f: {'val': [], 'idx': []} for f in list_feat}
    df_sc = {}
    for f in list_feat:
        for p in feat_dict_age:
            try:
                for el in feat_dict_age[p][f]:
                    ds_sc[f]['val'].append(el[1])
                    ds_sc[f]['idx'].append('::'.join([p, el[0]]))
            except KeyError:
                pass
        ds_sc[f]['val'] = [v[0] for v in scaler.fit_transform(np.array(ds_sc[f]['val']).reshape(-1, 1))]
        df_sc[f] = pd.DataFrame(ds_sc[f]['val'], index=ds_sc[f]['idx'], columns=['val'])

    df_vect = []
    for cl, feat in cl_stat.items():
        for f in list_feat:
            try:
                scl_f = [c for c in cl_stat[cl][f] if f in cl_stat[cl].keys()]
                for n in sorted(list(set(cl_stat.keys()))):
                    if n != cl and f in cl_stat[n].keys():
                        scl_f.extend([c for c in cl_stat[n][f]])
                scl_f = scaler.fit_transform(np.array(scl_f).reshape(-1, 1))
                m_feat_scaled = [v[0] for v in scl_f[:len(cl_stat[cl][f])]]
                df_vect.append([cl, f, np.mean(feat[f]), np.mean(m_feat_scaled)])
            except KeyError:
                pass
                # df_vect.append([cl, f, None])
    df_vect = sorted(df_vect, key=lambda x: x[0])
    # print(cl_stat)

    df_lab = {}
    for cl, p_vec in cl_p.items():
        for f in list_feat:
            for p in p_vec:
                try:
                    df_lab.setdefault(cl, list()).append([p, f, np.mean(feat_dict[p][f])])
                except KeyError:
                    pass
                    # df_lab.setdefault(cl, list()).append([p, f, None])
        df_lab[cl] = sorted(df_lab[cl], key=lambda x: x[0])

    df_lab_age = []
    df_clsep = []
    for cl, p_vec in cl_p.items():
        for f in list_feat:
            for p in p_vec:
                try:
                    for el in feat_dict_age[p][f]:
                        try:
                            df_lab_age.append(['-'.join([cl, p]), f, p_sex[p]] + [float(el[0]), el[1],
                                                                                  df_sc[f].loc['::'.join([p, el[0]]),
                                                                                               'val'].item()])
                            df_clsep.append([cl, p, f, p_sex[p], float(el[0]), el[1],
                                             df_sc[f].loc['::'.join([p, el[0]]),
                                                          'val'].item()])
                        except ValueError:
                            df_lab_age.append(['-'.join([cl, p]), f, p_sex[p]] + [float(el[0]), el[1],
                                                                                  np.mean(df_sc[f].loc[
                                                                                              '::'.join([p, el[0]]),
                                                                                              'val'].tolist())])
                            df_clsep.append(
                                [cl, p, f, p_sex[p], float(el[0]), el[1], np.mean(df_sc[f].loc['::'.join([p, el[0]]),
                                                                                               'val'].tolist())])
                except KeyError:
                    pass
    df_lab_age = sorted(df_lab_age, key=lambda x: (x[0], x[1], x[3]))
    df_clsep = sorted(df_clsep, key=lambda x: (x[0], x[1], x[2], x[4]))

    df_ttest = pairwise_ttest(list_feat, cl_stat)
    df_ttest.to_csv(os.path.join(datapath, 'level-{0}'.format(level),
                                 'pairwise-ttest-level{0}.csv'.format(level)),
                    index=False)

    df = pd.DataFrame(df_vect, columns=['cluster', 'feat', 'm_score', 'm_score_scaled'])
    clusters = sorted(list(set(df['cluster'])))

    output_file(filename=os.path.join(datapath, 'level-{0}'.format(level),
                                      'HMP-level-{0}.html'.format(level)),
                mode='inline')
    colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41",
              "#550b1d"]
    mapper = LinearColorMapper(palette=colors, low=df.m_score_scaled.min(), high=df.m_score_scaled.max())

    TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"

    p = figure(title="Mean scores level {0} for {1} clusters".format(level, len(clusters)),
               x_range=clusters, y_range=list_feat,
               x_axis_location="above", plot_width=600, plot_height=900,
               toolbar_location='below')

    TOOLTIPS = [('m_score', '@m_score'),
                ('feat', '@feat')]
    p.add_tools(HoverTool(tooltips=TOOLTIPS))

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "8pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = pi / 3

    p.rect(x="cluster", y="feat", width=1, height=1,
           source=df,
           fill_color={'field': 'm_score_scaled', 'transform': mapper},
           line_color=None)

    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="8pt",
                         # ticker=BasicTicker(desired_num_ticks=len(colors)),
                         # formatter=PrintfTickFormatter(format="%d.2"),
                         label_standoff=6, border_line_color=None, location=(0, 0))
    p.add_layout(color_bar, 'right')
    save(p)

    df_allcl = pd.DataFrame(df_lab_age, columns=['cllab', 'feat', 'sex', 'age', 'score', 'scaled_score'])

    df_sepa = {}
    for cl in cl_p.keys():
        for vect in df_clsep:
            if vect[0] == cl:
                df_sepa.setdefault(cl, list()).append(vect[1:])

    for cl, dat in df_sepa.items():
        df_sepa[cl] = pd.DataFrame(df_sepa[cl], columns=['lab', 'feat', 'sex', 'age', 'score', 'scaled_score'])

    output_file(filename=os.path.join(datapath, 'level-{0}'.format(level),
                                      'HMP-level-{0}-allSubj.html'.format(level)),
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
    save(p)

    for cl in df_sepa.keys():
        output_file(filename=os.path.join(datapath, 'level-{0}'.format(level),
                                          'HMP-level-{0}-cl-{1}.html'.format(level, cl)),
                    mode='inline')
        colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41",
                  "#550b1d"]
        mapper = LinearColorMapper(palette=colors,
                                   low=df_sepa[cl].scaled_score.min(),
                                   high=df_sepa[cl].scaled_score.max())

        lab = sorted(list(set(df_sepa[cl]['lab'])))
        list_feat = sorted(list(set(df_sepa[cl]['feat'])))
        p = figure(title="Scores level {0}".format(level, cl),
                   x_range=lab, y_range=list_feat,
                   x_axis_location="above", plot_width=1500, plot_height=1000,
                   toolbar_location='below')

        TOOLTIPS = [('age', '@age'),
                    ('sex', '@sex'),
                    ('score', '@score'),
                    ('feat', '@feat'),
                    ('lab', '@lab')]

        p.add_tools(HoverTool(tooltips=TOOLTIPS))

        p.grid.grid_line_color = None
        p.axis.axis_line_color = None
        p.axis.major_tick_line_color = None
        p.xaxis.major_label_text_font_size = "7pt"
        p.yaxis.major_label_text_font_size = "7pt"
        p.axis.major_label_standoff = 0
        p.xaxis.major_label_orientation = pi / 2

        p.rect(x="lab", y="feat", width=1, height=1,
               source=df_sepa[cl],
               fill_color={'field': 'scaled_score', 'transform': mapper},
               line_color=None)

        color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="8pt",
                             ticker=BasicTicker(desired_num_ticks=len(colors)),
                             formatter=PrintfTickFormatter(format="%d.2"),
                             label_standoff=6, border_line_color=None, location=(0, 0))
        p.add_layout(color_bar, 'right')
        save(p)

    for c, st in cl_stat.items():
        print("Cluster {0} -- Level-{1}\n".format(c, level))
        for s, val in st.items():
            print("Term {0} -- Mean score: {1:.2f} ({2:.2f}, {3:.2f}) -- SD: {4:.2f})".format(s, np.mean(val),
                                                                                              np.min(val), np.max(val),
                                                                                              np.std(val)))
        print("\n")
    print("\n\n")


def pairwise_ttest(list_feat, score_dict):
    pval_dict = {ky: {} for ky in list_feat}
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


if __name__ == '__main__':
    inspect_subcl(datapath, 1)

    inspect_subcl(datapath, 2)

    inspect_subcl(datapath, 3)

    inspect_subcl(datapath, 4)
