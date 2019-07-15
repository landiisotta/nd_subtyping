import os
import csv
import subprocess
import umap
import utils as ut
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from bokeh.models import LinearColorMapper, BasicTicker, PrintfTickFormatter, ColorBar, HoverTool, ColumnDataSource
from bokeh.plotting import figure, output_file, save


'''
Functions
'''


def hclust(data):
    n_cl_selected = []
    for it in range(ut.n_iter):
        idx = np.random.randint(0, len(data), int(len(data)*ut.subsampl))
        sub_data = [data[i] for i in idx]
        best_silh = 0
        for n_clu in range(ut.min_cl, ut.max_cl):
            hclu = AgglomerativeClustering(n_clusters=n_clu)
            lab_cl = hclu.fit_predict(sub_data)
            tmp_silh = silhouette_score(sub_data, lab_cl)
            if tmp_silh > best_silh:
                best_silh = tmp_silh
                best_lab_cl = lab_cl
                best_n_clu = n_clu
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
    lab_cl = hclu.fit_predict(data)
    silh = silhouette_score(data, lab_cl)
    print('(*) Number of clusters %d -- Silhouette score %.2f' % (best_n_clu, silh))

    num_count = np.unique(lab_cl, return_counts=True)[1]
    for idx, nc in enumerate(num_count):
        print("Cluster {0} -- Numerosity {1}".format(idx, nc))
    print('\n')
    print('\n')
    
    return lab_cl



def inspect_subcl(datapath, level, cl_info, subj_cl, 
                  behr, behr_age, mod, feat_data=None):
    
    p_sex = {s: l[1] for s, l in cl_info.values().items()}
    
    if mod != 'feat':
        feat_dict = {p: {} for p in behr.keys()}
        feat_dict_age = {p: {} for p in behr_age.keys()}
        for p1, seq1, p2, seq2 in zip(behr.items(), behr_age.items()):
            for s1, s2,  in zip(seq1, seq2):
                ss1 = s1.split('::')
                feat_dict[p1].setdefault('::'.join(ss1[0:len(ss1)-1]), 
                                         list()).append(int(ss1[-1]))
                ss2 = s2[1].split('::')
                feat_dict_age[p2].setdefault('::'.join(ss2[0:len(ss2) - 1]),
                                             list()).append((s2[0], int(ss2[-1])))

        cl_stat = {cl: {} for cl in set(subj_cl.values())}
        for subj, cl in subj_cl.items():
            for ky, ft in feat_dict[subj].items():
                if len(ft) > 1:
                    mft = np.mean(ft)
                    cl_stat[cl].setdefault(ky, list()).append(mft)
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
            ds_sc[f]['val'] = [v[0] for v in scaler.fit_transform(np.array(ds_sc[f]['val']).reshape(-1,1))]
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
                #df_vect.append([cl, f, None])
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
                    #df_lab.setdefault(cl, list()).append([p, f, None])
        df_lab[cl] = sorted(df_lab[cl], key=lambda x: x[0])

    df_lab_age = []
    for cl, p_vec in cl_p.items():
        for f in list_feat:
            for p in p_vec:
                try:
                    for el in feat_dict_age[p][f]:
                        try:
                            df_lab_age.append(['-'.join([cl, p]), 
                                               f, p_sex[p]] + [float(el[0]), el[1],
                                               df_sc[f].loc['::'.join([p, el[0]]),
                                                                       'val'].item()])
                        except ValueError:
                            df_lab_age.append(['-'.join([cl, p]), 
                                               f, p_sex[p]] + [float(el[0]), el[1],
                                                               np.mean(df_sc[f].loc['::'.join([p, el[0]]),
                                                                                    'val'].tolist())])
                except KeyError:
                    pass
    df_lab_age = sorted(df_lab_age, key=lambda x: (x[0], x[1], x[3]))

    df_ttest = pairwise_ttest(list_feat, cl_stat)
    df_ttest.to_csv(os.path.join(datapath, 'level-{0}'.format(level), 
                                 '{0}-pairwise-ttest-level{1}.csv'.format(mod, 
                                 level)), 
                    index=False)

    df = pd.DataFrame(df_vect, columns=['cluster', 'feat', 'm_score', 'm_score_scaled'])
    _heatmap_plot(df, datapath, level, mean_s=True)

    df_allcl = pd.DataFrame(df_lab_age, columns=['cllab', 'feat', 'sex', 'age', 'score', 'scaled_score'])
    _heatmap_plot(df_allcl, datapath, level, mean_s=False)

    for c, st in cl_stat.items():
        print("Cluster {0} -- Level-{1}\n".format(c, level))
        for s, val in st.items():
            print("Term {0} -- Mean score: \
                   {1:.2f} ({2:.2f}, {3:.2f}) \
                   -- SD: {4:.2f})".format(s, np.mean(val),
                                           np.min(val), np.max(val), 
                                           np.std(val)))
        print("\n")
    print("\n\n")


'''
Private functions
'''


def _load_data(indir, mod):
    if mod == 'feat':
        with open(indir, 'cohort-feat-wide-scaled.csv') as f:
            rd = csv.reader()
            next(rd)
            sc_feat = {r[0]: r[1:] for r in rd}
        return sc_feat
    else:
        with open(indir, '{0}-mtx.csv'.format(mod)) as f:
            rd = csv.reader(f)
            emb_mtx = {r[0]: r[1:] for r in rd}
        return emb_mtx


def _age_(birthDate):
    days_in_year = 365.2425
    bDate = datetime.strptime(birthDate, '%d/%m/%Y').date()
    currentAge = (date.today() - bDate).days / days_in_year
    return currentAge


def _pairwise_ttest(list_feat, score_dict):
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


def _heatmp_plot(data, datapath, level, mod, mean=False):
    
    colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", 
              "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
    TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"
    
    if mean:
        output_file(filename=os.path.join(datapath, 'level-{0}'.format(level),
                                          '{0}-HMP-level-{1}.html'.format(mod, 
                                          level)),
                                          mode='inline')
        mapper = LinearColorMapper(palette=colors, low=data.m_score_scaled.min(), 
                                   high=data.m_score_scaled.max())


        clusters = sorted(list(set(data['cluster'])))
        list_feat = sorted(list(set(data['feat'])))
        p = figure(title="Mean scores level {0} for {1} clusters".format(level, 
                   len(clusters)),
                   x_range=clusters, y_range=list_feat,
                   x_axis_location="above", plot_width=600, plot_height=900,
                   toolbar_location='below')

        TOOLTIPS = [('m_score', '@m_score'),
                    ('feat', '@feat'),
                    ('cl', '@cluster')]
    
        p.add_tools(HoverTool(tooltips=TOOLTIPS))

        p.grid.grid_line_color = None
        p.axis.axis_line_color = None
        p.axis.major_tick_line_color = None
        p.axis.major_label_text_font_size = "8pt"
        p.axis.major_label_standoff = 0
        p.xaxis.major_label_orientation = pi / 3

        p.rect(x="cluster", y="feat", width=1, height=1,
               source=data,
               fill_color={'field': 'm_score_scaled', 'transform': mapper},
               line_color=None)

    else:
        output_file(filename=os.path.join(datapath, 'level-{0}'.format(level),
                                          '{0}-HMP-level-{1}-allSubj.html'.format(mod, 
                                          level)),
                                          mode='inline')
        mapper = LinearColorMapper(palette=colors, 
                                   low=data.scaled_score.min(), 
                                   high=data.scaled_score.max())

        lab = sorted(list(set(data['cllab'])))

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
               source=data,
               fill_color={'field': 'scaled_score', 'transform': mapper},
               line_color=None)

    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="8pt",
                         #ticker=BasicTicker(desired_num_ticks=len(colors)),
                         #formatter=PrintfTickFormatter(format="%d.2"),
                         label_standoff=6, border_line_color=None, location=(0, 0))
    p.add_layout(color_bar, 'right')
    save(p)


def _plot_dendrogram(data, lab_cl, datapath, mod):
    colormap = [c for c in ut.col_dict if c not in ut.c_out]
    colormap_rid = [colormap[cl] for cl in sorted(list(set(lab_cl)))]
    colors_en = [colormap_rid[v] for v in lab_cl]

    linked = linkage(data, 'ward')
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
    plt.savefig(os.path.join(datapath, 'dendrogram-{0}.png'.format(mod)))


def _scatter_plot(data, id_subj, lab_cl, dem, mod):

    umap_mtx = umap.UMAP(random_state=42).fit_transform(data)
    
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

    output_file(filename=os.path.join(datadir, '{0}-plot-interactive.html'.format(mod)), mode='inline')
    p = figure(plot_width=800, plot_height=800, tools=plotTools)
    p.add_tools(HoverTool(tooltips=TOOLTIPS))
    p.circle('x', 'y', legend='cluster', source=source, color={"field": 'cluster', "transform": cmap})
    save(p)


'''
Main function
'''


if __name__ == '__main__':

   with open(os.path.join(args.datadir, 
             'level-{0}'.format(args.level), 
             'cohort-vocab.csv')) as f:
        rd = csv.reader(f)
        next(rd)
        idx_to_bt = {r[1]: r[0] for r in rd}

    behr = {}
    behr_age = {}
    with open(os.path.join(args.datadir, 
              'level-{0}'.format(args.level), 
              'cohort-behr.csv')) as f:
        rd = csv.reader(f)
        next(rd)
        for r in rd:
            behr.setdefault(r[0], list()).extend([idx_to_bt[idx] for idx in r[2::]])
            behr_age.setdefault(r[0], list()).extend([(r[1], idx_to_bt[idx]) for idx in r[2::]])
        
    for mod in ut.model_vect[str(args.level)]:
        print("Hierarchical clustering and inspection \
               for model: {0}".format(mod.upper()))
        data_dict = _load_data(os.path.join(args.datadir, 
                               'level-{0}'.format(args.level)), 
                               mod)
        cl_vec = hclust(list(data_dict.values()))
        # plot dendrogram
        _plot_dendrogram(list(data_dict.values()), cl_vec, os.path.join(args.datadir, 
                         'level-{0}'.format(args.level)), mod)
        subj_cl {s: cl_vec[idx] for idx, s in enumerate(list(data_dict.keys()))}

        with open(os.path.join(datadir, 'person-instrument.csv')) as f:
            rd = csv.reader(f)
            next(rd)
            subj_ins = {}
            for r in rd:
                if r[-1] != 'emotionavailabilityscales':
                    subj_ins.setdefault(r[0], set()).add(r[-1])
        for s, c in subj_cl.items():
            cl_ins.setdefault(c, set()).update(subj_ins[s])
        
        with open(os.path.join(arg.datadir, 'level-{0}'.format(args.level), 
                  '{0}-cl.txt'.format(mod)), 'w') as f:
            wr = csv.writer(f)
            wr.writerow(['ID_SUBJ', 'CLUSTER'])
            for s, c in subj_cl.items():
                wr.writerow([s, c])

        with open(os.path.join(args.datadir, 'person-demographics.csv')) as f:
            rd = csv.reader(f)
            next(rd)
            cl_info = {}
            for r in rd:
                cl_info.setdefault(subj_cl[r[0]], dict()).setdefault(r[0], 
                                                          list()).extend(
                                                          [_age_(r[1]), 
                                                          r[3], int(r[4])])
        _scatter_plot(list(data_dict.values()), list(data_dict.keys()), cl_vec,
                      cl_info.values(), mod)

        dict_demo = {}
        for cl, info in cl_info.items():
            age = []
            sex = []
            n_enc = []
            for el in info.values():
                age.append(el[0])
                sex.append(el[1])
                n_enc.append(el[2])
            dict_demo[cl] = [age, sex, n_enc]
            print("Cluster: {0}".format(cl))
            print("Mean age: {0:.2f} (SD = {1:.2f}) \
                   -- Min/Max ({2:.2f}, {3:.2f})".format(np.mean(age),
                                                         np.std(age),
                                                         np.min(age),
                                                         np.max(age)))
            print("Average number of encounters: {0:.2f} \
                   -- Min/Max ({1}, {2})".format(np.mean(n_enc),
                                                 np.min(n_enc), 
                                                 np.max(n_enc)))
            print("Sex counts: F = {0} -- M = {1}".format(sex.count('Femmina'), 
                                                          sex.count("Maschio")))
            print("Instrument list:")
            for ins in cl_ins[cl]:
                print(ins)
            print("\n")

        with open(os.path.join(args.datadir, 'level-{0}'.format(args.level),
                  '{0}-cl-demographics.csv'.format(mod)), 'w') as f:
            wr = csv.writer(f)
            wr.writerow(['CLUSTER', 'AGE', 'SEX', 'N_ENCOUNTERS'])
            for cl, dem in dict_demo.items():
                wr.writerow([cl] + dem)
        print("Checking for confounders (age, sex, number of encounters):") 
        command = 'Rscript'
        path2script = os.path.expanduser('~/Documents/behavioraly_phenotyping/src/test-demog-cl.R')
        arg = [args.datapath, args.level, mod]
        cmd = [command, path2script] + arg
        x = subprocess.check_output(cmd, 
        universal_newlines=True)
