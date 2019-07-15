import os
import csv
import numpy as np
from statsmodels.stats.proportion import proportions_ztest

datafolder = 'odf-data-2019-05-29-16-12-13'
datadir = os.path.join(os.path.expanduser('~/Documents/nd_subtyping/data'), datafolder)
level = 'level-1'

with open(os.path.join(datadir, level, 'person-cluster.txt')) as f:
    rd = csv.reader(f)
    next(rd)
    p_clu = {r[0]: r[1] for r in rd}

with open(os.path.join(datadir, level, 'cohort-behr.csv')) as f:
    rd = csv.reader(f)
    next(rd)
    behr = {r[0]: r[2::] for r in rd}

term_cl = {}
set_terms = set()
for p_id, cl in p_clu.items():
    term_cl.setdefault(cl, list()).extend(behr[p_id])
    set_terms.update(behr[p_id])

term_prop = {t: [None] * len(term_cl) for t in set_terms}
for cl, term in term_cl.items():
    for t in set_terms:
        term_prop[t][int(cl)] = (term.count(t), len(term))

# ztest of proportion with bonferroni-corrected p-values
s = (len(term_cl.keys())-1, len(term_cl.keys())-1)
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
                result_ztest[t]['pval'][cl][idx-1] = pval
                count_comp += 1
            idx += 1
    result_ztest[t]['pval'] = result_ztest[t]['pval'] * count_comp

