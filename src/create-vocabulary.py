import csv
import os
import re
import argparse
import sys
import numpy as np
from time import time


def create_vocabulary(indir, outdir, level=None):
    headers = _load_headers(indir, 'header-tables.csv')
    cscores = _load_data(indir, 'person-scores.csv')  # [ins_str, age in yrs, sex, DOB, DOA, ...]

    # behavioral ehrs (create vocabulary terms)
    behr = {}
    for lab, ins in cscores.items():
        for el in ins:
            if bool(re.match('ados', el[0])):
                tmp = el[0:3] + ['::'.join([el[0],
                                            headers[el[0]][idx - 1],
                                            el[idx]]) for idx in range(6, len(el)) if el[idx] != '']
            elif bool(re.match('vin|psi|srs', el[0])):
                tmp = el[0:3] + ['::'.join([el[0],
                                            el[6],
                                            headers[el[0]][idx - 1],
                                            el[idx]]) for idx in range(7, len(el)) if el[idx] != '']
            else:
                tmp = el[0:3] + ['::'.join([el[0], headers[el[0]][idx - 1],
                                            el[idx]]) for idx in range(6, len(el)) if el[idx] != '']
            behr.setdefault(lab, list()).append(tmp)
        behr[lab] = sorted(behr[lab], key=lambda x: (x[1], x[0]))
    # select the depth of scores
    deep_behr = {}
    if level == '1':
        for lab, ins in behr.items():
            for el in ins:
                if bool(re.match('leit', el[0])):
                    tmp = el[0:3] + list(filter(lambda x: bool(re.search('scaled', x)), el))
                elif bool(re.match('vinel', el[0])):
                    tmp = list(filter(lambda x: bool(re.search('scaled', x)), el))
                    tmp = el[0:3] + list(map(lambda x: '::'.join(['vineland::caretaker'] + x.split('::')[2::]), tmp))
                elif bool(re.match('wa|wi|wpp', el[0])):
                    tmp = list(filter(lambda x: bool(re.match('scaled_(bd|si|mr|vc::|ss|oa|in|cd|co|pc::|pcn::)',
                                                               x)), el))
                    tmp = el[0:3] + list(map(lambda x: '::'.join(['wechsler', x.split('_')[1]]), tmp))
                elif bool(re.match('ados-2modulo[1|toddler]', el[0])):
                    tmp = list(filter(lambda x: bool(re.search('\.[a|b|d]', x)), el))
                    tmp = list(map(lambda x: '::'.join([x.split('::')[0], x.split('.')[1]]), tmp))
                    tmp_mod = []
                    for t in tmp:
                        try:
                            ss = t.split('::')[1]
                            sc = t.split('::')[2]
                            if ss == 'd1' or ss == 'b1' or ss == 'd2':
                                tmp_mod.append('::'.join(['ados', ss, sc]))
                            else:
                                tmp_mod.append(t)
                        except IndexError:
                            pass
                    tmp = el[0:3] + tmp_mod
                elif bool(re.match('ados-2modulo[2|3|4]', el[0])):
                    tmp = list(filter(lambda x: not (bool(re.search('tot|score|lang|algor', x))),
                                      el))
                    tmp_mod = []
                    for t in tmp:
                        try:
                            ss = t.split('::')[1]
                            sc = t.split('::')[2]
                            if ss == 'd1' or ss == 'b1' or ss == 'd2':
                                tmp_mod.append('::'.join(['ados', ss, sc]))
                            else:
                                tmp_mod.append(t)
                        except IndexError:
                            pass
                    tmp = el[0:3] + tmp_mod
                elif bool(re.match('psi', el[0])):
                    tmp = el[0:3] + list(filter(lambda x: bool(re.search('raw', x))
                                                          and not (bool(re.search('raw_dr|raw_ts', x))),
                                                el))
                elif bool(re.match('srs', el[0])):
                    tmp = list(filter(lambda x: bool(re.search('raw', x))
                                                          and not (bool(re.search('raw_tot', x))),
                                                el))
                    tmp = el[0:3] + list(map(lambda x: '::'.join(['srs::caretaker'] + x.split('::')[2::]), tmp))
                elif bool(re.match('griffiths', el[0])):
                    tmp = list(filter(lambda x: bool(re.search('q_', x)), el))
                    tmp = el[0:3] + list(map(lambda x: '::'.join(['gmds'] + x.split('::')[1::]), tmp))
                deep_behr.setdefault(lab, list()).append(tmp)
    elif level == '2':
        for lab, ins in behr.items():
            for el in ins:
                if bool(re.match('wa|wi|wpp', el[0])):
                    tmp = list(filter(lambda x: bool(re.search('sumScaled_[PR|VC|V|P|WM|PS|PO|GL]', x)), el))
                    tmp = el[0:3] + list(map(lambda x: '::'.join(['wechsler', x.split('_')[1]]), tmp))
                elif bool(re.match('leit', el[0])):
                    tmp = el[0:3] + list(
                        filter(lambda x: bool(re.search('scaled', x)), el))  # scaled scores for LEITER
                elif bool(re.match('vinel', el[0])):
                    tmp = list(filter(lambda x: bool(re.search('sum_', x)), el))
                    tmp = el[0:3] + list(map(lambda x: '::'.join(['vineland::caretaker'] + x.split('::')[2::]), tmp))
                elif bool(re.match('ados-2modulo[1|toddler]', el[0])):
                    tmp = list(filter(lambda x: bool(re.search('\.sa_tot|\.rrb_tot', x)), el))
                    tmp = el[0:3] + list(map(lambda x: '::'.join(['ados'] + x.split('.')[1::]), tmp))
                elif bool(re.match('ados-2modulo[2|3]', el[0])):
                    tmp = list(filter(lambda x: bool(re.search('::rrb_tot|::sa_tot', x)),
                                      el))
                    tmp = el[0:3] + list(map(lambda x: '::'.join(['ados'] + x.split('::')[1::]), tmp))
                # elif bool(re.match('ados-2modulo4', el[0])):
                #     tmp = el[0:3] + list(filter(lambda x: bool(re.search('comm_|::si_|sbri_', x)), el))
                elif bool(re.match('psi', el[0])):
                    tmp = el[0:3] + list(filter(lambda x: bool(re.search('raw', x))
                                                          and not (bool(re.search('raw_dr|raw_ts', x))),
                                                el))  # same as level 1
                elif bool(re.match('srs', el[0])):
                    tmp = list(filter(lambda x: bool(re.search('raw', x))
                                                      and not (bool(re.search('raw_tot', x))),
                                                      el))  # same as level 1
                    tmp = el[0:3] + list(map(lambda x: '::'.join(['srs::caretaker'] + x.split('::')[2::]), tmp))
                elif bool(re.match('griffiths', el[0])):
                    # GMDS keep quotients
                    tmp = list(filter(lambda x: bool(re.search('q_', x)), el))
                    tmp = el[0:3] + list(map(lambda x: '::'.join(['gmds'] + x.split('::')[1::]), tmp))
                deep_behr.setdefault(lab, list()).append(tmp)
    elif level == '3':
        for lab, ins in behr.items():
            for el in ins:
                if bool(re.match('wa|wi|wpp', el[0])):
                    tmp = list(filter(lambda x: bool(re.search('::FSIQ', x)), el))
                    tmp = el[0:3] + list(map(lambda x: '::'.join(['wechsler', x.split('::')[1]]), tmp))
                elif bool(re.match('leit', el[0])):
                    tmp = el[0:3] + list(filter(lambda x: bool(re.search('composite_fr|::BIQ', x)), el))
                elif bool(re.match('vinel', el[0])):
                    tmp = list(filter(lambda x: bool(re.search('standard_ABC', x)), el))
                    tmp = el[0:3] + list(map(lambda x: '::'.join(['vineland::caretaker'] + x.split('::')[2:]), tmp))
                elif bool(re.match('ados-2modulo[1|2|3|toddler]', el[0])):
                    tmp = list(filter(lambda x: bool(re.search('::sarrb_tot|comparison_score', x)),
                                                el))
                    tmp = el[0:3] + list(map(lambda x: '::'.join(['ados'] + x.split('::')[1::]), tmp))
                # elif bool(re.match('ados-2modulo4', el[0])):
                #     tmp = el[0:3] + list(filter(lambda x: bool(re.search('commsi_|sbri_', x)), el))
                elif bool(re.match('psi', el[0])):
                    tmp = el[0:3] + list(filter(lambda x: bool(re.search('::raw_ts', x))
                                                          and not(bool(re.search('raw_dr', x))),
                                                el))
                elif bool(re.match('srs', el[0])):
                    tmp = list(filter(lambda x: bool(re.search('::raw_tot', x)),
                                                el))
                    tmp = el[0:3] + list(map(lambda x: '::'.join(['srs::caretaker'] + x.split('::')[2::]), tmp))
                elif bool(re.match('griffiths', el[0])):
                    tmp = list(filter(lambda x: bool(re.search('GQ', x)), el))
                    tmp = el[0:3] + list(map(lambda x: '::'.join(['gmds'] + x.split('::')[1::]), tmp))
                deep_behr.setdefault(lab, list()).append(tmp)
    else:
        for lab, ins in behr.items():
            for el in ins:
                if bool(re.match('wa|wi|wpp', el[0])):
                    tmp = list(filter(lambda x: bool(re.search('::FSIQ', x)), el))
                    tmp = el[0:3] + list(map(lambda x: '::'.join(['wechsler', x.split('::')[1]]), tmp))
                elif bool(re.match('leit', el[0])):
                    tmp = el[0:3] + list(filter(lambda x: bool(re.search('composite_fr|::BIQ', x)), el))
                elif bool(re.match('vinel', el[0])):
                    tmp = list(filter(lambda x: bool(re.search('standard_[MSD|DLSD|CD|SD|ABC]', x)), el))
                    tmp = el[0:3] + list(map(lambda x: '::'.join(['vineland', 'caretaker'] + x.split('::')[2::]), tmp))
                elif bool(re.match('ados-2modulo[1|toddler]', el[0])):
                    tmp = list(filter(lambda x: bool(re.search('\.sa_tot|\.rrb_tot|comparison_score', x)), el))
                    tmp = el[0:3] + list(map(lambda x: _subadosstring_(x), tmp))
                elif bool(re.match('ados-2modulo[2|3]', el[0])):
                    tmp = list(filter(lambda x: bool(re.search('::rrb_tot|::sa_tot|comparison_score', x)),
                                      el))
                    tmp = el[0:3] + list(map(lambda x: '::'.join(['ados'] + x.split('::')[1::]), tmp))
                elif bool(re.match('psi', el[0])):
                    tmp = el[0:3] + list(filter(lambda x: bool(re.search('::raw_ts', x))
                                                          and not(bool(re.search('raw_dr', x))),
                                                el))
                elif bool(re.match('srs', el[0])):
                    tmp = list(filter(lambda x: bool(re.search('::raw_tot', x)),
                                                el))
                    tmp = el[0:3] + list(map(lambda x: '::'.join(['srs', 'caretaker'] + x.split('::')[2::]), tmp))
                elif bool(re.match('griffiths', el[0])):
                    tmp = list(filter(lambda x: bool(re.search('q_|GQ', x)), el))
                    tmp = el[0:3] + list(map(lambda x: '::'.join(['gmds'] + x.split('::')[1::]), tmp))
                deep_behr.setdefault(lab, list()).append(tmp)

    lab_to_idx = {}
    idx_to_lab = {}
    idx = 0
    for lab, seq in deep_behr.items():
        for vec in seq:
            for v in vec[3::]:
                if v not in lab_to_idx:
                    lab_to_idx[v] = idx
                    idx_to_lab[idx] = v
                    idx += 1

    print("Depth level for test/subtest: {0} -- Vocabulary size: {1}\n".format(int(level), len(lab_to_idx)))
    seq_len = np.array([len(ins) for ins in behr.values()])
    print("Average length of behavioral sequences: {0}\n".format(np.mean(seq_len)))

    # write files
    with open(os.path.join(outdir, 'cohort-behr.csv'), 'w') as f:
        wr = csv.writer(f)
        wr.writerow(['ID_SUBJ', 'EVAL_AGE', 'TERM'])
        for lab, seq in deep_behr.items():
            for s in seq:
                wr.writerow([lab, s[1]] + [lab_to_idx[s[idx]] for idx in range(3, len(s))])

    with open(os.path.join(outdir, 'cohort-vocab.csv'), 'w') as f:
        wr = csv.writer(f)
        wr.writerow(['LABEL', 'INDEX'])
        for l, idx in lab_to_idx.items():
            wr.writerow([l, idx])


"""
Private Functions
"""


def _load_headers(indir, file):
    with open(os.path.join(indir, file)) as f:
        rd = csv.reader(f)
        mydict = {r[0]:r[1::] for r in rd}
    return mydict


def _load_data(indir, file):
    with open(os.path.join(indir, file)) as f:
        rd = csv.reader(f)
        mydict = {}
        for r in rd:
            mydict.setdefault(r[0], list()).append(r[1::])
    return mydict


def _subadosstring_(item):
    if re.search('comparison_score', item):
        item = '::'.join(['ados'] + item.split('::')[1::])
    else:
        item = '::'.join(['ados'] + item.split('.')[1::])
    return item


"""
Main Function
"""


def _process_args():
    parser = argparse.ArgumentParser(
        description='create vocabulary for behavioral phenotype,'
                    ' 3 possible levels')
    parser.add_argument(dest='indir', help='data directory')
    parser.add_argument(dest='level', help='choose behavioral profile depth')
    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    args = _process_args()
    print('')

    start = time()
    outdir = os.path.join(args.indir, 'level-{0}'.format(args.level))
    os.makedirs(outdir, exist_ok=True)
    create_vocabulary(indir=args.indir,
                      outdir=outdir,
                      level=args.level)
