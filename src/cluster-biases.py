import os
import csv
import numpy as np
import datetime
from scipy import stats
from datetime import date, datetime


def age(birthDate, assessmentDate):
    days_in_year = 365.2425
    try:
        assDate = datetime.strptime(assessmentDate, '%d/%m/%Y').date()
        bDate = datetime.strptime(birthDate, '%d/%m/%Y').date()
        assAge = (assDate - bDate).days / days_in_year
    except TypeError:
        bDate = datetime.strptime(birthDate, '%d/%m/%Y').date()
        assAge = -1
    currentAge = (date.today() - bDate).days / days_in_year
    return currentAge, assAge


level_dir = 'level-4'

data = os.path.expanduser('~/Documents/nd_subtyping/data/odf-data-2019-06-13-12-28-37/')
datadir = data + level_dir


# set of instruments for each subject
with open(os.path.join(data, 'person-instrument.csv')) as f:
    rd = csv.reader(f)
    next(rd)
    person_ins = {}
    for r in rd:
        if r[-1] != 'emotionavailabilityscales':
            person_ins.setdefault(r[0], set()).add(r[-1])

with open(os.path.join(datadir, 'person-cluster.txt')) as f:
    rd = csv.reader(f)
    next(rd)
    person_cl_ins = {}
    subj_cl = {}
    cl_subj = {}
    for r in rd:
        person_cl_ins.setdefault(r[1], set()).update(person_ins[r[0]])
        subj_cl[r[0]] = r[1]

with open(os.path.join(data, 'person-demographics.csv')) as f:
    rd = csv.reader(f)
    next(rd)
    cl_info = {}
    for r in rd:
        ca, _ = age(r[1], r[2])
        cl_info.setdefault(subj_cl[r[0]], list()).append([ca] + [r[3], int(r[4])])

dict_tests = {}
for cl, info in cl_info.items():
    age = []
    sex = []
    enc = []
    for el in info:
        age.append(el[0])
        sex.append(el[1])
        enc.append(el[2])
    dict_tests[cl] = [age, sex, enc]
    print("Cluster: {0}".format(cl))
    print("Mean age: {0:.2f} -- Min/Max ({1:.2f}, {2:.2f})".format(np.mean(age),
                                                           np.min(age),
                                                           np.max(age)))
    print("Average number of encounters: {0:.2f} -- Min/Max ({1}, {2})".format(np.mean(enc),
                                                                               np.min(enc), np.max(enc)))
    print("Sex counts: F = {0} -- M = {1}".format(sex.count('Femmina'), sex.count("Maschio")))
    print("Instrument list:")
    for ins in person_cl_ins[cl]:
        print(ins)
    print("\n")

with open(os.path.join(datadir, 'cluster-stats.csv'), 'w') as f:
    wr = csv.writer(f)
    wr.writerow(['CLUSTER', 'AGE', 'SEX', 'N_ENCOUNTERS'])
    for cl, vec in cl_info.items():
        for v in vec:
            wr.writerow([cl] + v)


print(stats.ttest_ind(dict_tests['0'][0], dict_tests['1'][0]))
print(stats.ttest_ind(dict_tests['1'][0], dict_tests['2'][0]))
print(stats.ttest_ind(dict_tests['2'][0], dict_tests['3'][0]))
print(stats.ttest_ind(dict_tests['0'][0], dict_tests['2'][0]))
print(stats.ttest_ind(dict_tests['0'][0], dict_tests['3'][0]))
print(stats.ttest_ind(dict_tests['1'][0], dict_tests['3'][0]))

print(stats.ttest_ind(dict_tests['0'][2], dict_tests['1'][2]))
print(stats.ttest_ind(dict_tests['1'][2], dict_tests['2'][2]))
print(stats.ttest_ind(dict_tests['2'][2], dict_tests['3'][2]))
print(stats.ttest_ind(dict_tests['0'][2], dict_tests['2'][2]))
print(stats.ttest_ind(dict_tests['0'][2], dict_tests['3'][2]))
print(stats.ttest_ind(dict_tests['1'][2], dict_tests['3'][2]))

