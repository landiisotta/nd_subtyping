import os
import csv

level = 4


with open('../data/odf-data-2019-06-13-12-28-37/level-4/cohort-behr.csv') as f:
    rd = csv.reader(f)
    next(rd)
    behr = {}
    for r in rd:
        behr.setdefault(r[0], list()).append(r[1::])
