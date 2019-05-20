import sqlalchemy
import pymysql
from sqlalchemy import create_engine, inspect, MetaData, select
import datetime
from datetime import date
from time import time
from decimal import *
import csv
import os 
import pickle


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def age(birthDate, assessmentDate):
    days_in_year = 365.2425
    try:
        assDate = datetime.datetime.strptime(assessmentDate, '%d/%m/%Y').date()
        bDate = datetime.datetime.strptime(birthDate, '%d/%m/%Y').date()
        assAge = (assDate - bDate).days / days_in_year
    except TypeError:
        bDate = datetime.datetime.strptime(birthDate, '%d/%m/%Y').date()
        assAge = -1
    currentAge = (date.today() - bDate).days / days_in_year
    return currentAge, assAge


SQLALCHEMY_CONN_STRING = 'mysql+pymysql://odflab:LAB654@192.168.132.114/odflab'
DATA_FOLDER_PATH = os.path.expanduser('~/Documents/nd_subtyping/data/')

# connect to the database
engine = create_engine(SQLALCHEMY_CONN_STRING)
conn = engine.connect()

# inspect the tables in the database
# inspector = inspect(engine) if we want to inspect the tables (inspector.get_table_names())
metadata = MetaData(engine, reflect=True)

subject_list = []
for table_name in metadata.tables:
    # if table_name != 'ados-2modulotoddler' and table_name != 'ados-2modulo4':
    table_tmp = metadata.tables[table_name]
    s = select([table_tmp.c.id_subj, table_tmp.c.date_birth, table_tmp.c.date_ass, table_tmp.c.sex])
    result = conn.execute(s)
    for r in result:
        try: 
            subject_list.append([r[0], r[1].strftime('%d/%m/%Y'), r[2].strftime('%d/%m/%Y'), r[3], table_name])
        except AttributeError:
            try:
                subject_list.append([r[0], r[1].strftime('%d/%m/%Y'), r[2], r[3],
                                    table_name])  # case datetime variable, None
            except AttributeError:
                subject_list.append([r[i] for i in range(len(r))] + [table_name])
subject_list.sort(key=lambda x: x[0])

table_names = []
tables = {}
for table_name in metadata.tables:
    table = metadata.tables[table_name]
    sql = select([c for c in table.c])
    result = conn.execute(sql)
    tables.setdefault(table_name, 
                      list()).extend([result.keys()]+[list(r[2::]) for r in result])

for t in tables.values():
    for l in t:
        for idx, el in enumerate(l):
            if type(el) is datetime.datetime:
                l[idx] = el.strftime('%d/%m/%Y')
            elif type(el) is Decimal:
                l[idx] = int(el)

subj_values = {}
header_tables = {}
subj_demographics = {}
for ins, meas in tables.items():
    header_tables.setdefault(ins, list()).extend(['eval_age'] + meas[0][3::])
    for m in meas[1::]:
        current_age, eval_age = age(m[2], m[3])
        if m[0] not in subj_values:
            subj_values[m[0]] = [[ins, eval_age] + m[1::]]
            subj_demographics[m[0]] = [current_age, m[2], m[3], m[1]]
        else:
            subj_values[m[0]].append([ins, eval_age] + m[1::])
for lab in subj_values.keys():
    subj_values[lab].sort(key=lambda x: x[1])

# Save objects
data_dir = '-'.join(['odf-data', datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')])
os.makedirs(os.path.join(DATA_FOLDER_PATH, data_dir))

save_obj(tables, os.path.join(DATA_FOLDER_PATH, data_dir, 'odf-tables'))

with open(os.path.join(DATA_FOLDER_PATH, data_dir, 'person-instrument.csv'), 'w') as f:
    wr = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    wr.writerow(['ID_SUBJ', 'DOB', 'DOA', 'SEX', 'INSTRUMENT'])
    for sl in subject_list:
        wr.writerow(sl)

with open(os.path.join(DATA_FOLDER_PATH, data_dir, 
                       'header-tables.csv'), 'w') as f:
    wr = csv.writer(f, delimiter=',')
    for h in header_tables:
        wr.writerow([h] + [c for c in header_tables[h]])

with open(os.path.join(DATA_FOLDER_PATH, data_dir,
                       'person-scores.csv'), 'w') as f:
    wr = csv.writer(f, delimiter=',')
    for l, m in subj_values.items():
        for i in range(len(m)):
            wr.writerow([l] + [v for v in m[i]])

with open(os.path.join(DATA_FOLDER_PATH, data_dir,
                       'person-demographics.csv'), 'w') as f:
    wr = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    wr.writerow(['ID_SUBJ', 'CURRENT_AGE', 'DOB', 'DOA', 'SEX'])
    for s, sd in subj_demographics.items():
        wr.writerow([s] + sd)
