## Behavioral stratification in Autism Spectrum Disorder

Run `create-tables.py` that outputs the following files:

- _person-instrument.csv_: ID_SUBJ, DOB, DOA, SEX, INSTRUMENT;

- _person-scores.csv_: file with the lab identifier as first column, instrument
  name in the second and assessment age in the third. Following, the scores of the 
  instrument as they appear in the Google sheets;

- _person_demographics.csv_: ID_SUBJ, CURRENT_AGE, DOB, DOA, SEX;
  DOB: date of birth;
  DOA: date of assessment.

- _header-tables.csv_: name of the instrument in the first column and following, the header
  from the corresponding Google sheet;

- _odf-tables.pkl_: `pickle` object storing the dictionary "table_name":{list representing 
  the Google sheet}.

Run `tokenize-dataset.sh` with the level of depth of tests (1 = subtests, 2=subscales, 3=indexes).

The bash file calls `create-vocabulary.py` that builds the vocabularies according to the specified level.

The vocabularies are seved in a folder _level-N_ with $N$ the level. 
