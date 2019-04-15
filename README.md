## Behavioral stratification in Autism Spectrum Disorder

Run `odfDBquery.py` that outputs the followng files:

- _person-instrument.csv_: ID_SUBJ, DOB, DOA, SEX, INSTRUMENT;

- _person-scores.csv_: file with the lab identifier as first column, instrument
  name in the second and assessment age in the third. Following, the scores of the 
  instrument as they appear in the Google sheets;

- _person_demographics.csv_: ID_SUBJ, CURRENT_AGE, SEX;

- _header-tables.csv_: name of the instrument in the first column and following, the header
  from the corresponding Google sheet;

- _odf-tables.pkl_: `pickle` object storing the dictionary "table_name":{list representing 
  the Google sheet}. 
