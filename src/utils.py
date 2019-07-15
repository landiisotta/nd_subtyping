import matplotlib


"""
Utils file storing parameters
"""


min_cl = 3
max_cl = 15

n_epoch = 1
batch_size = 5
# dimension of SVD and word embeddings
n_dim = 10

file_names = {'vocab': 'cohort-vocab.csv',
              'behr': 'cohort-behr.csv'}

col_dict = matplotlib.colors.CSS4_COLORS
c_out = ['bisque', 'mintcream', 'cornsilk', 'lavenderblush', 'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure',
         'beige', 'powderblue', 'floralwhite', 'ghostwhite', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow',
         'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue',
         'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'linen', 'palegoldenrod', 'palegreen',
         'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'mistyrose', 'lemonchiffon', 'lightblue',
         'seashell', 'white', 'blanchedalmond', 'oldlace', 'moccasin', 'snow', 'darkgray', 'ivory', 'whitesmoke']
