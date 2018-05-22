import pandas as pd

file = "/Users/yoon/Downloads/data2.csv"
df = pd.read_csv(file)

id_group = df.sort_values(['pcid','dt']).groupby('pcid')['tq'].apply(list)
df_id_group = id_group.add_suffix('_list').reset_index()
tokenized_contents = df_id_group['tq']

from gensim.models import Word2Vec
embedding_model = Word2Vec(tokenized_contents, size=100, window = 5, min_count=5)
embedding_model.most_similar(positive=['와플기계'])