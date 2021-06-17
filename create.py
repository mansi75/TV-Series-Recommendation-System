import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


stopset = set(stopwords.words('english'))

data = pd.read_csv('data.csv', nrows = 8000)

data['comb'] =  data['Genre'] + ' '+ data['Description'] 


cv = CountVectorizer(min_df = 0.2,stop_words=stopset)
count_matrix = cv.fit_transform(data['comb'].values.astype('U'))



sim = cosine_similarity(count_matrix)

np.save('similarity_matrix', sim)


data.to_csv('data.csv',index=False)

