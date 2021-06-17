import pandas as pd
import numpy as np
import re

data = pd.read_csv("All_Streaming_Shows.csv")


data.to_csv('data.csv',index = False)

data = data.loc[:,['Series Title','Genre','Description']]


def fin(li):
    return re.sub(r'[^\w\s]', ' ', li)

data['Genre'] = data['Genre'].apply(lambda x: fin(x))



data.drop_duplicates(subset ="Series Title", keep = 'last', inplace = True)
data['Series Title'] = data['Series Title'].str.lower()

data.to_csv('data.csv',index=False)





