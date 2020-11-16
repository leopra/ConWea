import os
import sys
import pickle
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from langdetect import detect
import numpy as np
import matplotlib.pyplot as plt

PATH = os.path.dirname(os.__file__)
sys.path.append(os.path.join(PATH,'Libraries-GP'))

# eutop Libraries
from SQLServer import DATABASE_CONFIG_NEW, sql_cnnt
from LeoLibs import CrunchbaseDatasetProcessing as cp

data = pd.read_sql_query("""select * from ml_crunchbase  where id in (select min(id) from ml_crunchbase group by cb_url)""",
    sql_cnnt("cnnt", DATABASE_CONFIG_NEW))


verticals = pd.read_sql_query("""select * from tb_verticals""",
    sql_cnnt("cnnt", DATABASE_CONFIG_NEW))
# remove url duplicates
data = data.drop_duplicates(subset=["cb_url"])
# remove companies with empy description
data = data[data['description'] != '—']

# create column with eutopia categories as list for 1hot encoding
# CREATES COLUMN 'LISTLABEL'
data = cp.assign_eutop_labelsV2(data)


#create single label dataframe compatible with conwea
dfcon = data[['id', 'description', 'listlabel']]
dfcon = dfcon[dfcon['listlabel'].map(len) > 0]

dfcon['todrop'] = ['Telecommunications & ICT' in r for r in dfcon['listlabel']]
dfcon = dfcon[dfcon['todrop'] == False]

#drop empty sentences
dfcon = dfcon[dfcon['description'].map(len) > 20]

def detectcatcherror(stringa):
    try:
        language = detect(stringa)
    except:
        language = "error"
        print("This row throws and error:", stringa)
    return language

#detect description not in english
#dfcon['lang'] = dfcon['description'].apply(lambda x: detectcatcherror(x))

with open('../data/eutopiavert/df.pkl', 'rb') as handle:
    dfcon = pickle.load(handle)

dfconengl = dfcon
dfconengl = dfcon[dfcon['lang'] == 'en']

mlb = MultiLabelBinarizer()
categories_1hot = mlb.fit_transform(dfconengl.listlabel)
categories_cols = mlb.classes_

# dataready = pd.DataFrame(categories_1hot, columns=categories_cols, index=data.index)],axis=1)
#
# #see categories distribution
# fig, ax = plt.subplots(1, 1, figsize=(15, 10))
# ax.bar(categories_cols, dataready[categories_cols].sum());
# ax.set_title('Categories')
# ax.grid()

dfconengl['label'] = list(categories_1hot)
#this is the final dataset onehotencoded
dfconengl = dfconengl[['description','label']]
#lowewrcase everything to check for errors
b['sentence'] = b['sentence'].apply(lambda x: x.lower())
dfconengl.columns= ['sentence','label']

#code to see labels distribution
twodmatrix = np.stack(data15k.label.values, axis=0)
labelcounts = np.sum(twodmatrix, axis=0)
plt.bar(range(0,13), labelcounts)

data15k = b.sample(20000)

data15keq = data15k[data15k['4']==0]

for i in range(13):
    b[str(i)] = b['label'].apply(lambda x: 1 if x[i]==1 else 0)

dataset = pd.DataFrame()
for i in range(13):
    x = b[b[str(i)]==1].sample(1200)

#code to see 1-2-3 labels distribution
multilabelpersample = np.sum(twodmatrix, axis=1)
counts=[]
for i in range(5):
    counts.append(np.count_nonzero(multilabelpersample == i))

plt.bar(range(5), counts)


out = b.reset_index().drop('index', axis=1)
out.to_pickle('./data/eutopiavert/df.pkl', protocol=3)

#
# #see descriptions length
# def clean_and_tokenize(text):
#     text = re.sub('[;,:\!\?\.\(\)\n]', ' ', text).replace('[\s+]', ' ')
#     return nltk.word_tokenize(text)
#
#
with open('./data/eutopiavert/df.pkl', 'rb') as handle:
    b = pickle.load(handle)


#
# smaller = b.groupby('label', as_index=False).apply(lambda x: x.sample(20))
# smaller = smaller.reset_index().drop('level_0', axis=1).drop('level_1', axis=1).sample(frac=1).reset_index()
# smaller.to_pickle('./data/models/df.pkl', protocol=3)

######### CODE TO SAMPLE BALANCED DATASET
with open('./data/eutopiavert/df.pkl', 'rb') as handle:
    b = pickle.load(handle)

#split labels in multiple columns for easier sampling
for i in range(13):
    b[str(i)] = b['label'].apply(lambda x: 1 if x[i]==1 else 0)

onelab = b[b['label'].map(sum)==1]
twolab = b[b['label'].map(sum)==2]
threelab = b[b['label'].map(sum)==3]
forlab = b[b['label'].map(sum)==4]

baldata = pd.DataFrame()
nsample = 1000

#sample for each class (if i can) then plot distribution
for i in range(13):
    x = onelab[onelab[str(i)] == 1]
    if len(x) > nsample:
        sam = x.sample(nsample)
    else:
        sam = x
    baldata = baldata.append(sam)

#see if it's balanced
twodmatrix = np.stack(baldata.label.values, axis=0)
labelcounts = np.sum(twodmatrix, axis=0)
plt.bar(range(0,13), labelcounts)

out = baldata.reset_index().drop('index', axis=1)
out.to_pickle('./data/eutopiavert12000balancedoneclass/df.pkl', protocol=3)