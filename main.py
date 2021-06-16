# Hotel Reviews Category and Sentiment Extractor
# Gabriela Kuhn - June 2021

import numpy as np
import pandas as pd
import re
import pickle
from nltk import word_tokenize
import matplotlib.pyplot as plt  # to create the bargraph
import pylab

# import data for mapping
df = pd.read_csv("dataset/customer_reviews.csv");
df_tx = pd.read_csv("dataset/taxonomy.csv");

# taxonomy words ending with (*)
def asterix_handler(asterixw, lookupw):
    mtch = "F"
    for word in asterixw:
        for lword in lookupw:
            if (word[-1:] == "*"):
                if (bool(re.search("^" + word[:-1], lword)) == True):
                    mtch = "T"
                    break
    return (mtch)


# treating punctuations.
def remov_punct(withpunct):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    without_punct = ""
    char = 'nan'
    for char in withpunct:
        if char not in punctuations:
            without_punct = without_punct + char
    return (without_punct)


# treating quotes("").
def remov_quote(withquote):
    quote = '"'
    without_quote = ""
    char = 'nan'
    for char in withquote:
        if char not in quote:
            without_quote = without_quote + char
    return (without_quote)

# classifing the objects as negative and pos = if pos = 1 if neg = -1
# Loading pickled objects.
with open("features_file.pickle", "rb") as features_file:
    features = pickle.load(features_file)

with open("classifier_file.pickle", "rb") as classifier_file:
    classifier = pickle.load(classifier_file)

def extract_features(review):
    feature_set = {}
    review_words = word_tokenize(review)
    for feature in features:
        feature_set["contains({})".format(feature)] = (feature in review_words)
    return feature_set

def get_sentiment_from(review):
   return (classifier.classify(extract_features(review)))

# identify Sentiment score
def findscore(test_data):
    sentiment = get_sentiment_from(test_data)
    if (sentiment == 'pos'):
        score = 1
    elif (sentiment == 'neg'):
        score = -1
    else:
        score = 0
    return (score)


# Category mapping
# split sentences and append one below the other for categorization
sentence_data = pd.DataFrame(columns=['slno', 'text'])

for d in range(len(df)):
    doc = (df.iloc[d, 1].split('.'))
    for s in ((doc)):
        temp = {'slno': [df['slno'][d]], 'text': [s]}
        sentence_data = pd.concat([sentence_data, pd.DataFrame(temp)])
        temp = ""

# drop empty text rows and export data
sentence_data['text'].replace('', np.nan, inplace=True);
sentence_data.dropna(subset=['text'], inplace=True);

data = sentence_data
cat2list = list(set(df_tx['Subtopic']))
data['Category'] = 0
mapped_data = pd.DataFrame(columns=['slno', 'text', 'Category']);
temp = pd.DataFrame()

#taxonomy
for k in range(len(data)):
    comment = remov_punct(data.iloc[k, 1])
    data_words = [str(x.strip()).lower() for x in str(comment).split()]
    data_words = list(filter(None, data_words))  # new way to do filter in python 3.x
    output = []

    for l in range(len(df_tx)):
        key_flag = False
        and_flag = False
        not_flag = False
        if (str(df_tx['PrimaryKeywords'][l]) != 'nan'):
            kw_clean = (remov_quote(df_tx['PrimaryKeywords'][l]))
        if (str(df_tx['AdditionalKeywords'][l]) != 'nan'):
            aw_clean = (remov_quote(df_tx['AdditionalKeywords'][l]))
        else:
            aw_clean = df_tx['AdditionalKeywords'][l]
        if (str(df_tx['ExcludeKeywords'][l]) != 'nan'):
            nw_clean = remov_quote(df_tx['ExcludeKeywords'][l])
        else:
            nw_clean = df_tx['ExcludeKeywords'][l]
        Key_words = 'nan'
        and_words = 'nan'
        and_words2 = 'nan'
        not_words = 'nan'
        not_words2 = 'nan'

        if (str(kw_clean) != 'nan'):
            key_words = [str(x.strip()).lower() for x in kw_clean.split(',')]
            key_words2 = set(w.lower() for w in key_words)

        if (str(aw_clean) != 'nan'):
            and_words = [str(x.strip()).lower() for x in aw_clean.split(',')]
            and_words2 = set(w.lower() for w in and_words)

        if (str(nw_clean) != 'nan'):
            not_words = [str(x.strip()).lower() for x in nw_clean.split(',')]
            not_words2 = set(w.lower() for w in not_words)

        if (str(kw_clean) == 'nan'):
            key_flag = False
        else:
            if set(data_words) & key_words2:
                key_flag = True
            else:
                if (asterix_handler(key_words2, data_words) == 'T'):
                    key_flag = True

        if (str(aw_clean) == 'nan'):
            and_flag = True
        else:
            if set(data_words) & and_words2:
                and_flag = True
            else:
                if (asterix_handler(and_words2, data_words) == 'T'):
                    and_flag = True
        if (str(nw_clean) == 'nan'):
            not_flag = False
        else:
            if set(data_words) & not_words2:
                not_flag = True
            else:
                if (asterix_handler(not_words2, data_words) == 'T'):
                    not_flag = True
        if (key_flag == True and and_flag == True and not_flag == False):
            output.append(str(df_tx['Subtopic'][l]))
            temp = {'slno': [data.iloc[k, 0]], 'text': [data.iloc[k, 1]], 'Category': [df_tx['Subtopic'][l]]}
            mapped_data = pd.concat([mapped_data, pd.DataFrame(temp)])

# output mapped data
mapped_data.to_csv("dataset/mapped_data.csv", index=False)

# Sentiment Mapping and Score
# reading category mapped data
catdata = pd.read_csv("dataset/mapped_data.csv")

# new variable score
catdata['score'] = '';
for s in range(len(catdata)):
    comment = remov_punct(catdata['text'][s])
    sentiment = findscore(comment)
    catdata['score'][s] = sentiment

# adding the ratings
topic = []
rate = []

for i in range(len(df_tx)):
    subtopic = df_tx['Subtopic'][i]
    sumcat = 0
    for j in range(len(catdata)):
        category = catdata['Category'][j]
        str1 = set(category.split(' '))
        str2 = set(subtopic.split(' '))
        sentiment = catdata['score'][j]
        if (str1 == str2):
            sumcat = sumcat + sentiment
    topic.append(subtopic)
    rate.append(sumcat)

# output the sentiment mapped data in cvc
catdata.to_csv("dataset/sentiment_mapped_data.csv", index=False)

print(topic)
print(rate)

# Plotting data in a graph
plt.bar(topic, rate)
pylab.xticks(topic, rotation=90)
plt.title('Score by Category')
plt.xlabel('Category')
plt.ylabel('Score')
plt.show()