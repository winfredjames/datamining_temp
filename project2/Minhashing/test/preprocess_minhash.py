from __future__ import division
from nltk.corpus import stopwords
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from bs4 import BeautifulSoup
from os import listdir
from nltk import PorterStemmer
import string
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import random
import sys


def clean_up(text):
    if text is not None:
        pattern = r'[\x02\x0F\x16\x1D\x1F]|\x03(\d{,2}(,\d{,2})?)?';
        text = re.sub(pattern, ' ', text);
        text = re.sub('[a-z]*&#.*;', ' ', text);
        text = re.sub('\d+', ' ', text)
        text = re.sub('<[^>]*>', ' ', text)
        # remove punctuations and convert to lower case
        text = "".join(word.lower() for word in text if word not in string.punctuation)
        # remove stop words
        stop_words = stopwords.words("english")
        task_related_common_words = ["reuter", "said", "told", "mln"]
        stop_words = stop_words + task_related_common_words
        text = " ".join(word for word in text.split() if word not in stop_words)
        # stem words
        stemmer = PorterStemmer()
        text = " ".join(stemmer.stem(word) for word in text.split())

    return str(text.encode('utf-8'))


def find_k_shingles(tf_idf_settings, tf_idf_result, count):
    # http://stackoverflow.com/questions/16078015/
    k_shingles = {}
    scores = zip(tf_idf_settings.get_feature_names(),
                 np.asarray(tf_idf_result.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    id = 0
    for item in sorted_scores:
        k_shingles[str(item[0])] = id
        print "adding :", item[0]
        if id == count:
            break
        id += 1
    return k_shingles


def main():
    parsed_documents = []
    # inputs
    k = int(raw_input("K shingles: 1,2,3,4 : "))
    feature_count = int(raw_input("No of features(words) 500,1000,1500..: "))
    no_hashes = int(raw_input("Enter the no. of hashes "))

    dir_path = "/Users/winfredjames/Desktop/datamining temp/datamining_temp/project2/Minhashing/test"

    for current_file in listdir(dir_path):
        print current_file
        raw_data = BeautifulSoup(open(dir_path + '/' + current_file), "html.parser")
        for document in raw_data.findAll("reuters"):
            if document.find("body") is not None:
                parsed_documents.append(clean_up(document.find("body").string))

    tf_idf_settings = TfidfVectorizer(ngram_range=(k, k))
    output = tf_idf_settings.fit_transform(parsed_documents)

    k_shingles = find_k_shingles(tf_idf_settings, output, feature_count)

    #output is binary matrix (shingles X no.of docs)
    output = np.zeros((k_shingles.__len__(), parsed_documents.__len__()), dtype=np.int)

    doc_id = 0
    for doc in parsed_documents:
        n_grams = ngrams(word_tokenize(doc), k)
        list = [' '.join(grams) for grams in n_grams]
        for each_gram in list:
            if k_shingles.__contains__(each_gram):
                output[k_shingles.get(each_gram)][doc_id] = 1
        doc_id += 1

    true_jaccard_similarity = np.zeros((parsed_documents.__len__(), parsed_documents.__len__()), dtype=np.float)

    for i in range(0, parsed_documents.__len__() - 1):
        for j in range(i + 1, parsed_documents.__len__()):
            equal_ones = 0
            any_ones = 0
            jac_sim = 0
            for l in range(0, k_shingles.__len__()):
                if output[l][i] == 1 and output[l][j] == 1:
                    equal_ones += 1
                elif output[l][i] == 1 or output[l][j] == 1:
                    any_ones += 1
            jac_sim = equal_ones / (equal_ones + any_ones)
            true_jaccard_similarity[i][j] = jac_sim


    a_const = random.sample(range(1, pow(2,18)), no_hashes)
    b_const = random.sample(range(1, pow(2,18)), no_hashes)
    large_prime =  77747

    signature_mat = np.empty((no_hashes, parsed_documents.__len__()),dtype=np.int)
    signature_mat.fill(pow(2,18))

    for i in range(0, output.__len__()):
        for j in range(0,no_hashes):
            h = ((a_const[j]*i + b_const[j]) % large_prime) % output.__len__()
            for k in range(0,parsed_documents.__len__()):
               if output[i][k]==1:
                if k < signature_mat[j][k]:
                    signature_mat[j][k] = h

    jaccard_similarity = np.zeros((parsed_documents.__len__(), parsed_documents.__len__()), dtype=np.float)

    for i in range(0, parsed_documents.__len__()-1):
        for j in range(i+1, parsed_documents.__len__()):
            count = 0
            for m in range(0, no_hashes):
                if signature_mat[m][i] == signature_mat[m][j]:
                    count+=1
            jaccard_similarity[i][j]=count/no_hashes

    print "end of program for break point purpose"


main()
