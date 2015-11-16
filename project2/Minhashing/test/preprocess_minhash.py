from nltk.corpus import stopwords
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from bs4 import BeautifulSoup
from os import listdir
from nltk import PorterStemmer
import string


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

def find_k_shingles(tf_idf_settings,tf_idf_result,count):
    # http://stackoverflow.com/questions/16078015/
    k_shingles= set()
    scores = zip(tf_idf_settings.get_feature_names(),
                 np.asarray(tf_idf_result.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    id=1
    for item in sorted_scores:
        k_shingles.add(item[0])
        print "adding :",item[0]
        id+=1;
        if id==count:
            break
    return k_shingles

def main():
    parsed_documents={}
    k = int(raw_input("K shingles: 1,2,3,4 : "))
    feature_count = int(raw_input("No of features(words) 500,1000,1500..: "))

    dir_path = "/Users/winfredjames/Desktop/datamining temp/datamining_temp/project2/Minhashing/test"
    doc_id=1
    for file in listdir(dir_path):
        print file
        raw_data = BeautifulSoup(open(dir_path + '/' + file), "html.parser")
        for document in raw_data.findAll("reuters"):
            temp=[]
            if document.find("body") is not None and document.find("topics") is not None:
                parsed_documents[doc_id]=clean_up(document.find("body").string);
                doc_id+=1;

    tf_idf_settings = TfidfVectorizer(ngram_range=(k,k))
    output = tf_idf_settings.fit_transform(parsed_documents.values())

    k_shingles=find_k_shingles(tf_idf_settings,output,feature_count)


    print "gg"
main()

