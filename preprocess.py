import re
import string
import time
import csv
import math

from os import listdir
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk import PorterStemmer

# clean function is used for removing uni-codes, color codes, punctuations, stopwords and stemming
# if you find any other unwanted codes in the body use this function to remove it.
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


def create_doc_term_matrix(term_lexicon, documents):
        document_term_matrix = []
        for document in documents:
            feature_vector = []
            feature_vector.append(document['TOPICS'])
            feature_vector.append(document['ALL_TOPICS'])
            feature_vector.append(document['PLACES'])
            for term in term_lexicon:
                if term in document['doc_term_freq']:
                    feature_vector.append(document['doc_term_freq'][term])
                else:
                    feature_vector.append(0)

            document_term_matrix.append(feature_vector)
        return document_term_matrix


def create_doc_term_matrix_tf_idf(term_lexicon, documents):
        document_term_matrix = []
        for document in documents:
            feature_vector = []
            feature_vector.append(document['TOPICS'])
            feature_vector.append(document['ALL_TOPICS'])
            feature_vector.append(document['PLACES'])
            for term in term_lexicon:
                if term in document['doc_term_freq']:
                    feature_vector.append(document['doc_term_freq'][term] * term_lexicon[term]['IDF'])
                else:
                    feature_vector.append(0)

            document_term_matrix.append(feature_vector)
        return document_term_matrix


def write_to_file_dtm(document_term_matrix, term_lexicon, file_name):
    with open(file_name, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar=',', quoting=csv.QUOTE_MINIMAL)

        headers = []
        headers.append('TOPICS')
        headers.append('ALL_TOPICS')
        headers.append('PLACES')
        for term in term_lexicon:
            headers.append(term)

        writer.writerow([header for header in headers])
        for document_feature_vector in document_term_matrix:
            writer.writerow([weight for weight in document_feature_vector])


def write_to_file_postings(posting, file_name):
    with open(file_name, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['ID', 'DOC_ID', 'FREQ'])
        for key in posting:
            writer.writerow([key, posting[key]['DOC_ID'], posting[key]['FREQ']])


def write_to_file_dictionary(term_lexicon, file_name):
    with open(file_name, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar=',', quoting=csv.QUOTE_MINIMAL)

        writer.writerow(['TERM', '#DOCS', 'FREQ', 'IDF', 'POSTING_ID'])
        for key in term_lexicon:
            writer.writerow(
                [key, term_lexicon[key]['NO_OF_DOCS'], term_lexicon[key]['FREQ'], term_lexicon[key]['IDF'],
                 term_lexicon[key]['POSTING_ID']])


class Parser:
    parsed_cleaned_up_documents = []
    skipped_documents = []

    def parse(self, raw_data, doc_id):

        for document in raw_data.findAll("reuters"):
            topics = document.find("topics")
            places = document.find("places")
            body = document.find("body")
            if (topics is None or
                        places is None or
                        body is None or topics.findAll("d") == []):
                doc_dict = {}
                # print "Skipping ", topics.text
                if topics is not None:
                    doc_dict['TOPICS'] = topics.text
                if places is not None:
                    doc_dict['PLACES'] = places.text
                if body is not None:
                    doc_dict['BODY'] = clean_up(body.string)

                self.skipped_documents.append(doc_dict)
            else:
                # doc_dict = {}
                # doc_dict['ID'] = doc_id
                # doc_dict['TOPICS'] = topics
                # doc_dict['PLACES'] = places.text
                # doc_dict['BODY'] = clean_up(body.string)
                # self.parsed_cleaned_up_documents.append(doc_dict)
                # doc_id += 1
                # print topics.text
                # if there are multiple topics, duplicating the document for each individual topic
                body = clean_up(body.string)
                for topic in topics.findAll("d"):
                    doc_dict = {}
                    doc_dict['ID'] = doc_id
                    doc_dict['ALL_TOPICS'] = topics
                    doc_dict['TOPICS'] = topic.text
                    doc_dict['PLACES'] = places.text
                    doc_dict['BODY'] = body
                    self.parsed_cleaned_up_documents.append(doc_dict)
                    doc_id += 1





        return doc_id


class PreProcessor:
    def __init__(self):
        self.document_term_matrix_sparse = []
        self.document_term_matrix_idf =[]
        self.parser = Parser()
        self.term_lexicon_dict = {}
        self.posting = {}
        self.tf_idf_filtered_term_lexicon = {}
        self.sparse_term_filtered_lexicon_dict = {}

    def build_lexicon_dict(self):
        lexicon_dict = {}
        for doc in self.parser.parsed_cleaned_up_documents:
            if doc is not None:
                for word in doc['BODY'].split():
                    lexicon_dict[word] = {'NO_OF_DOCS': 0, 'FREQ': 0, 'IDF': 0}

        return lexicon_dict

    def create_doc_term_freq(self):
        for document in self.parser.parsed_cleaned_up_documents:
            doc_term_freq = {}
            word_list = document['BODY'].split()

            for word in set(word_list):
                doc_term_freq[word] = word_list.count(word)

            document['doc_term_freq'] = doc_term_freq

    def build_postings(self):
        posting = {}
        posting_id = 0
        for word in self.term_lexicon_dict:
            self.term_lexicon_dict[word]['POSTING_ID'] = posting_id
            for document in self.parser.parsed_cleaned_up_documents:
                posting_doc_freq = {}
                if word in document['doc_term_freq']:
                    self.term_lexicon_dict[word]['NO_OF_DOCS'] += 1
                    self.term_lexicon_dict[word]['FREQ'] += document['doc_term_freq'][word]
                    posting_doc_freq['DOC_ID'] = document['ID']
                    posting_doc_freq['FREQ'] = document['doc_term_freq'][word]
                    posting[posting_id] = posting_doc_freq
                    posting_id += 1
        return posting

    def calc_idf(self):
        for word in self.term_lexicon_dict:
            self.term_lexicon_dict[word]['IDF'] = math.log(1 + self.parser.parsed_cleaned_up_documents.__len__()) \
                                                  / self.term_lexicon_dict[word]['NO_OF_DOCS']

    def remove_sparse_terms(self):
        sparse_term_lexicon_dict = {}

        for word in self.term_lexicon_dict:
            if (self.term_lexicon_dict[word][
                'NO_OF_DOCS']) / float(self.parser.parsed_cleaned_up_documents.__len__()) > 0.01:
                sparse_term_lexicon_dict[word] = self.term_lexicon_dict[word]

        return sparse_term_lexicon_dict

    def filter_based_on_tf_idf(self):
        tf_idf_filtered_term_lexicon = {}
        for word in self.term_lexicon_dict:
            if self.term_lexicon_dict[word]['FREQ'] * self.term_lexicon_dict[word]['IDF'] > 20.0:
                tf_idf_filtered_term_lexicon[word] = self.term_lexicon_dict[word]
        return tf_idf_filtered_term_lexicon

    def start_processing(self):
        begin_total_time = time.time()
        print "*************** Begin Processing ********************"
        # read all the files from the directory
        dir_path = "/Users/winfredjames/Desktop/reuters/"

        print "*************** Parsing Files ********************"
        start = time.time()
        doc_id = 0
        for file in listdir(dir_path):
            print 'Parse file ' + dir_path + '/' + file
            raw_file_data = BeautifulSoup(open(dir_path + '/' + file), "html.parser")
            doc_id = self.parser.parse(raw_file_data,doc_id)

        print "Time taken to parse and clean up all documents : ", float("{0:.2f}".format(time.time() - start)), "s"

        total_no_documents = self.parser.parsed_cleaned_up_documents.__len__()
        print "No. of documents correctly parsed : ", total_no_documents
        print "No. of documents skipped : ", self.parser.skipped_documents.__len__()

        print "*************** Create Document Term Freq for each document ********************"
        start = time.time()
        self.create_doc_term_freq()
        print "Time taken to create term freq for each document : ", float("{0:.2f}".format(time.time() - start)), "s"

        print "*************** Build Inverted Index ********************"
        start = time.time()
        self.term_lexicon_dict = self.build_lexicon_dict()
        self.posting = self.build_postings()
        print "Time taken to build inverted index : ", float("{0:.2f}".format(time.time() - start)), "s"

        print "No. of distinct terms in inverted index : ", self.term_lexicon_dict.__len__()

        print "*************** Calculate IDF ******************"
        start = time.time()
        self.calc_idf()
        print "Time taken to calculate IDF: ", float("{0:.2f}".format(time.time() - start)), "s"

        # print "*************** Calculate TF-IDF sum ******************"
        # start = time.time()
        # self.calc_tf_idf_sum()
        # print "Time taken to calculate TF-IDF sum: ", float("{0:.2f}".format(time.time() - start)), "s"

        print "*************** Write Inverted Index to file ******************"
        start = time.time()
        write_to_file_dictionary(self.term_lexicon_dict, 'complete_dictionary.csv')
        write_to_file_postings(self.posting, 'complete_posting.csv')
        print "Time taken to write inverted index to file: ", float("{0:.2f}".format(time.time() - start)), "s"

        print "*************** Remove Sparse Terms and write to file ******************"
        # filter out sparse terms and write to file
        start = time.time()
        self.sparse_term_filtered_lexicon_dict = self.remove_sparse_terms()
        write_to_file_dictionary(self.sparse_term_filtered_lexicon_dict, 'sparse_term_filtered_dictionary.csv')
        print "Time taken to filter out sparse terms and write to file: ", float(
            "{0:.2f}".format(time.time() - start)), "s"

        print "*************** Filter based on TF-IDF weight and write to file ******************"
        # filter based on tf-idf weight and write to file
        start = time.time()
        self.tf_idf_filtered_term_lexicon = self.filter_based_on_tf_idf()
        write_to_file_dictionary(self.tf_idf_filtered_term_lexicon, 'tf_idf_filtered_dictionary.csv')
        print "Time taken to filter out terms based on IDF and write to file: ", float(
            "{0:.2f}".format(time.time() - start)), "s"

        print "*************** Create Document Term Freq for TF-IDF based feature vector and write to file" \
              " ******************"
        # create document_term_matrix from tf_idf_filtered_lexicon_dict
        start = time.time()
        self.document_term_matrix_idf = create_doc_term_matrix_tf_idf(self.tf_idf_filtered_term_lexicon,self.parser.parsed_cleaned_up_documents)
        # write DTM to file
        write_to_file_dtm(self.document_term_matrix_idf, self.tf_idf_filtered_term_lexicon, "dtm_idf.csv")
        print "Time taken to create DTM based on IDF weights and write to file : ", float(
            "{0:.2f}".format(time.time() - start)), "s"

        print "*************** Create Document Term Freq for Sparse Term removal based feature vector " \
              "and write to file ******************"
        # create document_term_matrix from tf_idf_filtered_lexicon_dict
        start = time.time()
        self.document_term_matrix_sparse = create_doc_term_matrix(self.sparse_term_filtered_lexicon_dict,self.parser.parsed_cleaned_up_documents)
        # write DTM - sparse term matrix to file
        write_to_file_dtm(self.document_term_matrix_sparse, self.sparse_term_filtered_lexicon_dict, "dtm_sparse_term_removed.csv")
        print "Time taken to create DTM based on sparse terms removed and write to file : ", float(
            "{0:.2f}".format(time.time() - start)), "s"

        print "Total time taken for processing : ", time.time() - begin_total_time

    # def calc_tf_idf_sum(self):
    #     for term in self.term_lexicon_dict:
    #         posting_id =self.term_lexicon_dict[term]['POSTING_ID']
    #         no_of_docs = self.term_lexicon_dict[term]['NO_OF_DOCS']
    #         sum_tf_idf = 0
    #         for posting_id  in range(posting_id,posting_id +no_of_docs):
    #             sum_tf_idf += self.posting[posting_id]['FREQ'] * self.term_lexicon_dict[term]['IDF']
    #
    #         self.term_lexicon_dict[term]['SUM_TF_IDF'] = sum_tf_idf

pre_processor = PreProcessor()
pre_processor.start_processing()


