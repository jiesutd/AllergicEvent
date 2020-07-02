# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2018-05-05 14:05:49
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-02-01 17:16:07

from __future__ import division
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from utils import unigram2bigram, clinic_text_list_processing

class FeatureExtractor:
    def __init__(self, bigram=False, clean_text=True):
        self.use_bigram = bigram
        self.unigram_vectorizer = CountVectorizer()
        self.unigram_tfidftransformer = TfidfTransformer()
        self.clean_text = clean_text
        if self.use_bigram:
            self.bigram_vectorizer = CountVectorizer()
            self.bigram_tfidftransformer = TfidfTransformer()

    def generate_train_features(self, train_list):
        r''' load train dataset input x, build the features based on the train dataset and return the train features
        Args:
            train_list (list): size[num_instance], list of input sentences/documents
        Return:
            train_tfidf: tfidf features
        '''
        if self.clean_text:
            train_list = clinic_text_list_processing(train_list)

        train_vec = self.unigram_vectorizer.fit_transform(train_list)
        train_tfidf = self.unigram_tfidftransformer.fit_transform(train_vec).toarray()
        if self.use_bigram:
            train_bigram = unigram2bigram(train_list) ## list of list
            bigram_vec = self.unigram_vectorizer.fit_transform(train_bigram)
            bigram_tfidf = self.unigram_tfidftransformer.fit_transform(bigram_vec).toarray()
            train_tfidf = np.concatenate((train_tfidf, bigram_tfidf), axis=1)
        return train_tfidf

    def generate_decode_features(self, decode_list):
        r''' generate the features of dev or test or raw dataset
        Args:
            decode_list (list): size[num_instance], list of input sentences/documents
        Return:
            decode_tfidf (numpy array): tfidf features
        '''
        if self.clean_text:
            decode_list = clinic_text_list_processing(decode_list)
        decode_vec = self.unigram_vectorizer.transform(decode_list)
        decode_tfidf = self.unigram_tfidftransformer.transform(decode_vec).toarray()
        if self.use_bigram:
            decode_bigram = unigram2bigram(decode_list) ## list of list
            decode_vec = self.unigram_vectorizer.transform(decode_bigram)
            bigram_tfidf = self.unigram_tfidftransformer.transform(decode_vec).toarray()
            decode_tfidf = np.concatenate((decode_tfidf, bigram_tfidf), axis=1)
        return decode_tfidf






if __name__ == '__main__':

    print("FeatureExtractor")







