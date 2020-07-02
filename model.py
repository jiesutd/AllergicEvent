# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2018-05-05 14:05:49
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-02-01 21:00:26

# from __future__ import division
import numpy as np
import csv
import sys
import random
import time
from feature_extractor import FeatureExtractor
import xgboost as xgb
import copy
try:
    import cPickle as pickle
except ImportError:
    import pickle as pickle

from utils import build_alphabet

seed_num = 42
random.seed(seed_num)
np.random.seed(seed_num)


class Classifier:
    def __init__(self, model_name='SVM', use_bigram = False, positive_label=None, clean_text=True):
        print('-'*50)
        print("Building classifier...")
        self.model_name = model_name.upper()
        self.use_bigram = use_bigram
        self.features = FeatureExtractor(self.use_bigram, clean_text)
        self.label2id = {}
        self.id2label = {}
        self.positive_label = positive_label
        self.positive_id = None
        self.clf = None
        print("Model has been built:\n \tName: %s\n\tuse_bigram: %s"%(self.model_name, self.use_bigram))

    def build_label_alphabet(self, label_list, binary=True):
        print('-'*50)
        print("Build label alphabet...")
        self.label2id, self.id2label = build_alphabet(label_list, binary)
        print("\tLabel alphabet size: %s"%len(self.label2id))
        if self.positive_label == None:
            print("\tNot set positive label.")
        else:
            if self.positive_label in self.label2id:
                self.positive_id = self.label2id[self.positive_label]
                print("\tPositive label: %s, positive ID: %s"%(self.positive_label, self.positive_id))
            else:
                print("Error! Positive label does not exist in training data! Exit.")
                exit(1)


    def get_label_id(self, input_label_list):
        if len(self.label2id) == 0:
            print("Error in using Model! Should build_label_alphabet  befort get_label_id!")
            exit(1)
        return np.array([self.label2id[label] for label in input_label_list])

    def generate_train_features(self, train_list):
        return self.features.generate_train_features(train_list)

    def generate_decode_features(self, decode_list):
        return self.features.generate_decode_features(decode_list)

    def train(self, train_features, train_label_array):
        r''' train model from training data
        Args:
            train_features (numpy array, 2 dimension): num_instance x num_features
            train_label_array (numpy array): num_instance, label IDs of train instances
    '''
        print('-'*50)
        print("Train model... training instances: %s"%train_features.shape[0])
        if self.model_name == 'XGBOOST':
            self.clf = xgboost_train(train_features, train_label_array)
        elif self.model_name == 'SVM':
            self.clf = linearSVM_train(train_features,train_label_array)
        elif self.model_name == 'LOGISTIC':
            self.clf = logistic_classifier_train(train_features, train_label_array)
        elif self.model_name == 'RANDOMFOREST':
            self.clf = random_forest_train(train_features, train_label_array)
        else:
            print("Error: no model founded, should be among XGBOOST/SVM/LOGISTIC/RANDOMFOREST, input: ", self.model_name)
            exit(1)

    def decode_prob(self, decode_features):
        r''' train random forest model based on train data
        Args:
            decode_features (list): size[num_instance, feature_num]
        Return:
            decoded label probability (numpy array)
        '''

        if self.model_name == 'XGBOOST':
            ddecode = xgb.DMatrix(decode_features) 
            decode_preds = self.clf.predict(ddecode)
        else:
            decode_preds = self.clf.predict_proba(decode_features)
        return decode_preds

    def decode(self, decode_features):
        r''' train random forest model based on train data
        Args:
            decode_features (list): size[num_instance, feature_num]
        Return:
            decoded labels (numpy array)
        '''
        if self.model_name == 'XGBOOST':
            ddecode = xgb.DMatrix(decode_features) 
            decode_preds = self.clf.predict(ddecode)
        else:
            decode_preds = self.clf.predict(decode_features)
        return decode_preds

    def decode_raw(self, raw_input, decode_prob=False):
        r''' train random forest model based on train data
        Args:
            raw_input (list): size[num_instance], list of strings (sentences/documents)
            decode_prob (boolean): if decode the probability
        Return:
            decoded label or label probability (numpy array)
    '''
        raw_features = self.generate_decode_features(raw_input)
        if decode_prob:
            return self.decode_prob(raw_features)
        else:
            return self.decode(raw_features)



    def save(self,file_dir):
        if self.model_name == "XGBOOST":
            pass
            # new_dict = copy.deepcopy(self.__dict__)
        else:
            f = open(file_dir, 'wb')
            pickle.dump(self.__dict__, f, 2)
            f.close()
        print("Model is saved as file: %s"%file_dir)


    def load(self, file_dir):
        if self.model_name == "XGBOOST":
            pass
        else:
            f = open(file_dir, 'rb')
            tmp_dict = pickle.load(f)
            f.close()
            self.__dict__.update(tmp_dict)
        print("Model has been loaded from file: %s"%file_dir)


def xgboost_train(train_weight, train_label_array, parameters=(100, 20, 0.1)):
    r''' train logistic classifier model based on train data
        Args:
            train_weight (np.array): size[num_instance, input_dimension]
            train_label_array (np.array): size[num_instance], labels
            parameters (tuple): input parameters (max_depth, num_round, eta)
        Return:
            clf:  classifier
    '''
    print("Running XGBoost model...")
    
    dtrain = xgb.DMatrix(train_weight, label=train_label_array)
    label_alphabet_num = np.unique(train_label_array).size
    # print("unique size:",label_alphabet_num)
    # exit(0)
    # ddev = xgb.DMatrix(dev_weight, label=dev_label_array) 
    # dtest = xgb.DMatrix(test_weight) 
    param = {'max_depth':parameters[0], 'eta':parameters[2], 'eval_metric':'merror', 'silent':1, 'objective':'multi:softprob', 'num_class':label_alphabet_num}  # 参数
    clf = xgb.train(param, dtrain, parameters[1])
    return clf



def random_forest_train(train_weight, train_label_array, parameters=(100, 0)):
    r''' train random forest model based on train data
        Args:
            train_weight (np.array): size[num_instance, input_dimension], input x
            train_label_array (np.array): size[num_instance], labels
            parameters (tuple): input parameters (n_estimators, random_state)
        Return:
            clf: rf classifier
    '''
    print("Running random forest Classifier...")
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=parameters[0],  random_state=parameters[1])
    clf.fit(train_weight, train_label_array) 
    return clf



def logistic_classifier_train(train_weight, train_label_array, parameters=(100,)):
    r''' train logistic classifier model based on train data
        Args:
            train_weight (np.array): size[num_instance, input_dimension]
            train_label_array (np.array): size[num_instance], labels
            parameters (tuple): input parameters (max_iter)
        Return:
            clf:  classifier
    '''
    print("Running logistic Classifier...")
    from sklearn.linear_model import SGDClassifier
    clf = SGDClassifier(loss='log', max_iter=parameters[0])
    clf.fit(train_weight, train_label_array) 
    return clf


def linearSVM_train(train_weight, train_label_array, parameters=(100,)):
    r''' train logistic classifier model based on train data
        Args:
            train_weight (np.array): size[num_instance, input_dimension]
            train_label_array (np.array): size[num_instance], labels
            parameters (tuple): input parameters (max_iter)
        Return:
            clf:  classifier
    '''
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import SGDClassifier
    # clf = CalibratedClassifierCV(LinearSVC())
    clf = CalibratedClassifierCV(SGDClassifier(loss='hinge', max_iter=parameters[0]))
    clf.fit(train_weight, train_label_array) 
    return clf








if __name__ == '__main__':

    print("Model")







