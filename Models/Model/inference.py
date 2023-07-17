import os
import re
import json
import string
import shutil
import torch
import numpy as np
import pandas as pd

import transformers
from transformers import AutoTokenizer

from dataPreparation import Dataset_Preparation
from Model.model import Propaganda_Detection
from Model.evaluation import *

class Inferencer:
    def __init__(self, paths, checkpoint_tokenizer, checkpoint_model, hyper_params, techniques, logger_results):
        self.paths = paths
        self.checkpoint_tokenizer = checkpoint_tokenizer
        self.checkpoint_model = checkpoint_model
        self.hyper_params = hyper_params
        self.techniques = techniques
        self.logger_results = logger_results

        self.model = None
        self.tokenizer = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert self.device == torch.device('cuda')

    def get_model_and_tokenizer(self,):

        self.model = Propaganda_Detection(checkpoint_model=self.checkpoint_model, num_tags=len(self.techniques), device=self.device, hyper_params=self.hyper_params)
        # path = os.getcwd() + '/Switch_Files/'
        path = os.getcwd() + '/Model_Files/' + self.hyper_params['model_run'] + '/'
        checkpoint = torch.load(path + self.hyper_params['model_run'] + '.pt')
        self.model.load_state_dict(checkpoint['model'])
        self.tokenizer = AutoTokenizer.from_pretrained(path + self.hyper_params['model_run'] +'_tokenizer/')

        self.model = self.model.to(self.device)
        self.model.eval()
        print('##################################################')



    def get_predictions(self, df):

        # print(f"CLS token: {self.tokenizer.cls_token} | CLS token id: {self.tokenizer.cls_token_id}")
        # print(f"SEP token: {self.tokenizer.sep_token} | SEP token id: {self.tokenizer.sep_token_id}")
        # print(f"Vocab Size: {self.tokenizer.vocab}")

        classes = list(self.techniques.keys())
        id2techniques = {v: k for k, v in self.techniques.items()}

        predicted_techniques, original_techniques, gold_labels_list, pred_labels_list = [], [], [], []
        for i, f in df.iterrows():
            text = f['text']
            technique = f['technique']
            tokenized_sentence = self.tokenizer.encode(text)
            tokenized_sentence = tokenized_sentence[1:-1]
            input_ids = torch.tensor([tokenized_sentence])
            input_ids = input_ids.to(self.device)
            # input_ids.get_device() 0 is CUDA -1 is CPU

            with torch.no_grad(): 
                output = self.model(input_ids)
                labels = output[0].to('cpu').numpy()
            

            label_indices = np.argwhere(labels.reshape(-1) >= 0.05).reshape(-1)
            pred_labels  =  np.where(labels >= 0.05, 1, 0)
            pred_tags = [id2techniques[l] for l in label_indices]

            gold_indices = [self.techniques[l] for l in technique]
            gold_labels = np.zeros((len(self.techniques)))
            gold_labels[gold_indices] = 1
            gold_labels =  gold_labels.reshape(1,-1).astype(int)

            predicted_techniques.append(pred_tags)
            original_techniques.append(technique)
            gold_labels_list.append(gold_labels)
            pred_labels_list.append(pred_labels)


        for i, l in enumerate(pred_labels_list):
            assert len(l) == len(gold_labels_list[i])

        return predicted_techniques, original_techniques, gold_labels_list, pred_labels_list

    def get_evaluations(self, gold_labels_list, pred_labels_list):

        gold_labels_list = np.array(gold_labels_list)
        pred_labels_list = np.array(pred_labels_list)
        gold_labels_list = np.squeeze(gold_labels_list, axis=1)
        pred_labels_list = np.squeeze(pred_labels_list, axis=1)

        labels_ = list(self.techniques.keys())

        accuracy_score = get_accuracy_score(gold_labels_list, pred_labels_list)
        hamming_score = get_hamming_score(gold_labels_list, pred_labels_list)
        exact_match_ratio = get_exact_match_ratio(gold_labels_list, pred_labels_list)
        print('     Validation Accuracy Score: {}'.format(accuracy_score))
        print('     Validation Hamming Score: {}'.format(hamming_score))
        print('     Validation Exact Match Ratio: {}'.format(exact_match_ratio))
        classificationReport = get_classification_report(gold_labels_list, pred_labels_list, labels_)
        print('Classification Report\n')
        print(classificationReport)

        return classificationReport, hamming_score, exact_match_ratio, accuracy_score
    
    def run(self,):
        df_test = self.hyper_params['df_test']
        self.get_model_and_tokenizer()
        predicted_techniques, original_techniques, gold_labels_list, pred_labels_list = self.get_predictions(df_test)
        classificationReport, hamming_score, exact_match_ratio, accuracy_score = self.get_evaluations(gold_labels_list, pred_labels_list)
        if not self.hyper_params["debugging"]:
            self.logger_results.info('Validation Hamming Score: {}  |  Validation Exact Match Ratio: {} |  Validation Accuracy Score: {}'.format(hamming_score, exact_match_ratio, accuracy_score))
            self.logger_results.info('Classification Report:')
            self.logger_results.info('\n{}'.format(classificationReport))
            breakpoint()
            foldername = self.hyper_params['model_run']
            path = self.paths['Model_Files'] + foldername
            shutil.copy2(self.hyper_params['log_file'], path)
        return 
