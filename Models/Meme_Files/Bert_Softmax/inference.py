import torch
import numpy as np
import pandas as pd

import transformers
from transformers import AutoTokenizer

from Models.Bert_Softmax.dataProcessing import Data_Preprocessing
from Models.Bert_Softmax.bertModel import Propaganda_Detection
from Models.Bert_Softmax.evaluation import *

class Inferencer:
    def __init__(self, paths, checkpoint_tokenizer, checkpoint_model, hyper_params, techniques):
        self.paths = paths
        self.checkpoint_tokenizer = checkpoint_tokenizer
        self.checkpoint_model = checkpoint_model
        self.hyper_params = hyper_params
        self.techniques = techniques

        self.model = None
        self.tokenizer = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert self.device == torch.device('cuda')

    def get_model_and_tokenizer(self,):

        self.model = Propaganda_Detection(checkpoint_model=self.checkpoint_model, num_tags=len(self.techniques)+1)
        self.model.load_state_dict(torch.load(self.paths['Model_Files'] + 'model_bert.pt'))
        self.tokenizer = AutoTokenizer.from_pretrained(self.paths['Model_Files'] +'tokenizer/')

        self.model = self.model.to(self.device)
        # next(self.model.parameters()).is_cuda
        # next(self.model.parameters()).device
        self.model.eval()

    def get_test_data(self,):
        df_test = pd.read_csv(self.paths["Meme_Data_Test"])
        df_test = Data_Preprocessing.eval_dataframe(df_test)
        return df_test

    def get_predictions(self, df):

        gold_labels, pred_labels = [], []
        classes = list(self.techniques.keys())
        self.techniques['O'] = 0
        id2techniques = {v: k for k, v in self.techniques.items()}


        for i, f in df.iterrows():
            text = f['Text']
            technique = f['Technique']
            tokenized_sentence = self.tokenizer.encode(text)
            # print(f"CLS token: {self.tokenizer.cls_token} | CLS token id: {self.tokenizer.cls_token_id}")
            # print(f"SEP token: {self.tokenizer.sep_token} | SEP token id: {self.tokenizer.sep_token_id}")
            # print(f"Vocab Size: {self.tokenizer.vocab}")
            tokenized_sentence = tokenized_sentence[1:-1]
            input_ids = torch.tensor([tokenized_sentence])
            input_ids = input_ids.to(self.device)
            # input_ids.get_device() 0 is CUDA -1 is CPU

            with torch.no_grad(): 
                output = self.model(input_ids)
                label_indices = np.argmax(output[0].to('cpu').numpy(),axis=1)
            

            pred_tags = [id2techniques[l] for l in label_indices]
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
            
            pred_tags = set(pred_tags)
            if 'O' in pred_tags: pred_tags.remove('O')
            technique = set(technique)

            gold_labels.append(technique)
            pred_labels.append(pred_tags)

        return gold_labels, pred_labels, classes
    
    def run(self,):
        df_test = self.get_test_data()
        self.get_model_and_tokenizer()
        gold_labels, pred_labels, classes = self.get_predictions(df_test)
        macro_f1, micro_f1 = get_MultilabelBinarizer(gold_labels, pred_labels, classes)
        return macro_f1, micro_f1


