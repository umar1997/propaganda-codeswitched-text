import re
import json
import string 
import pandas as pd
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras_preprocessing.sequence import pad_sequences

class Dataset_Preparation:
    def __init__(self, paths, tokenizer, hyper_params, techniques):
        self.paths = paths
        self.tokenizer = tokenizer
        self.hyper_params = hyper_params
        self.techniques = techniques

    @staticmethod
    def read_techniques(filename):
        """
        Read the techniques json file into a dictionary
        """
        with open(filename, "r") as fp:
            techniques = json.load(fp)
        
        return techniques
    
    @staticmethod
    def clean_text(text):
        """
        Clean the text of data
        """
        punctuation_list  = string.punctuation  # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
        acceptable_list = "?\"\'().,!%"
        remove_list = list(filter(lambda punctuation_list: punctuation_list[0] not in acceptable_list, punctuation_list))
        remove_list.append('•')

        text = text.replace('\n', ' ')
        has_any_remove = any([char in remove_list for char in text])
        if has_any_remove:
            for r in remove_list:
                if r in text:
                    text = text.replace(r, ' ')
        has_any_accept = any([char in acceptable_list for char in text])
        if has_any_accept:
            for a in acceptable_list:
                if a in text and a not in "\"\'":
                    text = re.sub(re.escape(a) + r"{2,}", a,text)
                    text = text.replace(a, a+' ')
        text = re.sub(r' {2,}', ' ',text)
        text = text.strip()
        return text

    @staticmethod
    def read_json_files_to_df(json_file, hyper_params, split):
        """
        Read file from json format and convert into pandas datatframe.
        -> There is no need for the text fragment as we don't use it later on for the model
        * MEMES trained on Meme Training Data, Validated on CS Validation Data
        * ENGLISH trained on CS Training Translated Data, Validated on CS Validation Data
        * CS trained on CS Training Data, Validated on CS Validation Data
        """
        with open(json_file, 'r') as f:
                data = json.loads(f.read())

        if (hyper_params['domain_type'] == 'MEMES') and (split=='Training'):
            data_dict = dict()
            for i, example in enumerate(data):
                text = Dataset_Preparation.clean_text(example['text'])
                list_labels = example['labels']

                data_dict[i] = {'text' : text, 'technique' : [], 'text_fragment' : []}
                for label in list_labels:
                    technique = label['technique']
                    fragment = Dataset_Preparation.clean_text(label['text_fragment'])
                    # if fragment not in text:
                    #     raise Exception('Fragment cleaned different from text cleaned')
                    data_dict[i]['technique'].append(technique)
                    data_dict[i]['text_fragment'].append(fragment)
                    
                    assert len(data_dict[i]['technique']) == len (data_dict[i]['text_fragment'])

            data_df = pd.DataFrame(data_dict).transpose()

        elif (hyper_params['domain_type'] == 'ENGLISH') and (split=='Training'):
            data_dict = dict()
            for i, (_, example) in enumerate(data.items()):
                text = Dataset_Preparation.clean_text(example['translation'])
                list_labels = example['labels']

                data_dict[i] = {'text' : text, 'technique' : [], 'text_fragment' : []}
                for label in list_labels:
                    technique = label['technique']
                    fragment = Dataset_Preparation.clean_text(label['text_fragment'])
                    data_dict[i]['technique'].append(technique)
                    data_dict[i]['text_fragment'].append(fragment)
                    
                    assert len(data_dict[i]['technique']) == len (data_dict[i]['text_fragment'])

            data_df = pd.DataFrame(data_dict).transpose()

        # ALL VALIDATIONS AND ALL TESTS AND TRAINING CS    
        else:
            data_dict = dict()
            for i, (_, example) in enumerate(data.items()):
                text = Dataset_Preparation.clean_text(example['text'])
                list_labels = example['labels']

                data_dict[i] = {'text' : text, 'technique' : [], 'text_fragment' : []}
                for label in list_labels:
                    technique = label['technique']
                    fragment = Dataset_Preparation.clean_text(label['text_fragment'])
                    if fragment not in text:
                        raise Exception('Fragment cleaned different from text cleaned')
                    data_dict[i]['technique'].append(technique)
                    data_dict[i]['text_fragment'].append(fragment)
                    
                    assert len(data_dict[i]['technique']) == len (data_dict[i]['text_fragment'])

            data_df = pd.DataFrame(data_dict).transpose()
        return data_df


    def tokenize_preserve(self, fragments, techniques, text):
        assert len(fragments) == len(techniques)
        tokenized_words = self.tokenizer.tokenize(text)
        indices = [self.techniques[t] for t in techniques]
        labels = np.zeros((len(self.techniques)))
        labels[indices] = 1
        labels = labels.tolist()

        return tokenized_words, labels
            


    def get_text_and_labels(self, df):
        tokenized_words_list, labels_list = [], []
        for i, f in df.iterrows():
            tokenized_words, labels = self.tokenize_preserve(f['text_fragment'], f['technique'], f['text'])
            tokenized_words_list.append(tokenized_words)
            labels_list.append(labels)

        assert len(tokenized_words_list) == len(labels_list)
        for l in labels_list:
            assert len(l) == len(self.techniques)
        return tokenized_words_list, labels_list

    def get_encoded_data(self,tokenized_words_list, labels_list):
        # The reason wecan't keep padding token as self.tokenizer.pad_token_id whose value is 0 is because then our tags or labels will have
        # ['PAD']: 0, and when we are doing the attention mask we are making sure all ['PAD'] have an attention mask of 0
        # Attention masks allow us to send a batch into the transformer even when the examples in the batch have varying lengths. 
        # We do this by padding all sequences to the same length, then using the “attention_mask” tensor to identify which tokens are padding
        # So it is not included in the num_tags for our model classes and the NLL looks at 0 to num_tags-1 classes so we need the 0 class to be a class the model predicts

        # pad_token_id = -100
        # self.techniques[self.tokenizer.pad_token] = pad_token_id
        # id2techniques = {v: k for k, v in self.techniques.items()}
        # cls = [self.tokenizer.cls_token_id]
        # sep = [self.tokenizer.sep_token_id]
        input_ids = pad_sequences(
                            [self.tokenizer.convert_tokens_to_ids(tokenized_txt) for tokenized_txt in tokenized_words_list], # converts tokens to ids
                            maxlen= self.hyper_params['max_seq_length'], dtype='long',value=0.0,
                            truncating='post',padding='post')
        tags = labels_list
        # tags = pad_sequences(
        #                 labels_list, # Gets corresponding tag_id
        #                 maxlen= self.hyper_params['max_seq_length'], dtype='long', value= pad_token_id,
        #                 truncating='post',padding='post')

        attention_masks = [[float(i !=0.0) for i in ii]for ii in input_ids] # Float(True) = 1.0 for attention for only non-padded inputs

        assert len(input_ids) == len(attention_masks) # == len(tags)
        for i in range(len(input_ids)):
            assert len(input_ids[i]) == len(attention_masks[i]) # == len(tags[i])
        for t in tags:
            assert len(t) == len(self.techniques)

        return input_ids, tags, attention_masks
    def convert_to_tensors(self,input_ids, tags, attention_masks):

        input_ids, tag, masks = torch.tensor(input_ids), torch.tensor(tags), torch.tensor(attention_masks)
        return input_ids, tag, masks

    def data_loader(self,train_input_ids, train_tag, train_masks, val_input_ids, val_tag, val_masks):


        train_data = TensorDataset(train_input_ids, train_masks, train_tag)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.hyper_params['training_batch_size'])

        valid_data = TensorDataset(val_input_ids, val_masks, val_tag)
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=self.hyper_params['validation_batch_size'])

        return train_dataloader, valid_dataloader 

    def run(self,):

        train_df = self.hyper_params['df_train']
        train_tokenized_words_list, train_labels_list = self.get_text_and_labels(train_df)
        train_input_ids, train_tags, train_attention_masks = self.get_encoded_data(train_tokenized_words_list, train_labels_list)
        train_input_ids, train_tag, train_masks = self.convert_to_tensors(train_input_ids, train_tags, train_attention_masks)

        val_df = self.hyper_params['df_val']
        val_tokenized_words_list, val_labels_list = self.get_text_and_labels(val_df)
        val_input_ids, val_tags, val_attention_masks = self.get_encoded_data(val_tokenized_words_list, val_labels_list)
        val_input_ids, val_tag, val_masks = self.convert_to_tensors(val_input_ids, val_tags, val_attention_masks)

        train_dataloader, valid_dataloader = self.data_loader(train_input_ids, train_tag, train_masks, val_input_ids, val_tag, val_masks)

        print('-'*30)
        print('DATA INFORMATION')
        print('Training Batch Size: {}'.format(self.hyper_params['training_batch_size']))
        print('Training DataLoader: {}'.format(len(train_dataloader)))
        print()
        print('Validation Batch Size: {}'.format(self.hyper_params['validation_batch_size']))
        print('Validation DataLoader: {}'.format(len(valid_dataloader)))
        print('-'*30)


        return train_dataloader, valid_dataloader

if __name__ == '__main__':

    import transformers
    from transformers import AutoTokenizer
    checkpoint_tokenizer = 'Aimlab/xlm-roberta-roman-urdu-finetuned'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_tokenizer, do_lower_case = False)


    paths = {
            "Techniques":"./techniques.json",
            "Log_Folder":"./Log_Files/",
            "Model_Files":"./Model_Files/",
            "Training_Data": "./Data_Files/Splits/train_split.json",
            "Validation_Data": "./Data_Files/Splits/val_split.json",
            "Meme_Training_Data": "./Data_Files/Meme_Data_Splits/training_set_.json",
    }

    hyper_params = {
        'training_batch_size' : 16,
        'validation_batch_size': 16,
        'max_seq_length' : 256,
        'model_run' :  'BERT'
    }

    dataPrep = Dataset_Preparation(paths, tokenizer, hyper_params)
    dataPrep.run()