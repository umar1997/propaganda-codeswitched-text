import json
import pandas as pd

from keras_preprocessing.sequence import pad_sequences

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


class Data_Preprocessing:
    def __init__(self, paths, tokenizer, hyper_params):
        self.paths = paths
        self.files = [self.paths["Meme_Data_Train"], self.paths["Meme_Data_Val"], self.paths["Meme_Data_Test"]]
        self.techniques = self.read_techniques(self.paths["Techniques"])
        self.tokenizer = tokenizer
        self.hyper_params = hyper_params

    @staticmethod
    def read_techniques(filename):
        """
        Read the techniques json file into a dictionary
        """
        with open(filename, "r") as fp:
            techniques = json.load(fp)
        
        return techniques

    def read_csv_files(self,):
        """
        Read all csv files into a list of dataframes
        """
        dataframes_list = []
        for f in self.files:
            dataframes_list.append(pd.read_csv(f))

        return dataframes_list
    @staticmethod
    def eval_dataframe(df):
        """
        Change string list to list after reading from csv
        """
        df['Fragment'] = df['Fragment'].apply(eval)
        df['Technique'] = df['Technique'].apply(eval)

        return df


    def get_text_and_labels(self, df):
        tokenized_words_list, labels_list = [], []
        for i, f in df.iterrows():
            tokenized_words, labels = self.tokenize_preserve(f['Fragment'], f['Technique'], f['Text'])
            tokenized_words_list.append(tokenized_words)
            labels_list.append(labels)

        assert len(tokenized_words_list) == len(labels_list)
        return tokenized_words_list, labels_list

    def get_encoded_data(self,tokenized_words_list, labels_list):
        # The reason wecan't keep padding token as self.tokenizer.pad_token_id whose value is 0 is because then our tags or labels will have
        # ['PAD']: 0, and when we are doing the attention mask we are making sure all ['PAD'] have an attention mask of 0
        # Attention masks allow us to send a batch into the transformer even when the examples in the batch have varying lengths. 
        # We do this by padding all sequences to the same length, then using the “attention_mask” tensor to identify which tokens are padding
        # So it is not included in the num_tags for our model classes and the NLL looks at 0 to num_tags-1 classes so we need the 0 class to be a class the model predicts

        pad_token_id = -100
        self.techniques[self.tokenizer.pad_token] = pad_token_id
        self.techniques['O'] = 0
        id2techniques = {v: k for k, v in self.techniques.items()}
        # cls = [self.tokenizer.cls_token_id]
        # sep = [self.tokenizer.sep_token_id]
        input_ids = pad_sequences(
                            # cls + self.tokenizer.convert_tokens_to_ids(tokenized_txt) + sep
                              [self.tokenizer.convert_tokens_to_ids(tokenized_txt) for tokenized_txt in tokenized_words_list], # converts tokens to ids
                             maxlen= self.hyper_params['max_seq_length'], dtype='long',value=0.0,
                             truncating='post',padding='post')
        tags = pad_sequences(
                        [[self.techniques[l] for l in label]for label in labels_list], # Gets corresponding tag_id
                        maxlen= self.hyper_params['max_seq_length'], dtype='long', value= pad_token_id,
                        truncating='post',padding='post')

        attention_masks = [[float(i !=0.0) for i in ii]for ii in input_ids] # Float(True) = 1.0 for attention for only non-padded inputs

        assert len(input_ids) == len(tags) == len(attention_masks)
        for i in range(len(input_ids)):
            assert len(input_ids[i]) == len(tags[i]) == len(attention_masks[i])
        # print('Inputs: {}'.format(input_ids[0]))
        # print('Tags: {}'.format(tags[0]))
        # print('Attention Mask: {}'.format(attention_masks[0]))
        # print('Lengths Matching: {}, {}, {}'.format(len(input_ids[0]), len(tags[0]), len(attention_masks[0])))
        # print('Lengths: {}, {}, {}'.format(len(input_ids), len(tags), len(attention_masks)))

        return input_ids, tags, attention_masks
        

    def tokenize_preserve(self, fragments, techniques, text):
        assert len(fragments) == len(techniques)
        for i, f in enumerate(fragments):
            if f not in text:
                if techniques[i] == 'Repetition':
                    fragments.pop(i)
                    techniques.pop(i)
                else:
                    print(i, f)
                    raise Exception("Fragment not in text and not a repetition.")
            else:
                start = text.index(f)
                end = start + len(f)
                replace_text = "#"
                text = text[:start] +  replace_text + text[end:]

        tokenized_words = self.tokenizer.tokenize(text)
        labels = ["O"]*len(tokenized_words)
        assert len(tokenized_words) == len(labels)

        for i, f in enumerate(fragments):
            hash_index = tokenized_words.index('#')
            tokenized_words.pop(hash_index)
            labels.pop(hash_index)
            fragments_tokenized = self.tokenizer.tokenize(f)
            if len(tokenized_words) == 0:
                tokenized_words += fragments_tokenized
                labels += [techniques[i]]*len(fragments_tokenized)
            else:
                tokenized_words[hash_index:hash_index] = fragments_tokenized
                labels[hash_index:hash_index] = [techniques[i]]*len(fragments_tokenized)
        assert len(tokenized_words) == len(labels)
        return tokenized_words, labels

    def convert_to_tensors(self,input_ids_list, tags_list, attention_masks_list):

        tr_input, tr_tag, tr_masks = torch.tensor(input_ids_list[0]), torch.tensor(tags_list[0]), torch.tensor(attention_masks_list[0])
        val_input, val_tag, val_masks = torch.tensor(input_ids_list[1]), torch.tensor(tags_list[1]), torch.tensor(attention_masks_list[1])
        # test_input, test_tag, test_masks = torch.tensor(input_ids_list[2]), torch.tensor(tags_list[2]), torch.tensor(attention_masks_list[2])

        return tr_input, tr_tag, tr_masks, val_input, val_tag, val_masks #, test_input, test_tag, test_masks

    def data_loader(self,input_ids_list, tags_list, attention_masks_list):


        # tr_input, tr_tag, tr_masks, val_input, val_tag, val_masks, test_input, test_tag, test_masks = self.convert_to_tensors(input_ids_list, tags_list, attention_masks_list)
        tr_input, tr_tag, tr_masks, val_input, val_tag, val_masks = self.convert_to_tensors(input_ids_list, tags_list, attention_masks_list)

        train_data = TensorDataset(tr_input, tr_masks, tr_tag)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.hyper_params['training_batch_size'])

        valid_data = TensorDataset(val_input, val_masks, val_tag)
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=self.hyper_params['validation_batch_size'])

        # test_data = TensorDataset(test_input, test_masks, test_tag)
        # test_sampler = SequentialSampler(test_data)
        # test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=self.hyper_params['validation_batch_size'])

        return train_dataloader, valid_dataloader #, test_dataloader
    

    def run(self,):

        input_ids_list, tags_list, attention_masks_list = [], [], []
        dataframes_list = self.read_csv_files()
        for df in dataframes_list[:-1]: # Skipping test_set_.csv
            df = self.eval_dataframe(df)
            tokenized_words_list, labels_list = self.get_text_and_labels(df)
            input_ids, tags, attention_masks = self.get_encoded_data(tokenized_words_list, labels_list)
            input_ids_list.append(input_ids)
            tags_list.append(tags)
            attention_masks_list.append(attention_masks)
        
        
        train_dataloader, valid_dataloader = self.data_loader(input_ids_list, tags_list, attention_masks_list)
        return train_dataloader, valid_dataloader, self.techniques 

        # train_dataloader, valid_dataloader, test_dataloader = self.data_loader(input_ids_list, tags_list, attention_masks_list)
        # return train_dataloader, valid_dataloader, test_dataloader, self.techniques 