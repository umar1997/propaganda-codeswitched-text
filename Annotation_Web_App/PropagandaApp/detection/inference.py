import json
import torch
import numpy as np

class Predict:
    def __init__(self, config, model, tokenizer):

        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert self.device == torch.device('cuda')

        with open(self.config["id2techniques"], "r") as fp:
            self.techniques2id = json.load(fp)

        with open(self.config["colour2techniques"], "r") as fp:
            self.colour2techniques = json.load(fp)

        self.model = model
        self.tokenizer = tokenizer

        self.sentence = ""

        self.model = self.model.to(self.device)
        self.model.eval()
        
    def get_sentence_prediction(self, sentence_to_be_labelled):
        self.sentence = sentence_to_be_labelled
        tokens, labels = self.make_prediction(sentence_to_be_labelled)
        merged_list = self. merge_lists(tokens, labels)
        result_list = self.format_prediction(merged_list)
        final_output = self.get_colour_formatted_json(result_list)
        return final_output

    def get_colour_formatted_json(self, result_list):
        final_output = []
        technique2colour = {value:key for key, value in self.colour2techniques.items()}
        for t in result_list:
            x = dict(phrase = t[1], colour=technique2colour[t[0]])
            final_output.append(x)
        return final_output



    def merge_lists(self, list1, list2):
        merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
        return merged_list

    def format_prediction(self, merged_list): 
        # merged_list: List of tuples with first element as token and second as label

        sentence = self.sentence.lower()
        list_wo_space_info = merged_list
        list_with_space_info = []
        for ele in list_wo_space_info:
            start_index = sentence.find(ele[0])
            end_index = start_index + len(ele[0])
            if sentence[end_index:end_index+1] == " ":
                list_with_space_info.append([ele[0], ele[1], True])
            else:
                list_with_space_info.append([ele[0], ele[1], False])
            sentence = sentence[end_index:]

        result_list = []
        temp = list_wo_space_info[0][0]
        for i, t in enumerate(list_with_space_info):
            try:
                if t[1] == list_wo_space_info[i+1][1]:
                    label = t[1]
                    if t[2] == True:
                        temp = temp + " " + list_wo_space_info[i+1][0]
                    else:
                        temp = temp + list_wo_space_info[i+1][0]
                else:
                    label = list_wo_space_info[i][1]
                    result_list.append((label, temp))
                    temp = list_wo_space_info[i+1][0]
            except:
                result_list.append((label, temp))

        return result_list

    def make_prediction(self,sentence_to_be_labelled):

        classes = list(self.techniques2id.keys())
        self.techniques2id['O'] = 0
        id2techniques = {v: k for k, v in self.techniques2id.items()}

        tokenized_sentence = self.tokenizer.encode(sentence_to_be_labelled)
        tokenized_sentence = tokenized_sentence[1:-1]
        input_ids = torch.tensor([tokenized_sentence])
        input_ids = input_ids.to(self.device)


        with torch.no_grad(): 
            output = self.model(input_ids)
            label_indices = np.argmax(output[0].to('cpu').numpy(),axis=1)
            label_indices = label_indices.tolist()
            

            pred_tags = [id2techniques[l] for l in label_indices]
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])

            new_tokens, new_labels = [], []
            for token, label_idx in zip(tokens, label_indices):
                if token.startswith('##'):
                    new_tokens[-1] = new_tokens[-1] + token[2:]
                else:
                    new_labels.append(id2techniques[label_idx])
                    new_tokens.append(token)

            assert len(new_tokens) == len(new_labels)
            return  new_tokens, new_labels

            # for token, label in zip(new_tokens, new_labels): # Showing labels against the words
            #     print('{}\t{}'.format(label,token))
            