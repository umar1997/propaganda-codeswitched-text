import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss


from transformers import AutoModel, AutoConfig, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

# https://stackoverflow.com/questions/74297955/how-to-remove-layers-in-huggingfaces-transformers-bert-pre-trained-models
# def deleteLayers(model):  # must pass in the full bert model
#     embeddingLayer = model.bert.embeddings
#     encoderLayer = model.bert.encoder
#     poolerLayer = model.bert.pooler
#     newModuleList = nn.ModuleList()

#     newModuleList.append(embeddingLayer)
#     newModuleList.append(encoderLayer)
#     newModuleList.append(poolerLayer)
#     # Ignoring classifier and dropout layer
#     assert newModuleList == model.bert # False
#     return newModuleList

# model = deleteLayers(model)

class Propaganda_Detection(nn.Module):
    def __init__(self, checkpoint_model, num_tags, device, hyper_params):
    
        super(Propaganda_Detection, self).__init__()

        self.num_labels = num_tags
        self.device = device

        config=AutoConfig.from_pretrained(
                checkpoint_model, 
                output_attentions=True,
                output_hidden_states=True)
                # use_fast=False) # Added this for Deberta_V3
        if hyper_params['model_run'] == 'RUBERT':
            load_model = AutoModelForSequenceClassification.from_pretrained(checkpoint_model, config=config)
            # Removing the dropout layer and classifier layer
            self.model = load_model.bert
        else:
            self.model = AutoModel.from_pretrained(
                checkpoint_model,
                config=config
                )

        # self.model = AutoModelForSequenceClassification.from_pretrained(
        #     checkpoint_model,
        #     num_labels = num_tags
        # )
        self.input_dim = self.model.config.hidden_size # 768
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_labels),
        )

    def forward(self, 
        input_ids, 
        attention_mask=None,
        labels=None,
        training=None,
        token_type_ids=None,
    ):

        if training: # For training
            assert labels.shape[1] == 20
            # https://discuss.pytorch.org/t/how-to-confirm-parameters-of-frozen-part-of-network-are-not-being-updated/142482
            # for name, param in self.model.named_parameters():
            #     param.requires_grad = False

            # https://huggingface.co/docs/transformers/main_classes/output
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # sequence_output = outputs[0] #  outputs[0]=outputs.last_hidden_state
            

            # https://towardsdatascience.com/tips-and-tricks-for-your-bert-based-applications-359c6b697f8e#:~:text=pooler_output%20is%20the%20embedding%20of,from%20the%20last%20hidden%20state.
            # Either
            # sequence_output = outputs.pooler_output
            # Or
            max_length = outputs.last_hidden_state.shape[1] # last_hidden_state (12, 256, 1024/768)
            intermediate = torch.matmul(attention_mask.view(-1,1,max_length), outputs.last_hidden_state)
            sequence_output = torch.squeeze(intermediate,1)
            # attention_mask.view(-1,1,max_length).shape = torch.Size([12, 1, 256])
            # outputs.last_hidden_state.shape            = torch.Size([12, 256, 768])
            # intermediate.shape                         = torch.Size([12, 1, 768])
            logits = self.linear_relu_stack(sequence_output)

            if labels is not None:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
                return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,attentions=outputs.attentions)
        
        else: # For inference
            outputs = self.model(input_ids=input_ids)
            # Either
            # x = outputs.pooler_output
            # Or
            seq_len = outputs[0].shape[1] # torch.Size([1, 38, 768])
            attention_copy = torch.tensor(np.ones((seq_len)).reshape(1,1,-1)).to(self.device) # (1, 1, 38)
            attention_copy = attention_copy.type(torch.float)
            intermediate = torch.matmul(attention_copy, outputs.last_hidden_state)
            x = torch.squeeze(intermediate,1)
            logits = self.linear_relu_stack(x)
            return SequenceClassifierOutput(loss=None, logits=logits, hidden_states=None,attentions=None)


# BaseModelOutputWithPoolingAndCrossAttentions
# https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions

# last_hidden_state
# Sequence of hidden-states at the output of the last layer of the model
# (batch_size, sequence_length, hidden_size)
# outputs[0].shape
# torch.Size([16, 256, 768])

# pooler_output
# (batch_size, hidden_size)
# outputs[1].shape
# torch.Size([16, 768])


# attentions for each encoder
# (batch_size, num_heads, sequence_length, sequence_length)
# outputs[3][0-11].shape    # 12 num_heads
# torch.Size([16, 12, 256, 256])

# hidden_states
# (batch_size, sequence_length, hidden_size)
# one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer
# outputs[2][0-12].shape
# torch.Size([16, 256, 768])



#################### TOKENIZER
# https://huggingface.co/docs/tokenizers/api/tokenizer


#################### MODEL
# BertModel(
#     (embeddings): BertEmbeddings(
#         (word_embeddings): Embedding(28996, 768, padding_idx=0)
#         (position_embeddings): Embedding(512, 768)
#         (token_type_embeddings): Embedding(2, 768)
#         (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
#         (dropout): Dropout(p=0.1, inplace=False)
#     )
#     (encoder): BertEncoder(
#         (layer): ModuleList(
#             (0): BertLayer(
#                 (attention): BertAttention(
#                     (self): BertSelfAttention(
#                         (query): Linear(in_features=768, out_features=768, bias=True)
#                         (key): Linear(in_features=768, out_features=768, bias=True)
#                         (value): Linear(in_features=768, out_features=768, bias=True)
#                         (dropout): Dropout(p=0.1, inplace=False)
#                     )
#                     (output): BertSelfOutput(
#                         (dense): Linear(in_features=768, out_features=768, bias=True)
#                         (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
#                         (dropout): Dropout(p=0.1, inplace=False)
#                     )
#                 )
#             )
#             (intermediate): BertIntermediate(
#             (dense): Linear(in_features=768, out_features=3072, bias=True)
#             (intermediate_act_fn): GELUActivation()
#             )
#             (output): BertOutput(
#             (dense): Linear(in_features=3072, out_features=768, bias=True)
#             (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
#             (dropout): Dropout(p=0.1, inplace=False)
#             )
#         )
#     )
#     (pooler): BertPooler(
#         (dense): Linear(in_features=768, out_features=768, bias=True)
#         (activation): Tanh()
#     )
# )