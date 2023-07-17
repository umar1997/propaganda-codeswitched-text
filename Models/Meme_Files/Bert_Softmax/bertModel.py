import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput

class Propaganda_Detection(nn.Module):
    def __init__(self, checkpoint_model, num_tags):
    
        super(Propaganda_Detection, self).__init__()

        self.num_labels = num_tags
        self.input_dim = 768
        self.model = AutoModel.from_pretrained(
            checkpoint_model,
            config=AutoConfig.from_pretrained(
                checkpoint_model, 
                output_attentions=True,
                output_hidden_states=True)
            )
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
            # https://discuss.pytorch.org/t/how-to-confirm-parameters-of-frozen-part-of-network-are-not-being-updated/142482
            # for name, param in self.model.named_parameters():
            #     param.requires_grad = False

            # https://huggingface.co/docs/transformers/main_classes/output
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            sequence_output = outputs[0] #  outputs[0]=last hidden state
            x = sequence_output[:,:,:].view(-1,768)
            logits = self.linear_relu_stack(x)

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels.view(-1))
                return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,attentions=outputs.attentions)
        
        else: # For inference
            outputs = self.model(input_ids=input_ids)

            sequence_output = outputs[0] #  outputs[0]=last hidden state
            x = sequence_output[:,:,:].view(-1,768)
            logits = self.linear_relu_stack(x)
            return TokenClassifierOutput(loss=None, logits=logits, hidden_states=None,attentions=None)


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