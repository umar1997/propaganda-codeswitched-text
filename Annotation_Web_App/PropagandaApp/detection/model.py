import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoConfig
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