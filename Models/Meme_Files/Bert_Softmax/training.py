import time

import numpy as np
from tqdm import tqdm, trange

from Models.Bert_Softmax.evaluation import *

import torch
from torch.optim import AdamW, SGD, lr_scheduler

from transformers import get_linear_schedule_with_warmup


class Training:
    def __init__(self, paths, model, tokenizer, hyper_params, train_dataloader, valid_dataloader, techniques, logger_results):

        self.paths = paths
        self.model = model
        self.tokenizer = tokenizer
        self.hyper_params = hyper_params
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.techniques = techniques
        self.id2techniques = {v: k for k, v in self.techniques.items()}
        self.logger_results = logger_results

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert self.device == torch.device('cuda')

    def optimizer_and_lr_scheduler(self,):
        
        # All the layers weights should be updated or just the last classification layer
        if self.hyper_params['full_finetuning']:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                # Setting Weight Decay Rate 0.01 if it isnt bias, gamma and beta
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
                'weight_decay_rate': 0.01},
                # If it is set to 0.0
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
                'weight_decay_rate': 0.0}
            ]
        else: # Non Fine Tuning
            param_optimizer = list(self.model.classifier.named_parameters())
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
        
        # Optimizer
        if self.hyper_params['optimizer'] == 'AdamW':
            
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr= self.hyper_params['learning_rate'],
                eps= self.hyper_params['epsilon'],
                # weight_decay=0.01 Doing stuff above for weight decay
            )
        elif self.hyper_params['optimizer'] == 'SGD':

            optimizer = SGD(
                # self.model.parameters(),
                optimizer_grouped_parameters,
                lr=self.hyper_params['learning_rate'], # 0.1 usually
                momentum=0.9, # 0.9 usually
                dampening=0,
                weight_decay=self.hyper_params['weight_decay'],
                nesterov=False
            )
        # Total number of training steps is number of batches * number of epochs.
        total_steps = len(self.train_dataloader) * self.hyper_params['epochs']

        # Create the learning rate scheduler.
        if self.hyper_params['scheduler'] == 'LinearWarmup':
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=total_steps
            )
        
        elif self.hyper_params['scheduler'] == 'LRonPlateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.1, 
                patience=2, # Number of epochs with no improvement after which learning rate will be reduced
                threshold=0.001, 
                threshold_mode='rel', 
                cooldown=0, 
                min_lr=0, 
                eps=1e-08, 
                verbose=False)

        return optimizer, scheduler


    def training_and_validation(self, optimizer, scheduler):

        loss_values, validation_loss_values = [], []
        E = 1
        for _ in trange(self.hyper_params['epochs'], desc= "Epoch \n"):
            print('\n')
            print('     Epoch #{}'.format(E))
            self.logger_results.info('Epoch #{}'.format(E))
        
            start = time.time()

            self.model.train()
            total_loss=0 # Reset at each Epoch
            
            ###################### TRAINING
            for step, batch in enumerate(self.train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch # Mantained the order for both train_data/val_data
                
                self.model.zero_grad() # Clearing previous gradients for each epoch
                
                outputs = self.model(b_input_ids, attention_mask=b_input_mask, 
                labels=b_labels, training=True ,token_type_ids=None) # Forward pass
                
                loss = outputs[0]
                loss.backward() # Getting the loss and performing backward pass
                
                total_loss += loss.item() # Tracking loss
                
                # Preventing exploding grads
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.hyper_params['max_grad_norm'])
                
                optimizer.step() # Updates parameters
                if scheduler:
                    scheduler.step() # Update learning_rate

            avg_train_loss = total_loss/len(self.train_dataloader) 
            print('     Average Train Loss For Epoch {}: {}'.format(E, avg_train_loss))
            self.logger_results.info('Average Train Loss For Epoch {}: {}'.format(E, avg_train_loss))

            loss_values.append(avg_train_loss) # Storing loss values to plot learning curve
            ###################### VALIDATION
            self.model.eval()
            
            eval_loss = 0
            predictions, true_labels = [], []
            
            for batch in self.valid_dataloader:
                batch = tuple(t.to(self.device)for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                
                with torch.no_grad(): # No backprop
                    outputs = self.model(b_input_ids, attention_mask=b_input_mask, 
                    labels=b_labels, training=True ,token_type_ids=None) # Forward pas
                    
                # Getting Probabilities for Prediction Classes
                logits = outputs[1].detach().cpu().numpy() # 16 * 256 = (4096, 21)
                # logits_x = logits.reshape(self.hyper_params['validation_batch_size'], self.hyper_params['max_seq_length'], logits.shape[2])
                # Golden Labels
                label_ids = b_labels.to('cpu').numpy() # (16, 256)
                label_ids = label_ids.reshape(-1) # (4096,)
                
                loss = outputs[0]
                eval_loss += loss.item()

                predictions.extend(np.argmax(logits, axis=1).tolist()) # Taking Max among Prediction Classes
                true_labels.extend(label_ids.tolist())


            avg_eval_loss = eval_loss / len(self.valid_dataloader)
            print('     Average Val Loss For Epoch {}: {}'.format(E, avg_eval_loss))
            self.logger_results.info('Average Val Loss For Epoch {}: {}'.format(E, avg_eval_loss))

            validation_loss_values.append(avg_eval_loss)            
            
            pred_tags = [self.id2techniques[p] for p, l in zip(predictions, true_labels) if self.id2techniques[l] !='[PAD]']            
            valid_tags = [self.id2techniques[l]for l in true_labels if self.id2techniques[l] !='[PAD]']

            accuracyScore = get_accuracy(valid_tags, pred_tags)
            f1Score = get_f1_score(valid_tags, pred_tags)
            print('     Validation Accuracy: {}%'.format(accuracyScore))
            print('     Validation F-1 Score:{}'.format(f1Score))

            self.logger_results.info('Validation Accuracy: {}%  |  Validation F-1 Score:{}'.format(accuracyScore, f1Score))

            stop = time.time()
            print('     Epoch #{} Duration:{}'.format(E, stop-start))
            self.logger_results.info('Duration: {}\n'.format(stop-start))
            E+=1
            # print('-'*20)
        
        labels_ = list(self.techniques.keys())
        labels_.remove('[PAD]')

        classificationReport = get_classification_report(valid_tags, pred_tags)
        confusionMatrix = get_confusion_matrix(valid_tags, pred_tags, labels_)

        self.logger_results.info('Final Validation Accuracy: {}%  |  Final Validation F-1 Score:{}'.format(accuracyScore, f1Score))
        self.logger_results.info('Classification Report:')
        self.logger_results.info('\n{}'.format(classificationReport))
        self.logger_results.info('Confusion Matrix:')
        self.logger_results.info('\n{}'.format(confusionMatrix))

    def save_model(self,):

        torch.save(self.model.state_dict(), self.paths['Model_Files'] + 'model_bert.pt')
        self.tokenizer.save_pretrained(self.paths['Model_Files'] +'tokenizer/')
    
    def run(self,):
    
        optimizer, scheduler = self.optimizer_and_lr_scheduler()
        self.training_and_validation(optimizer, scheduler)
        self.save_model()

