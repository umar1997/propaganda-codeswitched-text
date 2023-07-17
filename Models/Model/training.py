import os
import time
import shutil

import numpy as np
from tqdm import tqdm, trange

from Model.evaluation import *

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

        self.checkpoint = {
            'model': None,
            'epoch': 0,
            'best_hamming_score': 0,
            'best_hamming_score_epoch': 0,
            'best_exact_match_ratio': 0,
            'best_exact_match_ratio_epoch': 0,
        }

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
        # self.hyper_params['epochs'] = 1
        for _ in trange(self.hyper_params['epochs'], desc= "Epoch \n"):
            print('\n')
            print('     Epoch #{}'.format(E))
            if not self.hyper_params["debugging"]:
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
            if not self.hyper_params["debugging"]:
                self.logger_results.info('Average Train Loss For Epoch {}: {}'.format(E, avg_train_loss))

            loss_values.append(avg_train_loss) # Storing loss values to plot learning curve
            ###################### VALIDATION
            self.model.eval()
            
            eval_loss = 0
            predictions, true_labels = [], []
            
            for step, batch in enumerate(self.valid_dataloader):
                batch = tuple(t.to(self.device)for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                
                with torch.no_grad(): # No backprop
                    outputs = self.model(b_input_ids, attention_mask=b_input_mask, 
                    labels=b_labels, training=True ,token_type_ids=None) # Forward pas
                
                # Getting Probabilities for Prediction Classes
                logits = outputs[1].detach().cpu().numpy()
                pred = np.where(logits>=0, 1, 0)

                # Golden Labels
                label_ids = b_labels.to('cpu').numpy()
                
                loss = outputs[0]
                eval_loss += loss.item()

                predictions.extend(pred.tolist())
                true_labels.extend(label_ids.tolist())

            predictions = np.array(predictions)
            true_labels = np.array(true_labels)

            avg_eval_loss = eval_loss / len(self.valid_dataloader)
            print('     Average Val Loss For Epoch {}: {}'.format(E, avg_eval_loss))
            if not self.hyper_params["debugging"]:
                self.logger_results.info('Average Val Loss For Epoch {}: {}'.format(E, avg_eval_loss))

            validation_loss_values.append(avg_eval_loss)   
            
            
            accuracy_score = get_accuracy_score(true_labels, predictions)
            hamming_score = get_hamming_score(true_labels, predictions)
            exact_match_ratio = get_exact_match_ratio(true_labels, predictions)
            print('     Validation Accuracy Score: {}'.format(accuracy_score))
            print('     Validation Hamming Score: {}'.format(hamming_score))
            print('     Validation Exact Match Ratio: {}'.format(exact_match_ratio))
            if not self.hyper_params["debugging"]:
                self.logger_results.info('Validation Hamming Score: {}  |  Validation Exact Match Ratio: {} |  Validation Accuracy Score: {}'.format(hamming_score, exact_match_ratio, accuracy_score))

            if hamming_score >= self.checkpoint['best_hamming_score']:
                self.checkpoint['best_hamming_score'] = hamming_score
                self.checkpoint['best_hamming_score_epoch'] = E
            if exact_match_ratio >= self.checkpoint['best_exact_match_ratio']:
                self.checkpoint['best_exact_match_ratio'] = exact_match_ratio
                self.checkpoint['best_exact_match_ratio_epoch'] = E

            stop = time.time()
            print('     Epoch #{} Duration:{}'.format(E, stop-start))
            if not self.hyper_params["debugging"]:
                self.logger_results.info('Duration: {}\n'.format(stop-start))
            E+=1
            # print('-'*20)
        
        self.checkpoint['epoch'] = E

        labels_ = list(self.techniques.keys())
        classificationReport = get_classification_report(true_labels, predictions, labels_)
        if not self.hyper_params["debugging"]:
            self.logger_results.info('Validation Hamming Score: {}  |  Validation Exact Match Ratio: {} |  Validation Accuracy Score: {}'.format(hamming_score, exact_match_ratio, accuracy_score))
            self.logger_results.info('Classification Report:')
            self.logger_results.info('\n{}'.format(classificationReport))

    def save_model(self,):
        if not self.hyper_params["debugging"]:

            foldername = self.hyper_params['model_run']
            path = os.getcwd() + '/Model_Files/' + foldername
            modelFolderExist = os.path.exists(path)
            if not modelFolderExist:
                os.makedirs(path)
            else:
                print('Exception: Model Folder already exists with this name. Assign path variable a new folder name.')
                breakpoint()
                # Do path = os.getcwd() + self.paths['Model_Files'] + 'new_name'
            self.checkpoint['model'] = self.model.state_dict()
            shutil.copy2(self.hyper_params['log_file'], path)
        
            torch.save(self.checkpoint, path + '/' + self.hyper_params['model_run'] + '.pt')
            self.tokenizer.save_pretrained(path + '/' + self.hyper_params['model_run']+ '_tokenizer/')

    def save_intermediate(self,):
        if not self.hyper_params["debugging"]:
            filename = self.hyper_params['model_run']
            path = os.getcwd() + '/Switch_Files/'

            self.checkpoint['model'] = self.model.state_dict()
            # Delete existing files
            if os.path.isfile(path + 'INTERMEDIATE.pt'):
                os.remove(path + 'INTERMEDIATE.pt')
            if os.path.exists(path + '/INTERMEDIATE' + '_tokenizer/'):
                shutil.rmtree(path + '/INTERMEDIATE' + '_tokenizer/')

            # Save new files
            if self.hyper_params['step'] == 'Last':
                torch.save(self.checkpoint, path + '/' + self.hyper_params['model_run'] + '.pt')
                self.tokenizer.save_pretrained(path + '/' + self.hyper_params['model_run'] + '_tokenizer/')
            else:
                torch.save(self.checkpoint, path + '/INTERMEDIATE' + '.pt')
                self.tokenizer.save_pretrained(path + '/INTERMEDIATE' + '_tokenizer/')

    
    
    def run(self,):
    
        optimizer, scheduler = self.optimizer_and_lr_scheduler()
        self.training_and_validation(optimizer, scheduler)
        if 'mode' in self.hyper_params:
            if self.hyper_params['mode'] == 'SWITCHES':
                self.save_intermediate()
        else:
            self.save_model()

