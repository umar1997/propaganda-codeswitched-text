from dataPreparation import Dataset_Preparation

from Models.Bert_Softmax.dataProcessing import Data_Preprocessing
from Models.Bert_Softmax.bertModel import Propaganda_Detection
from Models.Bert_Softmax.training import Training
from Models.Bert_Softmax.inference import Inferencer

from log import get_logger

import torch
import numpy as np

import argparse
from packaging import version

import transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification

pytorch_version = version.parse(transformers.__version__)
assert pytorch_version >= version.parse('3.0.0'), \
    'We now only support transformers version >=3.0.0, but your version is {}'.format(pytorch_version)

if __name__ == '__main__':

    # https://github.com/uf-hobi-informatics-lab/ClinicalTransformerNER/blob/master/src/run_transformer_ner.py
    parser = argparse.ArgumentParser()

    # ADD ARGUEMENTS
    parser.add_argument("--model_run", default='Bert_Softmax', type=str,
                        help="valid values: Bert_Softmax")
    parser.add_argument("--model_type", default='bert-base-cased', type=str,
                        help="valid values: bert-base-cased")
    parser.add_argument("--tokenizer_type", default='bert-base-cased', type=str,
                        help="valid values: bert-base-cased")
    parser.add_argument("--training", default=0, type=int,
                        help="valid values: 1 for Training, 0 for Validation")
    parser.add_argument("--seed", default=42, type=int,
                        help='random seed')
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="maximum number of tokens allowed in each sentence")
    parser.add_argument("--validation_batch_size", default=16, type=int,
                        help="The batch size for evaluation.")
    parser.add_argument("--training_batch_size", default=16, type=int,
                        help="The batch size for training")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for optimizer.")
    parser.add_argument("--num_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.1, type=float,
                        help="Weight Decay for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--optimizer", default='AdamW', type=str,
                        help="valid values: AdamW, SGD")
    parser.add_argument("--scheduler", default='LinearWarmup', type=str,
                        help="valid values: LinearWarmup, LRonPlateau")
    parser.add_argument("--full_finetuning", default=1, type=int,
                        help="Update weights for all layers or finetune last classification layer")
    parser.add_argument("--log_file", default='test.log', type=str,
                        help="Name of log file.")

    global_args = parser.parse_args()

    paths = {
        "Meme_Data": "./Meme_Data/",
        "Meme_Data_Train_Json": "./Meme_Data/training_set_.json",
        "Meme_Data_Val_Json": "./Meme_Data/dev_set_.json",
        "Meme_Data_Test_Json": "./Meme_Data/test_set_.json",
        "Meme_Data_Train":"./Meme_Data/training_set_.csv",
        "Meme_Data_Val":"./Meme_Data/dev_set_.csv",
        "Meme_Data_Test":"./Meme_Data/test_set_.csv",
        "Techniques":"./techniques.json",
        "Log_Folder":"./Log_Files/",
        "Model_Files":"./Model_Files/",
    }

    hyper_params = {
        "model_run": global_args.model_run,
        "model_type": global_args.model_type,
        "tokenizer_type": global_args.tokenizer_type,
        "training": bool(global_args.training),
        "max_seq_length": global_args.max_seq_length,
        "random_seed": global_args.seed,
        "training_batch_size": global_args.training_batch_size,
        "validation_batch_size": global_args.validation_batch_size,
        "learning_rate": global_args.learning_rate,
        "epsilon": global_args.adam_epsilon,
        "weight_decay": global_args.weight_decay,
        "epochs": global_args.num_epochs,
        "scheduler": global_args.scheduler,
        "optimizer": global_args.optimizer,
        "max_grad_norm": global_args.max_grad_norm,
        "full_finetuning": bool(global_args.full_finetuning),
    }



    ################################################## SEEDS
    seed = hyper_params['random_seed']
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    from datetime import datetime
    current_datetime = datetime.now()
    date_time = current_datetime.strftime("%d-%m-%Y_%H:%M:%S")
    

    ################################################## LOG FILE SET UP
    file_name = paths['Log_Folder'] + hyper_params['model_run'] + date_time #global_args.log_file
    logger_meta = get_logger(name='META', file_name=file_name, type='meta')
    logger_progress = get_logger(name='PORGRESS', file_name=file_name, type='progress')
    logger_results = get_logger(name='RESULTS', file_name=file_name, type='results')
    for i, (k, v) in enumerate(hyper_params.items()):
        if i == (len(hyper_params) - 1):
            logger_meta.warning("{}: {}\n".format(k, v))
        else:
            logger_meta.warning("{}: {}".format(k, v))


    ################################################## 
    checkpoint_model = hyper_params['model_type']
    checkpoint_tokenizer = hyper_params['tokenizer_type']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device == torch.device('cuda')


    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< BERT SOFTMAX >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if hyper_params['model_run'] == 'Bert_Softmax':
        ##################################################  MODEL + TOKENIZER
        techniques = Data_Preprocessing.read_techniques(paths['Techniques'])
        if hyper_params['training']:

            tokenizer = AutoTokenizer.from_pretrained(checkpoint_tokenizer, do_lower_case = True)
            model = Propaganda_Detection(checkpoint_model=checkpoint_model, num_tags=len(techniques)+1) # Adding 'O' token
            model = model.to(device)
            print('##################################################')
            logger_progress.critical('Model + Tokenizer Initialized')

            ##################################################  DATASET PREPARATION
            # Use when need to generate the non overlapped .csv files from the .json files in Meme_Data folder
            # dataRaw = Dataset_Preparation(paths)
            # dataRaw.run()

            ##################################################  DATA PROCESSING
            dataProcessed = Data_Preprocessing(paths, tokenizer, hyper_params)
            train_dataloader, valid_dataloader, techniques = dataProcessed.run()
            logger_progress.critical('Tokenizing sentences and encoding labels')
            logger_progress.critical('Data Loaders Created')


            ##################################################  TRAINING
            logger_progress.critical('Training Started')
            train = Training(paths, model, tokenizer, hyper_params, train_dataloader, valid_dataloader, techniques, logger_results)
            train.run()
            logger_progress.critical('Training Finished')
            logger_progress.critical('Model Saved')
        else:
            ################################################## INFERENCE
            print('##################################################')
            logger_progress.critical('Starting Inference')
            inference = Inferencer(paths, checkpoint_tokenizer, checkpoint_model, hyper_params, techniques)
            macro_f1, micro_f1 = inference.run()
            logger_results.info('Macro F1-Score | Micro F1-Score :  {} | {}'.format(macro_f1, micro_f1))
            logger_progress.critical('Inference Ended')
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<                 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>









    #----------------------------------------------------------------------------------
    script = """
    python main.py \
        --model_run Bert_Softmax \
        --training 1 \
        --model_type bert-base-cased \
        --tokenizer_type bert-base-cased \
        --max_seq_length 256 \
        --training_batch_size 16 \
        --validation_batch_size 16 \
        --learning_rate 5e-5 \
        --num_epochs 10 \
        --seed 42 \
        --adam_epsilon 1e-8 \
        --max_grad_norm 1.0 \
        --optimizer AdamW \
        --scheduler LinearWarmup \
        --full_finetuning 1 \
        --added_layers 0 \
        --log_file test.log
    """
