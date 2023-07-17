import json
import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from skmultilearn.model_selection import iterative_train_test_split

def read_techniques(filename):
        """
        Read the techniques json file into a dictionary
        """
        with open(filename, "r") as fp:
            techniques = json.load(fp)
        
        return techniques

def read_json(json_file):

    with open(json_file, 'r') as f:
        data = json.loads(f.read())
    return data

def read_json_files_to_df(data):
    """
    Read file from json format and convert into pandas datatframe.
    """

    data_dict = dict()
    for i, (key, example) in enumerate(data.items()):
        text = example['text']
        list_labels = example['labels']

        data_dict[i] = {'id': key, 'text' : text, 'technique' : []}
        for label in list_labels:
            technique = label['technique']
            data_dict[i]['technique'].append(technique)

    data_df = pd.DataFrame(data_dict).transpose()
    data_df = shuffle(data_df)
    data_df = data_df.reset_index(drop=True)
    return data_df

def create_arrays(df, techniques):

    id_list, label_list = [], []
    for i, f in df.iterrows():
        exmaple_id = f['id']
        examnple_techniques = f['technique']
        indices = [techniques[t] for t in examnple_techniques]
        labels = np.zeros((20))
        labels[indices] = 1
        label_list.append(labels)
        id_list.append(int(exmaple_id))
    
    label_list = np.array(label_list)
    id_list = np.array(id_list)[:, np.newaxis]

    return id_list, label_list

def make_splits(id_list, label_list):

    X_train_val, y_train_val, X_test, y_test = iterative_train_test_split(id_list, label_list, test_size = 0.15)
    X_train, y_train, X_val, y_val = iterative_train_test_split(X_train_val, y_train_val, test_size = 0.10)
    return X_train.reshape(-1).tolist(), X_val.reshape(-1).tolist(), X_test.reshape(-1).tolist()

def make_json_files(X_train, X_val, X_test, data):

    train_split = dict( ((key, data[key]) for key in list(map(str, X_train))) )
    val_split = dict( ((key, data[key]) for key in list(map(str, X_val))) )
    test_split = dict( ((key, data[key]) for key in list(map(str, X_test))) )
    
    
    with open('./Splits/train_split.json', "w") as fp:
            json.dump(train_split, fp, indent=4)
    with open('./Splits/val_split.json', "w") as fp:
            json.dump(val_split, fp, indent=4)
    with open('./Splits/test_split.json', "w") as fp:
            json.dump(test_split, fp, indent=4)


if __name__ == '__main__':

    seed = 42
    np.random.seed(seed)
    techniques_file = '../techniques.json'
    json_file = 'dataset.json'
    techniques = read_techniques(techniques_file)
    data = read_json(json_file)
    df = read_json_files_to_df(data)
    id_list, label_list = create_arrays(df, techniques)
    X_train, X_val, X_test = make_splits(id_list, label_list)
    make_json_files(X_train, X_val, X_test, data)