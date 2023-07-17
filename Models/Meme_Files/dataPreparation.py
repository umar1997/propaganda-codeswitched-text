import re
import json
import itertools
import pandas as pd
from time import time
from tqdm import tqdm

class Dataset_Preparation():
    def __init__(self, paths):
        self.paths = paths
        self.files = [self.paths["Meme_Data_Train_Json"], self.paths["Meme_Data_Val_Json"], self.paths["Meme_Data_Test_Json"]]
        self.techniques = self.read_techniques(self.paths["Techniques"])

    


    ####################################################################################### HELPER FUNCTIONS

    def isOverlap(self, a,b):
        return a[0] <= b[0] <= a[1] or b[0] <= a[0] <= b[1]

    def convertToListOfTuples(self, overlap_dict):
        return [(k,v) for k, values in overlap_dict.items() for v in values]

    def sortElementsInTuples(self, overlap_list):
        return [(k,v) if k<v else (v,k) for k, v in overlap_list]

    def removeDuplicates(self, overlap_list):
        return [t for t in (set(tuple(i) for i in overlap_list))]

    def getMaxValue(self, overlap_list):
        return max(overlap_list, key=lambda tup: tup[1])[1]

    def getTupleCombinations(self, numbered_array, max_value):
        max_value = max_value + 1
        all_combinations = []
        for i in range(0,max_value):
            all_combinations += list(itertools.combinations(numbered_array,i))
        return all_combinations

    def getNonOverlaps(self, tupleCombinations, overlap_list, max_value):
        non_overlap_tuples = tupleCombinations[:]
        unique_non_overlaps = []
        for overlap in overlap_list:
            for a_tuple in tupleCombinations:
                if set(overlap).issubset(a_tuple):
                    try:
                        non_overlap_tuples.remove(a_tuple)
                    except:
                        continue
        numbered_list = list(range(max_value+1))
        for t in non_overlap_tuples:
            if len(t) >= 2:
                unique_non_overlaps += t
        single_tuples = list(set(numbered_list) - set(unique_non_overlaps))
        non_overlap_tuples = [t for t in non_overlap_tuples if len(t) >= 2]
        for single in single_tuples:
            non_overlap_tuples.append((single,))

        new_non_overlap_tuples = non_overlap_tuples[:]
        for old_t in new_non_overlap_tuples:
            for new_t in new_non_overlap_tuples:
                if old_t != new_t:
                    if set(old_t).issubset(new_t):
                        try:
                            non_overlap_tuples.remove(old_t)
                        except:
                            continue
        return non_overlap_tuples
    
    ##########################################################################################################

    def get_data_splits(self,):
        """
        Print splits of training, validation and test data
        """
        for file in self.files :
            with open(file, 'r') as f:
                data = json.loads(f.read())
        
            print(file, len(data))

    def save_techniques(self,):
        """
        Go over all the json files and save unique techniques in a json file
        """
        techniques = set()
        for file in self.files :
            with open(file, 'r') as f:
                data = json.loads(f.read())

            for example in data:
                for list_labels in example['labels']:
                    techniques.add(list_labels['technique'])
        techniques = dict(enumerate(techniques))
        techniques = {y: x+1 for x, y in techniques.items()}

        with open("techniques.json", "w") as fp:
            json.dump(techniques, fp, indent=4)

    def read_techniques(self, filename):
        """
        Read the techniques json file into a dictionary
        """
        with open(filename, "r") as fp:
            techniques = json.load(fp)
        
        return techniques

    def clean_text(self, text):
        text = text.replace('\n',' ')
        text = text.replace('•', ' ')
        text = text.replace('*', ' ')
        text = text.replace('#', ' ')
        text = re.sub(r' {2,}', ' ',text)
        text = text.strip()
        return text
    
    def read_json_files_to_df(self,):
        """
        Read file from json format and convert into pandas datatframe.
        """
        dataframes_split = []
        for file in self.files :
            with open(file, 'r') as f:
                data = json.loads(f.read())
            

            data_dict = dict()

            for i, example in enumerate(data):
                text = self.clean_text(example['text'])
                list_labels = example['labels']

                data_dict[i] = {'text' : text, 'technique' : [], 'text_fragment' : []}
                for label in list_labels:
                    technique = label['technique']
                    fragment = self.clean_text(label['text_fragment'])
                    data_dict[i]['technique'].append(technique)
                    data_dict[i]['text_fragment'].append(fragment)
                
                assert len(data_dict[i]['technique']) == len (data_dict[i]['text_fragment'])

            data_df = pd.DataFrame(data_dict).transpose()

            dataframes_split.append(data_df)
        
        return dataframes_split


    def check_overlap(self,i, f):
        # Different overlap examples
        # a, b  = (10, 20), (15, 25)
        # a, b  = (5, 15), (10, 20)
        # a, b  = (10, 20), (10, 20)
        # a, b  = (15, 20), (10, 25)
        """
        Gives you possible combinations that dont overlap for e.g.
        I will never concede! NO WAY IN HELL BIDEN WON!
        ['I will never', 'never concede!', 'concede! NO', 'WAY IN', 'HELL BIDEN', 'BIDEN WON!']
        [(1, 3, 4), (1, 3, 5), (0, 2, 3, 4), (0, 2, 3, 5)]
        """
        index = i
        text = f['text']
        text_fragment_list = f['text_fragment']
        techniques = f['technique']

        no_labels = 1
        if len(techniques) == 0:
            no_labels = 0

        overlap_bool = 0
        assert len(text_fragment_list) == len(techniques)

        # Case 44 in training set
        # \n was labelled as Reductio ad hitlerum and when cleaning was done it changed to ''
        if '' in text_fragment_list:
            temp_fragment_list = [text_fragment_list[i] for i, txt in enumerate(text_fragment_list) if txt != '']
            techniques = [techniques[i] for i, txt in enumerate(text_fragment_list) if txt != '']
            text_fragment_list = temp_fragment_list
            assert len(text_fragment_list) == len(techniques)

        # Case 40 in val_set
        # Because the repeated words were so many the indexing of the repeated words was going all wrong and was creating infinite combinations
        if 'Repetition' in techniques:
            text_fragment_list = [text_fragment_list[i] for i, tech in enumerate(techniques) if tech == 'Repetition']
            techniques = ['Repetition']*len(text_fragment_list)
            return [text_fragment_list], [techniques], [text], [overlap_bool], [index], [no_labels]

        # Get list of tuples of indices
        index_tuples = []
        for tf in text_fragment_list:
            try:
                start_index = text.index(tf)
            except:
                raise Exception("No substring found!")
            end_index = start_index + len(tf)
            index_tuples.append((start_index, end_index))
        
        assert len(index_tuples) == len(text_fragment_list)

        # Get indices of overlapping fragments
        overlap_dict = dict()
        for i, tf in enumerate(text_fragment_list):
            for j, ind in enumerate(index_tuples):
                if i == j: continue
                else:
                    if self.isOverlap(index_tuples[i], ind):
                        try:
                            overlap_dict[i].append(j)
                        except:
                            overlap_dict[i] = [j]

        # Get non-overlapping indices combinations
        overlap_list = self.convertToListOfTuples(overlap_dict)
        overlap_list = self.sortElementsInTuples(overlap_list)
        overlap_list = self.removeDuplicates(overlap_list)
        if len(overlap_list) != 0:
            max_value = self.getMaxValue(overlap_list)
            numbered_array = list(range(0, max_value+1))
            tupleCombinations = self.getTupleCombinations(numbered_array, max_value)
            non_overlap = self.getNonOverlaps(tupleCombinations, overlap_list, max_value)
        else:
            non_overlap = []

        # Get new non overlapping example
        non_overlap_fragments, non_overlap_techniques = [], []
        if len(non_overlap) != 0:
            overlap_bool = 1
            for example in non_overlap:
                new_fragments, new_techniques = [], []
                for i in example:
                    new_fragments.append(text_fragment_list[i])
                    new_techniques.append(techniques[i])
                non_overlap_fragments.append(new_fragments)
                non_overlap_techniques.append(new_techniques)

            assert len(non_overlap_fragments), len(non_overlap_techniques)
            return non_overlap_fragments, non_overlap_techniques, [text]*len(non_overlap_fragments), [overlap_bool]*len(non_overlap_fragments), [index]*len(non_overlap_fragments), [no_labels]*len(non_overlap_fragments)

        return [text_fragment_list], [techniques], [text], [overlap_bool], [index], [no_labels]

    def get_non_overlap_df(self, df):
        """
        Get the new non overlapping dataframe.
        """
        FRAGMENT, TECHNIQUE, TEXT, OVERLAP, INDEX, LABEL = [], [], [], [], [], []

        for i, f in df.iterrows():

            frag, tech, txt, overlap, index, labels  = self.check_overlap(i,f)
            FRAGMENT += frag
            TECHNIQUE += tech
            TEXT += txt
            OVERLAP += overlap
            INDEX += index
            LABEL += labels


        df_x = pd.DataFrame()
        df_x['Fragment'] = pd.Series(FRAGMENT)
        df_x['Technique'] = pd.Series(TECHNIQUE)
        df_x['Text'] = pd.Series(TEXT)
        df_x['Overlap'] = pd.Series(OVERLAP)
        df_x['Index'] = pd.Series(INDEX)
        df_x['Label'] = pd.Series(LABEL)

        return df_x

    def run(self,):
        
        dataframes_split = self.read_json_files_to_df()
        train_df, val_df, test_df = dataframes_split

        print('Saving JSON files as Non-Overlapping CSV files.')
        for i, df_ in enumerate(dataframes_split):
            filename = self.files[i].split('/')[-1].split('.')[0]
            print('Saving {}...'.format(filename))
            if i != 2:
                df_final = self.get_non_overlap_df(df_)
            else:
                df_final = pd.DataFrame()
                df_final['Fragment'] = df_['text_fragment']
                df_final['Technique'] = df_['technique']
                df_final['Text'] = df_['text']
            save_filename = self.paths['Meme_Data'] + filename + '.csv'
            df_final.to_csv(save_filename)

        print('Saved successfully!')



if __name__ == '__main__':
    paths = {
        "Meme_Data": "./Meme_Data/",
        "Meme_Data_Train_Json": "./Meme_Data/training_set_.json",
        "Meme_Data_Val_Json": "./Meme_Data/dev_set_.json",
        "Meme_Data_Test_Json": "./Meme_Data/test_set_.json",
        "Meme_Data_Train":"./Meme_Data/training_set_.csv",
        "Meme_Data_Val":"./Meme_Data/dev_set_.csv",
        "Meme_Data_Test":"./Meme_Data/test_set_.csv",
        "Techniques":"./techniques.json",
        "Log_Folder":"./Log_Files/"
    }
    
    dataRaw = Dataset_Preparation(paths)
    dataRaw.run()