# PropagandaApp/core/views
from flask import render_template, request, Blueprint, flash, redirect, url_for

import json
import operator
import pandas as pd

annotation = Blueprint('annotation',__name__)

@annotation.route('/annotate', methods=['GET','POST'])
def annotate_text():

    dataset_filename = './PropagandaApp/Files/Dataset.csv'
    df_dataset = pd.read_csv(dataset_filename)

    with open('./PropagandaApp/detection/colour2techniques.json') as f:
        techniques = json.load(f)
    
    length = int(len(techniques)/2)

    techniques_ = [v for k,v in techniques.items()]
    color_ = [k for k,v in techniques.items()]

    techniques_1 = techniques_[:length]
    color_1 = color_[:length]
    length_1 = length

    techniques_2 = techniques_[length:]
    color_2 = color_[length:]
    length_2 = length

    count = len(df_dataset[df_dataset['INCLUDED'] == 0])
    count = str(count) + " / " +str(len(df_dataset))

    def check_if_frag_in_text(fragment, sentence):
        if fragment in sentence:
            return True
        else:
            return False

    def sort_labels(jsonObj):
        labels = jsonObj["labels"]
        labels.sort(key=operator.itemgetter('start_index'), reverse=False)
        jsonObj["labels"] = labels
        return jsonObj


    # Exclude or Include Sample
    if request.method == 'POST' and "includeSample" in request.form:
        package = request.form["includeSample"]
        colourBox, id_value= package.split("#####")

        indexPosition = df_dataset.index[df_dataset['ID'] == int(id_value)].tolist()
        assert len(indexPosition) == 1
        indexPosition = indexPosition[0]

        colourInclusion = None
        if colourBox == "WHITE2RED":
            df_dataset.at[indexPosition, 'INCLUDED'] = 1
            colourInclusion = "#FF0000"
        elif colourBox == "RED2WHITE":
            df_dataset.at[indexPosition, 'INCLUDED'] = 0
            colourInclusion = "#FFFFFF"
        else:
            raise Exception("Colour is neither WHITE2RED nor RED2WHITE.")

        df_dataset.to_csv(dataset_filename ,index=False)

        return colourInclusion




    # From ID get the Example ID and Text
    if request.method == 'POST' and "IDInformation" in request.form:
        id_value = request.form["IDInformation"]
        if (int(id_value) == 0) or (int(id_value) > len(df_dataset)): # When previous and next cross boundaries
            return json.dumps(dict(id='', text='', labels= 'Empty'))


        text = df_dataset.loc[df_dataset['ID'] == int(id_value), 'TEXT'].values[0]
        indexPosition = df_dataset.index[df_dataset['ID'] == int(id_value)].tolist()
        assert len(indexPosition) == 1
        indexPosition = indexPosition[0]

        colourBox = df_dataset.loc[df_dataset['ID'] == int(id_value), 'INCLUDED'].values[0]
        assert (colourBox == 1) or (colourBox == 0)
        if colourBox == 0:
            backgroundColour = "#FFFFFF"
        else:
            backgroundColour = "#FF0000"

            
        jsonLabels = df_dataset.loc[df_dataset['ID'] == int(id_value), 'LABELS'].values[0]
        try: # In case where there is no label
            jsonLabels = eval(jsonLabels)
        except:
            jsonLabels = dict(id=id_value, text=text, labels= [])
        jsonLabels['colourBackground'] = backgroundColour

        return json.dumps(jsonLabels)


    # Undo Annotation
    if request.method == 'POST' and "undoAnnotation" in request.form:
        id_value = request.form["undoAnnotation"]
        text = df_dataset.loc[df_dataset['ID'] == int(id_value), 'TEXT'].values[0]

        indexPosition = df_dataset.index[df_dataset['ID'] == int(id_value)].tolist()
        assert len(indexPosition) == 1
        indexPosition = indexPosition[0]

        jsonLabels = dict(id=id_value, text=text, labels= [])
        df_dataset.at[indexPosition, 'LABELS'] = jsonLabels
        df_dataset.to_csv(dataset_filename ,index=False)

        df_dataset_read = pd.read_csv(dataset_filename)
        jsonLabels = df_dataset_read.loc[df_dataset_read['ID'] == int(id_value), 'LABELS'].values[0]
        jsonLabels = eval(jsonLabels)

        return json.dumps(jsonLabels)

    
    # Update Original Text
    if request.method == 'POST' and "updateText" in request.form:
        package = request.form["updateText"]
        new_text, id_value= package.split("#####")
        new_text = str(new_text)

        old_text  = df_dataset.loc[df_dataset['ID'] == int(id_value), 'TEXT'].values[0]
        if new_text == old_text:
            return json.dumps(dict(id='', text='', labels= 'Empty'))

        
        indexPosition = df_dataset.index[df_dataset['ID'] == int(id_value)].tolist()
        assert len(indexPosition) == 1
        indexPosition = indexPosition[0]

        jsonLabels = dict(id=id_value, text=new_text, labels= [])
        df_dataset.at[indexPosition, 'LABELS'] = jsonLabels
        df_dataset.at[indexPosition, 'TEXT'] = new_text
        df_dataset.to_csv(dataset_filename ,index=False)

        return json.dumps(jsonLabels)



    # Annotate Selected Text
    if request.method == 'POST' and "annotateInfo" in request.form:
        package = request.form["annotateInfo"]
        id_value, technique_id, fragment = package.split("#####")
        text = df_dataset.loc[df_dataset['ID'] == int(id_value), 'TEXT'].values[0]

        no_error = check_if_frag_in_text(fragment, text)
        if not no_error:
            return json.dumps(dict(id='', text='', labels= 'Empty'))


        indexPosition = df_dataset.index[df_dataset['ID'] == int(id_value)].tolist()
        assert len(indexPosition) == 1
        indexPosition = indexPosition[0]

        jsonLabels = df_dataset.loc[df_dataset['ID'] == int(id_value), 'LABELS'].values[0]
        try: # In case where there is no label
            jsonLabels = eval(jsonLabels)
        except:
            jsonLabels = dict(id=id_value, text=text, labels= [])


        start_index = text.find(fragment)
        end_index = start_index + len(fragment)
        assert text[start_index:end_index] == fragment

        jsonLabels['labels'] += [{'start_index': str(start_index), 'end_index': str(end_index), 'text_fragment': fragment, 'technique':technique_id}]
        jsonLabels = sort_labels(jsonLabels)
        df_dataset.at[indexPosition, 'LABELS'] = jsonLabels
        df_dataset.to_csv(dataset_filename ,index=False)

        df_dataset_read = pd.read_csv(dataset_filename)
        jsonLabels = df_dataset_read.loc[df_dataset_read['ID'] == int(id_value), 'LABELS'].values[0]
        jsonLabels = eval(jsonLabels)
        
        return json.dumps(jsonLabels)

    return render_template('annotate_text.html', techniques_1=techniques_1, color_1=color_1, length_1=length_1, techniques_2=techniques_2, color_2=color_2, length_2=length_2, count=count)