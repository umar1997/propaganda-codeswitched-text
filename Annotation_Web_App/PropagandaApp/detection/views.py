# PropagandaApp/core/views
from flask import render_template, request, Blueprint, flash, redirect, url_for

from PropagandaApp import *
from PropagandaApp.detection.forms import DetectForm

import json
import random
import pandas as pd

detection = Blueprint('detection',__name__)

@detection.route('/detect', methods=['GET','POST'])
def detect_propaganda():

  
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

    # Another way
    # https://www.digitalocean.com/community/tutorials/how-to-use-web-forms-in-a-flask-application
    # https://getbootstrap.com/docs/4.0/components/forms/
    # Getting the model to reurn the prediction with responding colours
    if request.method == 'POST' and "result" in request.form:
        result = request.form["result"]
        final_output = predict.get_sentence_prediction(result)
        return json.dumps(final_output)
    
    # Getting the random sentence in the text box
    if request.method == 'POST' and "sentence" in request.form:
        sentence = request.form["sentence"] # Sentence = "random_sample"
        df_samples = pd.read_csv('./PropagandaApp/Files/Sample.csv')
        sampleList = list(df_samples['Text'])
        inputValue = random.choice(sampleList)

        return inputValue

    return render_template('detect_propaganda.html', techniques_1=techniques_1, color_1=color_1, length_1=length_1, techniques_2=techniques_2, color_2=color_2, length_2=length_2)