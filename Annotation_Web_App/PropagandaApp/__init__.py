# PropagandaApp/__init__.py
import os
from flask import Flask

app = Flask(__name__) # PropagandaApp

app.config['SECRET_KEY'] = 'random_secret_key'

# __file__      /home/umar/Desktop/Thesis/Propaganda_App/PropagandaApp/__init__.py
basedir = os.path.abspath(os.path.dirname(__file__))
# basedir       /home/umar/Desktop/Thesis/Propaganda_App/PropagandaApp

#################################### Model
import json
import torch

from transformers import AutoTokenizer
from PropagandaApp.detection.model import Propaganda_Detection
from PropagandaApp.detection.inference import Predict

with open("./PropagandaApp/config.json", "r") as f:
    config = json.load(f)
with open(config["id2techniques"], "r") as fp:
    id2techniques = json.load(fp)

# Comment these to stop loading model
model = Propaganda_Detection(checkpoint_model=config["checkpoint_model"], num_tags=len(id2techniques)+1)
model.load_state_dict(torch.load(config[ "model_path"]))
tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])

predict = Predict(config, model, tokenizer)


##################################### Blueprints
from PropagandaApp.core.views import core
from PropagandaApp.detection.views import detection
from PropagandaApp.annotation.views import annotation

app.register_blueprint(core)
app.register_blueprint(detection)
app.register_blueprint(annotation)