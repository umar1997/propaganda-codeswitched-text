# detection/forms.py
from flask_wtf import FlaskForm
from wtforms import SubmitField, TextAreaField
from wtforms.validators import DataRequired


class DetectForm(FlaskForm):
    
    text = TextAreaField(label='Detect Propaganda Technique For Your Sentence', validators=[DataRequired()])
    submit = SubmitField('Detect')