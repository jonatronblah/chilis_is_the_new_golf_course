from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField
from flask_wtf.file import FileField, FileRequired
from wtforms.validators import ValidationError, DataRequired, InputRequired, Email, EqualTo


class ModelForm(FlaskForm):
    quote = TextAreaField(u'Quote Sample')
    model = FileField(u'model')
    submit = SubmitField(u'Predict', validators=[FileRequired()])