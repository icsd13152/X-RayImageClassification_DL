import base64
import datetime
import io
import numpy as np
import dash
import os
from PIL import Image
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.express as px
import cv2
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
model = load_model('myCNNmodel_v11.h5')
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True)

def process(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (70, 70)) / 255.0
    x = np.array(image)
    return x

app.layout = html.Div([ # this code section taken from Dash docs https://dash.plotly.com/dash-core-components/upload
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-div'),
    html.Div(id='output-image'),
])


def parse_contents(contents, filename, date):
    # Remove 'data:image/png;base64' from the image string,
    encoded_image = base64.b64encode(open(os.getcwd() +'\\'+filename, 'rb').read())
    image = cv2.imread(encoded_image)
    print(image)
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents),
        html.Hr(),
        html.Div('Prediction'),

    ])

@app.callback(Output('output-image', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


# @app.callback(Output('output-div', 'children'),
#               Input('Predict','n_clicks'),
#               Input('output-image','filename'))
#               # State('stored-data','data'))
# def make_prediction(n, filename):
#     if n is None:
#         return dash.no_update
#     else:
#         image = process(filename)
#         prediction = model.predict(image)
#         y = np.argmax(prediction, axis = 1)
#         # print(y)
#
#         case = ""
#
#         if y[0] == 0:
#             case = "Normal"
#         elif y[0] == 1:
#             case = "Covid"
#         elif y[0] == 2:
#             case = "Pneumonia"
#         else:
#             case = "Lung Opacity"
#
#         out = (f'I am {prediction[0][y][0]:.2%} percent confirmed that this is a {case} case')
#         plot = plt.imshow(np.squeeze(image))
#         return plot



if __name__ == '__main__':
    app.run_server(debug=True)