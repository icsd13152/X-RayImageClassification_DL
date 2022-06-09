import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import cv2
import numpy as np
import base64
from dash.dependencies import (Input, Output, State)
from tensorflow import keras
from IPython.display import Image, display
import matplotlib.cm as cm
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D,BatchNormalization

def createNeuralNet():
    model = Sequential()    #128

    model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', input_shape = (70, 70, 3)))
    # model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))

    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    #64
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    #64
    model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu'))
    model.add(Dropout(0.3)) #0.5 without cw
    model.add(Flatten())
    model.add(Dense(1024, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dropout(0.2))
    # model.add(BatchNormalization())
    model.add(Dense(4,activation = 'softmax')) #for Covid/Normal/Pneumonia/Lung_opacity maybe we need activation softmax

    model.load_weights('myCNNmodel_W_v13.h5')

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),#learning_rate=0.001
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(),
                  # run_eagerly=True,
                  # update_weights=True,
                  )

    # model.summary()
    return model

# model = load_model('myCNNmodel_v11.h5')
model = createNeuralNet()
external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css'
]
# image_array = None
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config['suppress_callback_exceptions'] = True
app.title = 'X-Ray Classification App'
server = app.server

image_ops = ['None', 'Equalize', 'Flip', 'Mirror', 'Binarize', 'Invert', 'Solarize']
image_morphs = ['None', 'Erode', 'Dilate', 'Open', 'Close', 'Gradient', 'Boundary Extraction']

tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '10px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '5px solid #d6d6d6',
    'borderBottom': '3px solid #d6d6d6',
    'backgroundColor': '#7E8483',
    'color': 'white',
    'padding': '6px'
}

app.layout = html.Div([
    html.Meta(charSet='UTF-8'),
    html.Meta(name='viewport', content='width=device-width, initial-scale=1.0'),

    html.Div([
        html.Div(
            id='title-app',
            children=[
                html.H3(app.title)
            ],
            style={'textAlign' : 'center', 'paddingTop' : 30}
        ),
        html.Div([
            dcc.Upload(
                id='upload-image',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '70px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px',
                    'backgroundColor': '#F0F1F1'
                },
                multiple=True
            ),
        ], style={'paddingTop' : 50}),

    ], className='flex-item-left'),

    html.Div(id='result-in-out-image', className='flex-item-right'),
    # html.Div(id='result-predict', className='flex-item-right'),

], className='flex-container')


# app.layout = html.Div([
#     html.Meta(charSet='UTF-8'),
#     html.Meta(name='viewport', content='width=device-width, initial-scale=1.0'),
#
#     html.Div([
#         html.Div(
#             id='title-app',
#             children=[
#                 html.H3(app.title)
#             ],
#             style={'textAlign' : 'center', 'paddingTop' : 30}
#         ),
#         html.Div([
#             dcc.Upload(
#                 id='upload-image',
#                 children=html.Div([
#                     'Drag and Drop or ',
#                     html.A('Select Files')
#                 ]),
#                 style={
#                     'width': '100%',
#                     'height': '70px',
#                     'lineHeight': '60px',
#                     'borderWidth': '1px',
#                     'borderStyle': 'dashed',
#                     'borderRadius': '5px',
#                     'textAlign': 'center',
#                     'margin': '10px',
#                     'backgroundColor': '#F0F1F1'
#                 },
#                 multiple=True
#             )
#             ]),
#         ], style={'paddingTop' : 50}),
#         html.Div(id='result-in-out-image', className='flex-item-right'),
#
#     ], className='flex-container')

def read_image_string(contents):
    encoded_data = contents[0].split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # print(img.shape)
    image = cv2.resize(img,(70,70)) / 255.0
    # print(image.shape)
    # print(image)
    x = np.expand_dims(image,axis=0)
    # x = np.array(image)
    # print(x.shape)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return x,img

def parse_contents(contents, filename, date):
    image_mat = read_image_string(contents=contents)[1]
    image_array = read_image_string(contents=contents)[0]
    return image_mat,image_array


@app.callback(
    Output('result-in-out-image', 'children'),
    Input('upload-image', 'contents')
)
def set_output_layout(contents):

    in_out_image_div = html.Div([
            html.Div(
                children= [
                    html.H5('Image Used - Output'),
                    html.Div(id='output-image-op'),
                    # html.Div([
                    #     html.Button('Predict', id='prediction',n_clicks=0)
                    # ])
                ],
                style={'textAlign' : 'center', 'paddingTop' : 50}
            )
        ])

    return in_out_image_div

def predict(image_array):
    prediction = model.predict(image_array)
    y = np.argmax(prediction, axis = 1)
    # print(y)
    case = ""

    if y[0] == 0:
        case = "Normal"
    elif y[0] == 1:
        case = "Covid"
    elif y[0] == 2:
        case = "Pneumonia"
    else:
        case = "Lung Opacity"

    out = (f'I am {prediction[0][y][0]:.2%} percent confirmed that this is a {case} case')
    # plot = plt.imshow(np.squeeze(image_array))
    return prediction,out




def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(image_array, heatmap, cam_path="cam.jpg", alpha=0.5):
    # Load the original image
    # img = keras.preprocessing.image.load_img(img_path)
    # img = keras.preprocessing.image.img_to_array(img)
    img = image_array
    # img = img_path # kmouts

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    # display(Image(cam_path))
    return Image(cam_path),superimposed_img
@app.callback(
    Output('output-image-op', 'children'),
    [
        Input('upload-image', 'contents'),
        # Input('submit-val', 'n_clicks'),

        # -------
        State('upload-image', 'filename'),
        State('upload-image', 'last_modified'),
    ]
)
def get_uploaded_image(contents, filenames, dates):
    if contents is not None:
        imsrc,imgarr = parse_contents(contents, filenames, dates)
        pred = predict(imgarr)[1]
        probs = predict(imgarr)[0]
        heatmap = make_gradcam_heatmap(imgarr, model, "conv2d_2",0 )
        plot = save_and_display_gradcam(np.squeeze(imsrc),heatmap)[1]
        print(plot)
        print(probs[0])
        # df = pd.DataFrame(columns=['Confidence','Classes'])
        # df['Confidence'].append([probs[0][0],probs[0][1],probs[0][2],probs[0][3]],ignore_index=True)
        # df['Classes'].append(['Normal','Covid','Pneumonia','Lung Opacity'],ignore_index=True)
        # print(df.head())
        out_image_fig = px.imshow(imsrc, color_continuous_scale='gray')
        out_image_fig.update_layout(
            coloraxis_showscale=False,
            width=600, height=400,
            margin=dict(l=0, r=0, b=0, t=0)
        )
        out_image_fig.update_xaxes(showticklabels=False)
        out_image_fig.update_yaxes(showticklabels=False)
        #[probs[0][0],probs[0][1],probs[0][2],probs[0][3]]
        out_image_figBar = px.bar(x=probs[0], y= ['Normal','Covid','Pneumonia','Lung Opacity'])
        fig =  px.imshow(plot)
        output_result = html.Div([
            html.H5(pred),
            dcc.Graph(id='out-op-img', figure=out_image_fig,style={'display': 'inline-block'}),
            dcc.Graph(id='out-op-img', figure=out_image_figBar,style={'display': 'inline-block'}),
            dcc.Graph(id='out-op-img', figure=fig),
        ], style={'paddingTop' : 50})

        return output_result



@app.callback(
    Output('result-predict', 'children'),
    Input('submit-val', 'n_clicks')
    # State('input-on-submit', 'value')
)
def update_output(n_clicks, value):
    if n_clicks > 0:
        print(n_clicks)
    pred = predict()[1]
    print(pred)
    output_result = html.Div([
        dcc.Graph(id='out-pred-img', figure=pred)

    ], style={'paddingTop' : 50})
    return output_result


if __name__ == '__main__':
    app.run_server(debug=False)