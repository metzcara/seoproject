import numpy as np
import xgboost as xgb
from dash import Dash, html, dcc, Input, Output, callback
import plotly.express as px

from sklearn.preprocessing import StandardScaler 

model = xgb.Booster()
model.load_model('/Users/avery/seoproject/pcamse0653.json') # path to saved xgboost model

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Receive median background value prediction."),
    html.H6("created by Avery Metzcar"),
    html.Div([
        "Temperature (°F): ",
        dcc.Input(id='temp', value=0, type='number')
    ]),
    html.Div([
        "Relative Humidity (%): ",
        dcc.Input(id='rh', value=0, type='number')
    ]),
    html.Div([
        "Wind Speed (mph): ",
        dcc.Input(id='wspd', value=0, type='number')
    ]),
    html.Div([
        "Moon Angle (deg): ",
        dcc.Input(id='mang', value=0, type='number')
    ]),
    html.Br(),
    html.Div("Median Background Prediction: ",id='pred')

])


@callback(
    Output(component_id='pred', component_property='children'),
    Input(component_id='temp', component_property='value'),
    Input(component_id='rh', component_property='value'),
    Input(component_id='wspd', component_property='value'),
    Input(component_id='mang', component_property='value')
)
def get_med_pred(input1,input2,input3,input4):
    y = [input1,input2,input3,input4]
    y = np.asarray(y)
    y = y.reshape(1, -1)
    y = StandardScaler().fit_transform(y)
    pred = model.predict(xgb.DMatrix(y))

    return f'Output: {pred[0]}'


if __name__ == '__main__':
    app.run(debug=True)


