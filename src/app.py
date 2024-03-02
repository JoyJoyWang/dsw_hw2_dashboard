

import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import numpy as np
from flask import Flask, render_template_string

data = pd.read_csv('weather.csv')
data['Ftemp'] = (data['Ktemp'] - 273.15) * (9/5) + 32
data['Time'] = pd.to_datetime(data['time'])

yearly_avg_temp = data.groupby(data['Time'].dt.year)['Ftemp'].mean()

preci_data = {
"Year": list(range(1950, 2022)),
"Annual Precipitation (inches)": [
    36.89, 44.40, 41.51, 45.20, 35.58, 39.90, 36.25, 36.49, 40.94, 38.77, # 1950-1959
    46.39, 39.32, 37.15, 34.28, 32.99, 26.09, 39.90, 49.12, 43.57, 48.54, # 1960-1969
    35.29, 56.77, 67.03, 57.23, 47.69, 61.21, 41.28, 54.73, 49.81, 52.13, # 1970-1979
    44.55, 38.11, 41.40, 80.56, 57.03, 38.82, 42.95, 46.39, 44.67, 65.11, # 1980-1989
    60.92, 45.18, 43.35, 44.28, 47.39, 40.42, 56.19, 43.93, 48.69, 42.50, # 1990-1999
    45.42, 35.92, 45.21, 58.56, 51.97, 55.97, 59.89, 61.67, 53.61, 53.62, # 2000-2009
    49.37, 72.81, 38.51, 46.32, 53.79, 40.97, 42.17, 45.04, 65.55, 53.03, # 2010-2019
    45.35, 59.73 # 2020-2021
]
}

preci_data = pd.DataFrame(preci_data)


app = dash.Dash(__name__)
server=app.server


app.layout = html.Div([
    html.H1("Interactive Temperature Analysis"),
    dcc.Graph(id='temperature-plot'),
    html.Label("Enter Threshold Temperature (°F):"),
    dcc.Input(id='threshold-input', type='number', value=55, min=0, step=1),
    dcc.Graph(id='average-temperature-plot'),
    dcc.Graph(id='precipitation-plot')
])


@app.callback(
    [Output('temperature-plot', 'figure'),
     Output('average-temperature-plot', 'figure'),
     Output('precipitation-plot', 'figure')],
    [Input('threshold-input', 'value')]
)
def update_plot(threshold):

    # ----------------------------- fig 1 -----------------------------
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Time'], y=data['Ftemp'], mode='lines', name='Temperature'))
    fig.update_layout(
        title='Line Plot of Temperature',
        xaxis=dict(title='Time'),
        yaxis=dict(title='Ftemp (Fahrenheit)'),
        xaxis_tickangle=-45,
        showlegend=True
    )
    # bar
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )


    # ----------------------------- annual fig -----------------------------
    avg_temp_fig = go.Figure()
    avg_temp_fig.add_trace(go.Scatter(x=yearly_avg_temp.index, y=yearly_avg_temp.values, mode='lines', name='Average Temperature'))
    # add threshold on plot
    avg_temp_fig.add_shape(type="line",
                            x0=yearly_avg_temp.index.min(), y0=threshold,
                            x1=yearly_avg_temp.index.max(), y1=threshold,
                            line=dict(color="red", width=2, dash="dash"),
                            name='Threshold')
    # highlight years
    for year, avg_temp in yearly_avg_temp.items():
        if avg_temp > threshold:
            avg_temp_fig.add_annotation(x=year, y=avg_temp, text=str(year), showarrow=True, arrowhead=1)
    avg_temp_fig.update_layout(
        title='Average Temperature Each Year',
        xaxis=dict(title='Year'),
        yaxis=dict(title='Ftemp (Fahrenheit)'),
        showlegend=True
    )

#     # ----------------------------- fig 3 -----------------------------
#     preci_fig = px.line(df, x="Year", y="Annual Precipitation (inches)", title="Annual Precipitation at Central Park (1950-2022)",
#                   labels={"Annual Precipitation (inches)": "Precipitation (inches)"})
#     preci_fig.update_layout(
#         xaxis=dict(
#             rangeselector=dict(
#                 buttons=list([
#                     dict(count=1, label="1y", step="year", stepmode="backward"),
#                     dict(count=5, label="5y", step="year", stepmode="backward"),
#                     dict(count=10, label="10y", step="year", stepmode="backward"),
#                     dict(step="all")
#                 ])
#             ),
#             rangeslider=dict(
#                 visible=True
#             ),
#             type="date"
#         )
#     )

    # ----------------------------- fig 3 -----------------------------
    preci_fig = go.Figure()
    preci_fig.add_trace(go.Scatter(x=preci_data['Year'], y=preci_data['Annual Precipitation (inches)'], mode='lines', name='Annual Precipitation'))
    preci_fig.add_trace(go.Scatter(x=yearly_avg_temp.index, y=yearly_avg_temp.values, mode='lines', name='Average Temperature'))
    # z = np.polyfit(preci_data['Annual Precipitation (inches)'], yearly_avg_temp.values, 1)
    # p = np.poly1d(z)
    # preci_fig.add_trace(go.Scatter(x=preci_data['Annual Precipitation (inches)'], y=p(preci_data['Annual Precipitation (inches)']), mode='lines', name='Trendline'))

    preci_fig.update_layout(
        title='Annual Precipitation in New York',
        xaxis=dict(title='Year'),
        yaxis=dict(title='Precipitation (inches)'),
        showlegend=True
    )
    preci_fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )


    return fig, avg_temp_fig,preci_fig


if __name__ == '__main__':
    app.run_server(debug=True)
    correlation = np.corrcoef(preci_data['Annual Precipitation (inches)'], yearly_avg_temp.values)[0, 1]
    print("Correlation coefficient:", correlation)

from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

# normalization
scaler = StandardScaler()
scaled_precipitation = scaler.fit_transform(preci_data['Annual Precipitation (inches)'].values.reshape(-1, 1))
scaled_temperature = scaler.fit_transform(yearly_avg_temp.values.reshape(-1, 1))

# calculate cor & p
correlation, p_value = pearsonr(scaled_precipitation.flatten(), scaled_temperature.flatten())

print("Correlation coefficient:", correlation)
print("P value:", p_value)


