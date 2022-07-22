import json
from textwrap import dedent as d

from db.sqlCont import getPostgresConn
from db.queries import getAllStations, getPollutantFromDateRange

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import numpy as np
from geojson import Point
from db.names import findTable
from datetime import date
from datetime import date, timedelta
from datetime import datetime as dt

import geopandas as pd

# https://dash.plot.ly/interactive-graphing
# https://plot.ly/python-api-reference/

COLORS = ['r', 'g', 'b', 'y']

# plot_date_format = "%b %d %H hrs"
plot_date_format = "%l %p, %b %d"
app = dash.Dash(__name__)

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

conn = getPostgresConn()
stations_geodf = getAllStations(conn)
pollutant = "PM2.5"
default_date = date.today()
table = findTable(pollutant)
default_date = date.today()
default_stations = [x for x in stations_geodf.index.values]
time_range = 1

app.layout = html.Div([
    html.Div([
        dcc.DatePickerSingle(
            id='calendar',
            date=dt.today().strftime("%Y-%m-%d")
        ),
        html.Button(
            '-',
            id='prevday',
        ),
        html.Button(
           '+',
            id='nextday',
        ),
    ]),

    dcc.Graph(
        id="id-map",
        figure=dict(
            data=[
                dict(
                    lat=[pt.y for pt in stations_geodf.geometry],
                    lon=[pt.x for pt in stations_geodf.geometry],
                    text=stations_geodf['nombre'].values,
                    type="scattermapbox",
                    customdata=stations_geodf.index.values,
                    # https://plot.ly/python-api-reference/generated/plotly.graph_objects.Scattermapbox.html
                    # fill="none", # none, toself, (only toself is working
                    marker=dict(
                        color='blue'
                    ),
                    hovertemplate='Station: %{text} <extra></extra>'
                )
            ],
            layout=dict(
                mapbox=dict(
                    layers=[],
                    center=dict(
                        lat=19.4, lon=-99.14
                    ),
                    style='carto-positron',
                    # open-street-map, white-bg, carto-positron, carto-darkmatter,
                    # stamen-terrain, stamen-toner, stamen-watercolor
                    pitch=0,
                    zoom=8,
                ),
               autosize=True,
            )
        )
    ),

    html.Div(className='row', children=[
        html.Div([
            dcc.Markdown("**Ozone values for**"),
            html.Span(id='hover-data'),
            dcc.Graph(
                id='station-plot',
                figure={}),
            dcc.Graph(
                id='anomaly-plot',
                figure={}),
        ]),
    ])
])

@app.callback(
    Output('calendar', 'date'),
    [Input('prevday', 'n_clicks_timestamp'),
    Input('nextday', 'n_clicks_timestamp')],
    [State('calendar', 'date')])
def change_day(prev_day, next_day, cur_date):
    new_date = dt.strptime(cur_date.split(' ')[0], '%Y-%m-%d')
    if (prev_day is not None) and (next_day is None):
        new_date = new_date + timedelta(days=-1)
    if (next_day is not None) and (prev_day is None):
            new_date = new_date + timedelta(days=1)
    if (next_day is not None) and (prev_day is not None):
        if next_day > prev_day:
            new_date = new_date + timedelta(days=+1)
        else:
            new_date = new_date + timedelta(days=-1)
    return new_date.strftime("%Y-%m-%d")


@app.callback(
    Output('hover-data', 'children'),
    [Input('id-map', 'hoverData')])
def display_figure_title(hoverData):
    if hoverData is not None:
        selected_stations = hoverData['points'][0]['customdata']
    else:
        selected_stations = default_stations

    name = stations_geodf.loc[selected_stations]['nombre']
    return html.P(name)

@app.callback(
    [Output('station-plot', 'figure'),
    Output('anomaly-plot', 'figure')],
    [Input('id-map', 'clickData'),
     Input('calendar', 'date'),
     Input('id-map', 'selectedData')])
def display_figure_plot(hoverData, cur_date, selectedData):
    if cur_date is None:
        return getEmptyFigure(''), getEmptyFigure('')

    if hoverData is None:
        selected_stations = default_stations
    else:
        selected_stations = [hoverData['points'][0]['customdata']]

    if selectedData is not None:
        selected_stations = [x['customdata'] for x in selectedData['points']]

    cur_date = dt.strptime(cur_date.split(' ')[0], '%Y-%m-%d')
    start_date = cur_date - timedelta(days=time_range)
    end_date = cur_date + timedelta(days=time_range)
    stations_data = pd.GeoDataFrame(getPollutantFromDateRange(conn, table, start_date, end_date, selected_stations),
                                   columns=['date', 'value', 'id'])

    name = stations_geodf.loc[selected_stations]['nombre']
    if stations_data.shape[0] == 0:
        return getEmptyFigure(name), getEmptyFigure(name)
    else:
        figure = {
            'data': getData(stations_data),
            'layout': {
                'title': F'{pollutant} around {cur_date.strftime("%B %d, %Y")}',
                'clickmode': 'event+select',
                'legend': {
                    'x': 1,
                    'y': 1
                },
            }
        }
        figure_anomaly = {
            'data': getDataAnomaly(stations_data),
            'layout': {
                'title': F'{pollutant} around {cur_date.strftime("%B %d, %Y")}',
                'clickmode': 'event+select',
                'legend': {
                    'x': 1,
                    'y': 1
                },
            }
        }
        return figure, figure_anomaly

def getDataAnomaly(stations_data):
    all_stations = stations_data['id'].unique()
    data = []
    all_dates = stations_data.sort_values(by=['date'])['date'].unique()
    composedData = pd.GeoDataFrame(index=all_dates)
    # TODO improve this hack to get nulls on the rows where there is no data
    for cur_station in all_stations:
        tempSeries = stations_data[stations_data['id'] == cur_station].set_index('date')
        composedData[cur_station] = tempSeries['id']
        composedData[F"{cur_station}_value"] = tempSeries['value']

    mean_data = composedData.mean(axis=1)

    for idx, cur_station in enumerate(all_stations):
        dates_station = composedData[cur_station].index
        dates_str = [x.strftime(plot_date_format) for x in dates_station]
        values = composedData[F"{cur_station}_value"] - mean_data
        data.append({
            'x': dates_str,
            'y': values,
            'name': stations_geodf.loc[cur_station]['nombre'],
            'type': 'line',
            'line_shape': 'spline',
            'mode': 'lines+markers'
        })

    return data

def getData(stations_data):
    all_stations = stations_data['id'].unique()
    data = []
    all_dates = stations_data.sort_values(by=['date'])['date'].unique()
    composedData = pd.GeoDataFrame(index=all_dates)
    # TODO improve this hack to get nulls on the rows where there is no data
    for cur_station in all_stations:
        tempSeries = stations_data[stations_data['id'] == cur_station].set_index('date')
        composedData[cur_station] = tempSeries['id']
        composedData[F"{cur_station}_value"] = tempSeries['value']

    for idx, cur_station in enumerate(all_stations):
        dates_station = composedData[cur_station].index
        dates_str = [x.strftime(plot_date_format) for x in dates_station]
        values = composedData[F"{cur_station}_value"]
        data.append({
            'x': dates_str,
            'y': values,
            'name': stations_geodf.loc[cur_station]['nombre'],
            'type': 'line',
            'line_shape': 'spline',
            'mode': 'lines+markers'
        })

    mean_data = composedData.mean(axis=1)
    all_dates_str = [x.strftime(plot_date_format) for x in composedData.index]
    data.append({
        'x': all_dates_str,
        'y': mean_data,
        'name': 'MEAN',
        'type': 'line',
        'line_shape': 'spline',
        'line': {
            'color': 'black',
            'width': 5,
            'dash': 'dash',
        },
        'mode': 'lines+markers'
    })

    return data


def getEmptyFigure(name):
    figure = {
        'data': [
            {
                'x': [],
                'y': [],
                'name': '',
            },
        ],
        'layout': {
            'title': F'No data for station {name}'
        }
    }
    return figure

if __name__ == '__main__':
    app.run_server(debug=True)
