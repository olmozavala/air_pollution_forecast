import json
from constants.AI_params import VisualizationResultsParams, TrainingParams
from conf.TrainingUserConfiguration import get_visualization_config
import dash_bootstrap_components as dbc
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
from viz.layout import get_layout
from viz.figure_generator import get_default_figure

import geopandas as geopd
import pandas as pd

# https://dash.plot.ly/interactive-graphing
# https://plot.ly/python-api-reference/

COLORS = ['r', 'g', 'b', 'y']
date_format = '%Y-%m-%d'
date_format_ext = '%Y-%m-%dT%H:%M:%S'

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

config = get_visualization_config()
run_name = config[TrainingParams.config_name]
print(F"Reading data {config[VisualizationResultsParams.gt_data_file]}....")
gt_data = pd.read_csv(config[VisualizationResultsParams.gt_data_file], index_col = 0, parse_dates = True)
nn_data = pd.read_csv(config[VisualizationResultsParams.nn_output], index_col=0, parse_dates=True)
metrics_data = pd.read_csv(config[VisualizationResultsParams.nn_metrics], index_col=0, parse_dates=True)
print("Done!")

desired_dates = nn_data.index
stations = nn_data.columns

conn = getPostgresConn()
stations_geodf = getAllStations(conn)
pollutant = "O3"
table = findTable(pollutant)
# default_date = date.today()
default_date = desired_dates[0]
default_station = stations[0]
time_range = 1

app = get_layout(default_date, stations_geodf, gt_data, nn_data, run_name)


@app.callback(
    Output('calendar', 'date'),
    [Input('prevday', 'n_clicks_timestamp'),
     Input('nextday', 'n_clicks_timestamp')],
    [State('calendar', 'date')])
def change_day(prev_day, next_day, cur_date):
    try:
        new_date = dt.strptime(cur_date.split(' ')[0], date_format)
    except:
        new_date = dt.strptime(cur_date.split(' ')[0], date_format_ext)

    if (prev_day is not None) and (next_day is None):
        new_date = new_date + timedelta(days=-1)
    if (next_day is not None) and (prev_day is None):
        new_date = new_date + timedelta(days=1)
    if (next_day is not None) and (prev_day is not None):
        if next_day > prev_day:
            new_date = new_date + timedelta(days=+1)
        else:
            new_date = new_date + timedelta(days=-1)
    return new_date

@app.callback(
    [Output('metrics-plot', 'figure'),
     Output('merged-plot', 'figure'),
     Output('nn-plot', 'figure'),
     Output('gt-plot', 'figure'),
     Output('U-plot', 'figure'),
     Output('V-plot', 'figure'),
     Output('RAINC-plot', 'figure'),
     Output('T2-plot', 'figure'),
     Output('max-daily-plot', 'figure'),
     ],
    [Input('id-map', 'clickData'),
     Input('calendar', 'date'),
     Input('id-map', 'selectedData')])
def display_figure(hoverData, cur_date, selectedData):
    if hoverData is None:
        selected_stations = [default_station]
    else:
        selected_stations = [hoverData['points'][0]['customdata']]

    if selectedData is not None:
        selected_stations = [x['customdata'] for x in selectedData['points']]

    try:
        cur_date = dt.strptime(cur_date.split(' ')[0], date_format_ext)
    except:
        cur_date = dt.strptime(cur_date.split(' ')[0], date_format)

    start_date = cur_date - timedelta(days=time_range)
    end_date = cur_date + timedelta(days=time_range)
    name = stations_geodf.loc[selected_stations]['nombre']

    filtered_stations = []
    for selected_station in selected_stations:
        if (selected_station in gt_data.columns) and (selected_station in nn_data.columns):
            filtered_stations.append(selected_station)

    try:
        data_gt = gt_data[start_date:end_date][filtered_stations]
        data_nn = nn_data[start_date:end_date][filtered_stations]
        data_metrics = metrics_data[filtered_stations]
    except Exception as e:
        return getEmptyFigure(name), getEmptyFigure(name), getEmptyFigure(name)

    date_display = "%A, %B %d, %Y"
    figure_metrics = get_default_figure(getMetrics(data_metrics),
                                        title=F'{pollutant} around {cur_date.strftime(date_display)}')
    figure_max_daily = get_default_figure(getDataMaxDaily(gt_data[stations], nn_data[stations]),
                                        title=F'Max {pollutant} for ALL stations ')
    figure_merged =get_default_figure(getDataMerged(data_gt, data_nn),
                                      title=F'{pollutant} around {cur_date.strftime(date_display)}',
                                      range=[0,150])
    figure_nn = get_default_figure(getDataSingle(data_nn, append_txt='NN'),
                                   title= F'NN {pollutant} around {cur_date.strftime(date_display)}',
                                   range=[0, 150])
    figure_gt = get_default_figure(getDataSingle(data_gt, append_txt='GT'),
                                   title=F'GT {pollutant} around {cur_date.strftime(date_display)}',
                                   range=[0, 150])
    start_date = start_date - timedelta(hours=24)
    end_date = end_date - timedelta(hours=24)
    figure_U = get_default_figure(getMeteoData(gt_data[start_date:end_date], 'U'), title='U')
    figure_V = get_default_figure(getMeteoData(gt_data[start_date:end_date], 'V'), title='V')
    figure_RAINC = get_default_figure(getMeteoData(gt_data[start_date:end_date], 'RAINC'), title='RAINC')
    figure_Temp = get_default_figure(getMeteoData(gt_data[start_date:end_date], 'T2'), title='Temperature')

    return figure_metrics, figure_merged, figure_nn, figure_gt, figure_U, figure_V, figure_RAINC, figure_Temp, figure_max_daily


def getMetrics(data):
    all_stations = data.columns
    metric = 'r2'
    metric_idx = data.index.str.find(metric) != -1
    output_data = []
    for cur_station in all_stations:
        values = data[metric_idx][cur_station].values
        names = data[metric_idx].index.values
        output_data.append({
            'x': names,
            'y': values,
            'name': F'{cur_station}',
            'type': 'bar',
        })
    return output_data


def getMeteoData(data, meteo_var):
    all_cols = data.columns
    meteo_columns = [x for x in all_cols if x.find(meteo_var) != -1]

    output_data = []
    values = []
    all_dates = []
    if len(data) > 0:
        all_dates = data.index
        for c_date in all_dates:
            values.append(data.loc[c_date][meteo_columns].mean())

    output_data.append({
            'x': all_dates,
            'y': values,
            'name': F'Name',
            'type': 'line',
            'line_shape': 'spline',
            'mode': 'lines+markers'
        })

    return output_data


def getDataSingle(data, append_txt=''):
    all_stations = data.columns
    output_data = []
    for cur_station in all_stations:
        dates_station = data[cur_station].index
        dates_str = [x.strftime("%b %d %H hrs") for x in dates_station]
        values = data[cur_station].values
        output_data.append({
            'x': dates_str,
            'y': values,
            'name': F'{append_txt} {cur_station}',
            'type': 'line',
            'line_shape': 'spline',
            'mode': 'lines+markers'
        })

    return output_data

def getDataMaxDaily(data_gt, data_nn, append_txt=''):
    output_data = []
    temp_gt_group = data_gt.groupby(by=[data_gt.index.dayofyear])
    temp_nn_group = data_nn.groupby(by=[data_nn.index.dayofyear])
    mean_gt = temp_gt_group.max()
    mean_nn = temp_nn_group.max()
    output_data.append({
        'x': np.unique(data_gt.index.dayofyear),
        'y': mean_gt.max(axis=1),
        'name': F'GT {append_txt}',
        'type': 'line',
        'line_shape': 'spline',
        'mode': 'lines+markers'
    })
    output_data.append({
        'x': np.unique(data_gt.index.dayofyear),
        'y': mean_nn.max(axis=1),
        'name': F'NN {append_txt}',
        'type': 'line',
        'line_shape': 'spline',
        'mode': 'lines+markers'
    })

    return output_data


def getDataMerged(data, nn_data):
    all_stations = data.columns
    output_data = []

    all_dates = np.array(data.index)
    all_dates = np.concatenate((all_dates, np.array(nn_data.index)))

    composedData = pd.DataFrame(index=all_dates.sort())
    # TODO improve this hack to get nulls on the rows where there is no data
    for cur_station in all_stations:
        tempSeries = data[cur_station]
        composedData[cur_station] = tempSeries

    for cur_station in all_stations:
        dates_station = composedData[cur_station].index
        dates_str = [x.strftime("%b %d %H hrs") for x in dates_station]
        values = composedData[cur_station].values
        output_data.append({
            'x': dates_str,
            'y': values,
            'name': cur_station,
            'type': 'line',
            'line_shape': 'spline',
            'mode': 'lines+markers'
        })

    composedData = pd.DataFrame(index=all_dates)
    # TODO improve this hack to get nulls on the rows where there is no data
    for cur_station in all_stations:
        tempSeries = nn_data[cur_station]
        composedData[cur_station] = tempSeries

    for cur_station in all_stations:
        dates_station = composedData[cur_station].index
        dates_str = [x.strftime("%b %d %H hrs") for x in dates_station]
        values = composedData[cur_station].values
        output_data.append({
            'x': dates_str,
            'y': values,
            'name': F'nn_{cur_station}',
            'type': 'line',
            'line_shape': 'spline',
            'mode': 'lines+markers'
        })

    return output_data


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
    # app.run_server(debug=True, port=8051)
    app.run_server(debug=False, port=8053, host='146.201.212.214')
