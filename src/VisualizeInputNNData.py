import json
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
import datetime
from viz.layout import get_layout_input_NN

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

meteo_variables = ['U10', 'V10', 'RAINC', 'T2', 'RAINNC', 'SWDOWN', 'GLW', 'UV10MAG']
conn = getPostgresConn()
stations_geodf = getAllStations(conn)
pollutant = "O3"
table = findTable(pollutant)
# default_date = date.today()
year = 2012
default_date = dt.strptime(F'{year}-01-01', date_format)
time_range = 1

print("Reading data....")
gt_data = pd.read_csv(F'/data/UNAM/Air_Pollution_Forecast/Data/MergedDataCSV/Current/{year}_cont_otres_AllStations.csv',
                      index_col = 0, parse_dates = True)
print("Done!")
app = get_layout_input_NN(default_date, stations_geodf, meteo_variables)

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
    [Output(F'{x}-plot', 'figure') for x in meteo_variables],
    [Input('id-map', 'clickData'),
     Input('calendar', 'date'),
     Input('id-map', 'selectedData')])
def display_figure(hoverData, cur_date, selectedData):
    if cur_date is None:
        cur_date = default_date.strftime(date_format)
    try:
        cur_date = dt.strptime(cur_date.split(' ')[0], date_format_ext)
    except:
        cur_date = dt.strptime(cur_date.split(' ')[0], date_format)

    start_date = cur_date - timedelta(days=time_range)
    end_date = cur_date + timedelta(days=time_range)

    start_date = start_date - timedelta(hours=24)
    end_date = end_date - timedelta(hours=24)

    all_figures = []
    for meteo_var in meteo_variables:
        all_figures.append(test(meteo_var, gt_data, start_date, end_date))

    return all_figures

def test(meteo_var, gt_data, start_date, end_date):
    figure_Temp = {
        'data': getMeteoData(gt_data[start_date:end_date], meteo_var),
        'layout': {
            'title': F'{meteo_var}',
            'clickmode': 'event+select',
            'legend': {
                'x': 1,
                'y': 1
            },
        }
    }
    return figure_Temp

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
    app.run_server(debug=True)
