from db.sqlCont import getPostgresConn
from db.queries import getAllStations, getCountValidData, getMaxDailySQL

from dash.dependencies import Input, Output, State
from db.names import findTable
from viz.layout import get_layout_db

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

conn = getPostgresConn()
stations_geodf = getAllStations(conn)
stations_ids = stations_geodf.index.values
pollutant = "O3"
table = findTable(pollutant)
app = get_layout_db()

@app.callback(
    [Output('nandata-plot', 'figure'),
     Output('nandata-plotall', 'figure'),
    Output('max_otres_plot', 'figure')],
[Input('drop-years', 'value')])
def display_figure(cur_year):
    print(cur_year)
    if cur_year is None:
        cur_year = 2010
    else:
        cur_year = int(cur_year)
    figure_nandata= {
        'data': getDataByYear(cur_year, sinceYear=False)
        ,
        'layout': {
            'title': F'Available data by station for the year {cur_year}',
            'clickmode': 'event+select',
            'legend': {
                'x': 1,
                'y': 1
            },
        }
    }
    figure_nandataall= {
        'data': getDataByYear(cur_year, sinceYear=True)
        ,
        'layout': {
            'title': F'Available data by station since {cur_year}',
            'clickmode': 'event+select',
            'legend': {
                'x': 1,
                'y': 1
            },
        }
    }
    figure_max_daily = {
        'data': getMaxDaily(cur_year)
        ,
        'layout': {
            'title': F'Daily max Ozone for year {cur_year}',
            'clickmode': 'event+select',
            'legend': {
                'x': 1,
                'y': 1
            },
        }
    }

    return figure_nandata, figure_nandataall, figure_max_daily


def getMaxDaily(year):
    print(F"Geting data for year {year}....")
    output_data = []
    res = getMaxDailySQL(conn, year)
    max_values = res['max'].values
    dates = res['year'].astype(str) + res['month'].astype(str) + res['day'].astype(str)
    output_data.append({
        'x': dates,
        'y': max_values,
        'type': 'line+marker',
    })
    print("Done!!!")
    return output_data


def getDataByYear(year, sinceYear=False):
    print("Geting data for year {year}....")
    output_data = []
    res = getCountValidData(conn, year, sinceYear)
    stations = res['id_est'].values
    values_per_station = []
    for cur_station in stations_ids:
        if cur_station in stations:
            values_per_station.append(res[res['id_est'] == cur_station]['value'].values[0])
        else:
            values_per_station.append(0)
    output_data.append({
        'x': stations_ids,
        'y': values_per_station,
        'type': 'bar',
    })
    print("Done!!!")
    return output_data


if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
    
