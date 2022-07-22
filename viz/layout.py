import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from viz.figure_generator import get_default_figure, get_error_by_date, get_error_by_date_by_station

def get_html_cols_for_error_metric(gt_data, nn_data, plot_type, metric, by_station=True):
    cols = []
    for id, name in plot_type.items():
        if by_station:
            cols.append(
                dbc.Col([
                    dcc.Graph(
                        id=F'plot_error_by_{by_station}_{name}_{metric}',
                        figure=get_default_figure(get_error_by_date_by_station(gt_data, nn_data, type=id),
                                                  title=F'{metric} by {name}')),
                ], width=3)
            )
        else:
            cols.append(
                dbc.Col([
                    dcc.Graph(
                        id=F'plot_error_by_{by_station}_{name}_{metric}',
                        figure=get_default_figure(get_error_by_date(gt_data, nn_data, type=id, metric=metric),
                                                  title=F'{metric} by {name}')),
                ], width=3)
            )
    return cols

def get_layout(default_date, stations_geodf, gt_data, nn_data, title):
    navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Link", href="#")),
            dbc.DropdownMenu(
                nav=True,
                in_navbar=True,
                label="Menu",
                children=[
                    dbc.DropdownMenuItem("Temperature"),
                    dbc.DropdownMenuItem("Salinity"),
                    dbc.DropdownMenuItem(divider=True),
                    dbc.DropdownMenuItem("Velocity"),
                ],
            ),
        ],
        brand=title,
        brand_href="#",
        sticky="top",
    )


    plot_type = {'M':'month', 'W':'week', 'WD':'week day', 'H':'hour'}
    mse_plots_by_station = get_html_cols_for_error_metric(gt_data, nn_data, plot_type, metric='RMSE', by_station=True)
    mse_plots = get_html_cols_for_error_metric(gt_data, nn_data, plot_type, metric='RMSE', by_station=False)
    err_plots = get_html_cols_for_error_metric(gt_data, nn_data, plot_type, metric='ERR', by_station=False)
    mae_plots = get_html_cols_for_error_metric(gt_data, nn_data, plot_type, metric='MAE', by_station=False)

    body = dbc.Container([
        # Global information

        dbc.Row(mse_plots_by_station),
        dbc.Row(mse_plots),
        dbc.Row(err_plots),
        dbc.Row(mae_plots),
        dbc.Row([
            dbc.Col([
                dcc.DatePickerSingle(
                    id='calendar',
                    date=default_date),
                html.Button(
                    '-',
                    id='prevday',
                ),
                html.Button(
                    '+',
                    id='nextday',
                ),
            ]),
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id='metrics-plot',
                    figure={}),
            ], width=6),
            dbc.Col([
                dcc.Graph(
                    id='max-daily-plot',
                    figure={}),
            ], width=6),
        ]),
        dbc.Row([
            dbc.Col([
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
            ],
                width=6),
            dbc.Col([
                dcc.Graph(
                    id='merged-plot',
                    figure={}),
            ], width=6),
        ]),

        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id='nn-plot',
                    figure={}),
            ], width=6),
            dbc.Col([
                dcc.Graph(
                    id='gt-plot',
                    figure={}),
            ], width=6),
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id='U-plot',
                    figure={}),
            ], width=3),
            dbc.Col([
                dcc.Graph(
                    id='V-plot',
                    figure={}),
            ], width=3),
            dbc.Col([
                dcc.Graph(
                    id='RAINC-plot',
                    figure={}),
            ], width=3),
            dbc.Col([
                dcc.Graph(
                    id='T2-plot',
                    figure={}),
            ], width=3),
        ]),
    ], fluid=True)

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = html.Div([navbar, body])

    return app


def get_layout_input_NN(default_date, stations_geodf, meteo_variables):
    navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Link", href="#")),
            dbc.DropdownMenu(
                nav=True,
                in_navbar=True,
                label="Menu",
                children=[
                    dbc.DropdownMenuItem("Temperature"),
                    dbc.DropdownMenuItem("Salinity"),
                    dbc.DropdownMenuItem(divider=True),
                    dbc.DropdownMenuItem("Velocity"),
                ],
            ),
        ],
        brand="Demo",
        brand_href="#",
        sticky="top",
    )

    body = dbc.Container([
        dbc.Row([
            dbc.Col([
                dcc.DatePickerSingle(
                    id='calendar',
                    date=default_date)
            ]),
            dbc.Col([
                html.Button(
                    '-',
                    id='prevday',
                ),
            ]),
            dbc.Col([
                html.Button(
                    '+',
                    id='nextday',
                ),
            ]),
        ]),
        dbc.Row([
            dbc.Col([
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
            ],
                width=6),
        ]),
        dbc.Row([get_all_meteo_plots(meteo_var) for meteo_var in meteo_variables]),
    ], fluid=True)

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                    suppress_callback_exceptions=True)
    app.layout = html.Div([navbar, body])

    return app


def get_all_meteo_plots(meteo_var):
    cur_col = dbc.Col([
        dcc.Graph(
            id=F'{meteo_var}-plot',
            figure={}),
    ], width=4)
    return cur_col

def get_layout_db():
    navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Link", href="#")),
            dbc.DropdownMenu(
                nav=True,
                in_navbar=True,
                label="Menu",
                children=[
                    dbc.DropdownMenuItem("Temperature"),
                ],
            ),
        ],
        brand="Demo",
        brand_href="#",
        sticky="top",
    )

    years = [{'label': F'{year}', 'value': F'{year}'} for year in range(2010,2021)]
    body = dbc.Container([
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id='drop-years',
                    options=years,
                    value=F'2015'
                )
            ], width=2),
            dbc.Col([
                dcc.Graph(
                    id='nandata-plot',
                    figure={}),
            ], width=6),
            dbc.Col([
                dcc.Graph(
                    id='nandata-plotall',
                    figure={}),
            ], width=6),
            dbc.Col([
                dcc.Graph(
                    id='max_otres_plot',
                    figure={}),
            ], width=6),
        ]),
    ], fluid=True)

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = html.Div([navbar, body])

    return app


