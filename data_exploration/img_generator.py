# ** file img_generator.py
# file img_generator.py (cells para poder hacer widgets)

# Widget Graficar por estacion, fecha y contaminates [checkbox]. Si es
# mas de un contaminante habria que normalizar los datos para mostrarlos
# en la misma grafica.

# Widget Graficar contaminante vs variable meteorologica [fecha,
# contaminante, campo meteorologico, estacion, horas]. La idea aqui
# seria ver la relacion entre el contaminante y el campo meteorologico.

# Widget Graficar promedio por estacion y varianza. Aqui la idea seria
# ver cuanto varian los valores por estacoin. [start_date, end_date,
# stations=all []]

#%%
import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import interact, VBox, DatePicker, SelectMultiple, Button, Output
from data_generator import pollutant_by_stations, average_meteo

# Function to plot meteorological and pollutant data without normalization
def plot_meteo_data_without_normalization(start_date, end_date, stations, pollutants, meteorological_fields, y_min=None, y_max=None, out=None):
    combined_df = pd.DataFrame()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    symbols = ['-', '--', '-.', ':', 'o', 'v', '^']

    # Create a color map for stations
    station_colors = {station: colors[i % len(colors)] for i, station in enumerate(stations)}

    # Process pollutant data
    for i, station in enumerate(stations):
        for j, pollutant in enumerate(pollutants):
            try:
                pollutant_data = pollutant_by_stations(start_date.strftime('%Y-%m-%d %H:%M'), end_date.strftime('%Y-%m-%d %H:%M'), [station], pollutant)
                pollutant_df = pd.DataFrame(pollutant_data)
                combined_df = pd.concat([combined_df, pollutant_df.add_prefix(f'{pollutant}_{station}_')], axis=1)
            except KeyError as e:
                print(f"Error: Pollutant {pollutant} not found. Exception: {e}")
                continue

    # Process meteorological data
    for i, meteorological_field in enumerate(meteorological_fields):
        try:
            meteo_df = average_meteo(start_date.strftime('%Y-%m-%d %H:%M'), (end_date - start_date).days * 24, meteorological_field)
            combined_df = pd.concat([combined_df, meteo_df.rename(f'{meteorological_field}')], axis=1)
        except KeyError as e:
            print(f"Error: Meteorological field {meteorological_field} not found. Exception: {e}")
            continue

    # Plot the data
    with out:
        out.clear_output()
        plt.figure(figsize=(12, 6))
        
        # Plot pollutant data
        for col in combined_df.columns:
            for k, pollutant in enumerate(pollutants):
                if pollutant in col:
                    station = col.split('_')[-1]
                    try:
                        plt.plot(combined_df.index, combined_df[col], symbols[k % len(symbols)], color=station_colors[station], label=f'{pollutant} ({station})')
                    except KeyError as e:
                        print(f"Error: Station {station} not found in station colors. Exception: {e}")
                        continue
        
        # Plot meteorological data
        for col in combined_df.columns:
            if any(meteorological_field in col for meteorological_field in meteorological_fields):
                plt.plot(combined_df.index, combined_df[col], symbols[i % len(symbols)], color='k', label=f'{col}')

        plt.title('Relationship between Pollutants and Meteorological Fields')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.show()

# Function to create and display the widget for unnormalized data
def plot_by_station_date_and_meteo_pollutant_unnormalized(start_date="2022-05-03 00:00", end_date="2022-05-06 00:00", ok_stations=['UAX', 'MER', 'XAL', 'PED'], y_min=None, y_max=None):
    # Widget configuration
    start_date_picker = DatePicker(description='Start Date', value=pd.to_datetime(start_date))
    end_date_picker = DatePicker(description='End Date', value=pd.to_datetime(end_date))
    stations_select = SelectMultiple(description='Stations', options=ok_stations, value=ok_stations)
    pollutant_select = SelectMultiple(description='Pollutants', options=['cont_otres', 'cont_pmdoscinco', 'cont_pmdiez', 'cont_nox'], value=['cont_otres', 'cont_nox'])
    meteorological_field_select = SelectMultiple(description='Meteo Fields', options=['T2', 'U10', 'V10', 'RAINC'], value=['T2', 'RAINC'])
    button = Button(description="Recalculate")
    out = Output()

    def update_plot(button):
        plot_meteo_data_without_normalization(start_date_picker.value, end_date_picker.value, stations_select.value, pollutant_select.value, meteorological_field_select.value, y_min, y_max, out)

    button.on_click(update_plot)

    # User interface
    ui = VBox([start_date_picker, end_date_picker, stations_select, pollutant_select, meteorological_field_select, button, out])

    display(ui)



#%%

import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import interact, VBox, DatePicker, SelectMultiple, Button, Output
from data_generator import pollutant_by_stations, average_meteo

# Function to normalize data
def normalize_data(df, mean=None, std=None):
    if mean is None or std is None:
        return (df - df.mean()) / df.std()
    else:
        return (df - mean) / std

# Function to plot meteorological and pollutant data
def plot_meteo_data(start_date, end_date, stations, pollutants, meteorological_fields, out):
    combined_df = pd.DataFrame()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    symbols = ['-', '--', '-.', ':', 'o', 'v', '^']

    # Create a color map for stations
    station_colors = {station: colors[i % len(colors)] for i, station in enumerate(stations)}

    # Process pollutant data
    for i, station in enumerate(stations):
        for j, pollutant in enumerate(pollutants):
            try:
                pollutant_data = pollutant_by_stations(start_date.strftime('%Y-%m-%d %H:%M'), end_date.strftime('%Y-%m-%d %H:%M'), [station], pollutant)
                pollutant_df = pd.DataFrame(pollutant_data)
                normalized_pollutant_df = normalize_data(pollutant_df)
                combined_df = pd.concat([combined_df, normalized_pollutant_df.add_prefix(f'{pollutant}_{station}_')], axis=1)
            except KeyError as e:
                print(f"Error: Pollutant {pollutant} not found. Exception: {e}")
                continue

    # Process meteorological data
    for i, meteorological_field in enumerate(meteorological_fields):
        try:
            meteo_df = average_meteo(start_date.strftime('%Y-%m-%d %H:%M'), (end_date - start_date).days * 24, meteorological_field)
            normalized_meteo_df = normalize_data(meteo_df)
            combined_df = pd.concat([combined_df, normalized_meteo_df.rename(f'{meteorological_field}')], axis=1)
        except KeyError as e:
            print(f"Error: Meteorological field {meteorological_field} not found. Exception: {e}")
            continue

    # Plot the data
    with out:
        out.clear_output()
        plt.figure(figsize=(12, 6))
        
        # Plot pollutant data
        for col in combined_df.columns:
            for k, pollutant in enumerate(pollutants):
                if pollutant in col:
                    station = col.split('_')[-1]
                    try:
                        plt.plot(combined_df.index, combined_df[col], symbols[k % len(symbols)], color=station_colors[station], label=f'{pollutant} ({station})')
                    except KeyError as e:
                        print(f"Error: Station {station} not found in station colors. Exception: {e}")
                        continue
        
        # Plot meteorological data
        for col in combined_df.columns:
            if any(meteorological_field in col for meteorological_field in meteorological_fields):
                plt.plot(combined_df.index, combined_df[col], symbols[i % len(symbols)], color='k', label=f'{col}')

        plt.title('Relationship between Pollutants and Meteorological Fields')
        plt.xlabel('Date')
        plt.ylabel('Normalized Value')
        plt.legend()
        plt.show()

# Function to create and display the widget
def plot_by_station_date_and_meteo_pollutant(start_date="2022-05-03 00:00", end_date="2022-05-06 00:00", ok_stations=['UAX', 'MER', 'XAL', 'PED']):
    # Widget configuration
    start_date_picker = DatePicker(description='Start Date', value=pd.to_datetime(start_date))
    end_date_picker = DatePicker(description='End Date', value=pd.to_datetime(end_date))
    stations_select = SelectMultiple(description='Stations', options=ok_stations, value=ok_stations)
    pollutant_select = SelectMultiple(description='Pollutants', options=['cont_otres', 'cont_pmdoscinco', 'cont_pmdiez', 'cont_nox'], value=['cont_otres', 'cont_nox'])
    meteorological_field_select = SelectMultiple(description='Meteo Fields', options=['T2', 'U10', 'V10', 'RAINC'], value=['T2', 'RAINC'])
    button = Button(description="Recalculate")
    out = Output()

    def update_plot(button):
        plot_meteo_data(start_date_picker.value, end_date_picker.value, stations_select.value, pollutant_select.value, meteorological_field_select.value, out)

    button.on_click(update_plot)

    # User interface
    ui = VBox([start_date_picker, end_date_picker, stations_select, pollutant_select, meteorological_field_select, button, out])

    display(ui)


# %%
import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import interact, Checkbox, VBox, DatePicker, SelectMultiple, Button, Output
from data_generator import pollutant_by_stations

# Function to normalize data
def normalize_data(df):
    return (df - df.min()) / (df.max() - df.min())

# Function to plot data
def plot_data(start_date, end_date, stations, pollutants, out):
    symbols = ['-', '--', '--*', '-.', ':', 'o']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    data_frames = []
    for i, pollutant in enumerate(pollutants):
        try:
            data = pollutant_by_stations(start_date.strftime('%Y-%m-%d %H:%M'), end_date.strftime('%Y-%m-%d %H:%M'), stations, pollutant)
            data = pd.DataFrame(data)
            if len(pollutants) > 1:
                data = normalize_data(data)
            data_frames.append((data, symbols[i % len(symbols)]))
        except KeyError as e:
            print(f"Error: Pollutant {pollutant} not found. Exception: {e}")
            continue

    with out:
        out.clear_output()
        plt.figure(figsize=(12, 6))
        for j, station in enumerate(stations):
            try:
                for i, (data, symbol) in enumerate(data_frames):
                    plt.plot(data.index, data[station], symbol, label=f'{station} - {pollutants[i]}', color=colors[j % len(colors)])
            except KeyError as e:
                print(f"Error: Station {station} not found. Exception: {e}")
                continue
        plt.title('Pollutants by Station')
        plt.xlabel('Date')
        plt.ylabel('Normalized Value' if len(pollutants) > 1 else 'Value')
        plt.legend()
        plt.show()

# Function to create and display the widget
def plot_by_station_date_and_pollutant(start_date="2022-05-14 00:00", end_date="2022-05-16 00:00", ok_stations=['UAX', 'MER', 'XAL', 'PED']):
    # Widget configuration
    start_date_picker = DatePicker(description='Start Date', value=pd.to_datetime(start_date))
    end_date_picker = DatePicker(description='End Date', value=pd.to_datetime(end_date))
    stations_select = SelectMultiple(description='Stations', options=ok_stations, value=ok_stations)
    pollutants_checkboxes = [
        Checkbox(description='cont_otres', value=True),
        Checkbox(description='cont_pmdoscinco', value=True),
        Checkbox(description='cont_pmdiez', value=True),
        Checkbox(description='cont_nox', value=True),
        Checkbox(description='cont_nodos', value=False)
    ]

    pollutants_dict = {
        'cont_otres': 'cont_otres',
        'cont_pmdoscinco': 'cont_pmdoscinco',
        'cont_pmdiez': 'cont_pmdiez',
        'cont_nox': 'cont_nox',
        'cont_nodos': 'cont_nodos'
    }

    out = Output()
    button = Button(description="Recalculate")

    def update_plot(button):
        selected_pollutants = [pollutants_dict[p.description] for p in pollutants_checkboxes if p.value]
        plot_data(start_date_picker.value, end_date_picker.value, stations_select.value, selected_pollutants, out)

    button.on_click(update_plot)

    # User interface
    ui = VBox([start_date_picker, end_date_picker, stations_select] + pollutants_checkboxes + [button, out])

    display(ui)
