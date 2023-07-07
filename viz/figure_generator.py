import calendar
from matplotlib import pyplot as plt
import numpy as np
from os.path import join

MONTHS = [calendar.month_name[x] for x in range(1,13)]
WEEK_DAYS = list(calendar.day_name)

def get_default_figure(data, title, range=None):
    figure = {
        'data': data,
        'layout': {
            'title': title,
            'clickmode': 'event+select',
            'legend': {
                'x': 1,
                'y': 1
            }
        }
    }
    if not(range is None):
        figure['layout']['yaxis'] = {'range': range}

    return figure



def get_error_by_date(gt_data, nn_data, type="M", metric='rmse'):
    """
    :param gt_data:
    :param nn_data:
    :param type:  'M' months, 'W' weeks, 'H' hours, 'WD' week day
    :return:
    """
    stations = nn_data.columns
    output_data = []
    if metric.lower() == 'rmse':
        square_error = (gt_data[stations] - nn_data[stations])**2
    elif metric.lower() == 'err':
        square_error = nn_data[stations] - gt_data[stations]
    elif metric.lower() == 'mae':
        square_error = np.abs(nn_data[stations] - gt_data[stations])

    # Mean along all the stations
    if metric.lower() == 'rmse':
        mean_error = square_error.mean(axis=1) ** .5
    else:
        mean_error = square_error.mean(axis=1)
    square_error.dropna(inplace=True)

    if type == "M":
        temp_group = mean_error.groupby(by=[mean_error.index.month])
    elif type == "W":
        temp_group = mean_error.groupby(by=[mean_error.index.week])
    elif type == "H":
        temp_group = mean_error.groupby(by=[mean_error.index.hour])
    elif type == "WD":
        temp_group = mean_error.groupby(by=[mean_error.index.weekday])

    error = temp_group.mean()
    if type == "M":
        x = MONTHS
    elif type == "WD":
        x = WEEK_DAYS
    else:
        x = np.arange(error.shape[0])

    output_data.append({
            'x': x,
            'y': error.values,
            'name': 'all',
            'type': 'bar',
        })

    return output_data

def get_error_by_date_by_station(gt_data, nn_data, type="M", metric='rmse'):
        """

        :param gt_data:
        :param nn_data:
        :param type:  'M' months, 'W' weeks, 'H' hours, 'WD' week day
        :return:
        """
        stations = nn_data.columns
        output_data = []
        for cur_station in stations:
            if metric.lower() == 'rmse':
                square_error = (gt_data[cur_station] - nn_data[cur_station]) ** 2
            elif metric.lower() == 'err':
                square_error = nn_data[cur_station] - gt_data[cur_station]
            elif metric.lower() == 'mae':
                square_error = np.abs(nn_data[cur_station] - gt_data[cur_station])

            # Mean along all the stations
            if metric.lower() == 'rmse':
                mean_error = square_error ** .5
            else:
                mean_error = square_error

            if type == "M":
                temp_group = mean_error.groupby(by=[mean_error.index.month])
            elif type == "W":
                temp_group = mean_error.groupby(by=[mean_error.index.week])
            elif type == "H":
                temp_group = mean_error.groupby(by=[mean_error.index.hour])
            elif type == "WD":
                temp_group = mean_error.groupby(by=[mean_error.index.weekday])

            error = temp_group.mean()
            if type == "M":
                x = MONTHS
            elif type == "WD":
                x = WEEK_DAYS
            else:
                x = np.arange(error.shape[0])

            output_data.append({
                'x': x,
                'y': error.values,
                'name': cur_station,
                'type': 'bar',
            })

        return output_data


def addColumn(col_name, start_idx, end_idx, df, ax, size=20, scatter=False, line_style='-'):
    '''
    Add a column to the plot. It can be plot or scatter
    '''
    if scatter:
        ax.scatter(df.index[start_idx:end_idx], 
               df[col_name][start_idx:end_idx], 
               label=col_name, s=size)
    else:
        ax.plot(df.index[start_idx:end_idx], 
               df[col_name][start_idx:end_idx], 
               lw = size, linestyle=line_style,
               label=col_name)


def plot_input_output_data(X_df, Y_df, cur_station, cur_pollutant, output_folder, model_name):
    fig, ax = plt.subplots(1,3, figsize=(30,10))
    station = cur_station
    times_to_plot = 48
    start_idx = 104
    end_idx = start_idx + times_to_plot

    # Plot main pollutant for single station
    addColumn(f"cont_{cur_pollutant}_{station}", start_idx, end_idx, X_df, ax[0], 
            size=4, scatter=False)

    # Add the predicted values Y (next 24 hours)
    # for c_hour in range(forecasted_hours, forecasted_hours+1):
    # for c_hour in range(forecasted_hours-5, forecasted_hours+1):
    for c_hour in range(1, 5):
        addColumn(f"plus_{c_hour:02d}_cont_{cur_pollutant}_{station}", 
                start_idx, end_idx, Y_df, ax[0], size=10)

    for c_hour in range(1, 5):
        addColumn(f"minus_{c_hour:02d}_cont_{cur_pollutant}_{station}", 
                  start_idx, end_idx, X_df, ax[0], size=10, line_style='--')

    # Plot some meteo columns
    meteo_col = "T2"
    plot_hr = 0
    cuadrants = 4
    tot_cuadrants = int(cuadrants**2 )
    cols = [f"{meteo_col}_{i}_h{plot_hr}" for i in range(tot_cuadrants)]
    meteo_data = X_df.loc[start_idx, cols]
    meteo_img = np.zeros((cuadrants, cuadrants))
    # Fill the meteo image with the data from the dataframe
    for i in range(cuadrants):
        for j in range(cuadrants):
            meteo_img[i,j] = meteo_data[f"{meteo_col}_{i*cuadrants+j}_h{plot_hr}"]

    ax[2].imshow(meteo_img, cmap='hot', interpolation='nearest')

    ax[0].legend()
    # Add some of the time columns
    addColumn(f"sin_day", start_idx, end_idx, X_df, ax[1], size=10)
    addColumn(f"cos_day", start_idx, end_idx, X_df, ax[1], size=10)
    addColumn(f"half_sin_day", start_idx, end_idx, X_df, ax[1], size=10)

    plt.show()
    plt.savefig(join(output_folder, f'{model_name}.png'))
    plt.close()
