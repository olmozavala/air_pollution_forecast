import calendar
import numpy as np

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

