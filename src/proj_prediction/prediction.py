import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from img_viz.common import create_folder
import os

def executeMetric(GT, NN, metric):
    not_nans = np.logical_not(np.isnan(GT))
    a = GT[not_nans].astype(np.int32)
    b = NN[not_nans].astype(np.int32)
    error = metric(a, b)
    # c = a-b
    # erroroz = np.mean((a - b)**2)
    # from scipy.stats import linregress
    # slope, intercept, r_value, p_value, std_err = linregress(a, b)
    # sop = r_value**2
    # from sklearn.metrics import r2_score
    # per = r2_score(a,b)

    return error


def compute_metrics(gt, nn, metrics, split_info, output_file, column_names=[], by_column=True):
    """
    Compute the received metrics and save the results in a csv file
    :param gt: Dataframe with the values stored by station by column
    :param nn: Result of the NN
    :param metrics:
    :param split_info:
    :param output_file:
    :param column_names:
    :param by_column:
    :return:
    """

    # Eliminate those cases where the original output is unknown

    train_ids = split_info.iloc[:,0]
    val_ids = split_info.iloc[:,1]
    test_ids = split_info.iloc[:,2]
    val_ids = val_ids.drop(pd.isna(val_ids).index.values)
    train_ids = train_ids.drop(pd.isna(train_ids).index.values)
    test_ids = test_ids.drop(pd.isna(test_ids).index.values)

    output_file = output_file.replace('.csv','')
    create_folder(os.path.dirname(output_file))

    if by_column:
        if len(column_names) == 0:
            column_names = [str(i) for i in range(len(gt[0]))]

        all_metrics = list(metrics.keys())
        all_metrics += [F"{x}_training" for x in metrics.keys()]
        all_metrics += [F"{x}_validation" for x in metrics.keys()]
        all_metrics += [F"{x}_test" for x in metrics.keys()]
        metrics_result = pd.DataFrame({col: np.zeros(len(metrics)*4) for col in column_names}, index=all_metrics)

        for metric_name, metric_f in metrics.items():
            for cur_col in column_names:
                # All errors
                GT = gt[cur_col].values
                NN = nn[cur_col].values
                error = executeMetric(GT, NN, metric_f)
                metrics_result[cur_col][metric_name] = error
                # Training errors
                if len(train_ids) > 0:
                    GT = gt[cur_col][train_ids].values
                    NN = nn[cur_col][train_ids].values
                    error = executeMetric(GT, NN, metric_f)
                else:
                    error = 0
                metrics_result[cur_col][F"{metric_name}_training"] = error
                # Validation errors
                if len(val_ids) > 0:
                    GT = gt[cur_col][val_ids].values
                    NN = nn[cur_col][val_ids].values
                    error = executeMetric(GT, NN, metric_f)
                else:
                    error = 0
                metrics_result[cur_col][F"{metric_name}_validation"] = error
                # Test errors
                if len(test_ids) > 0:
                    GT = gt[cur_col][test_ids].values
                    NN = nn[cur_col][test_ids].values
                    error = executeMetric(GT, NN, metric_f)
                else:
                    error = 0
                metrics_result[cur_col][F"{metric_name}_test"] = error
                # import matplotlib.pyplot as plt
                # print(metric_f(GT[0:100], NN[0:100]))
                # plt.plot(GT[0:100])
                # plt.plot(NN[0:100])
                # plt.show()

        metrics_result.to_csv(F"{output_file}.csv")
        nn_df = pd.DataFrame(nn, columns=column_names, index=gt.index)
        nn_df.to_csv(F"{output_file}_nnprediction.csv")

    return metrics_result
