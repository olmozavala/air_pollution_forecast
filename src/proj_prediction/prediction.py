import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from img_viz.common import create_folder
import os

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

    train_ids = split_info.iloc[:,0]
    val_ids = split_info.iloc[:,1]
    test_ids = split_info.iloc[:,2]

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
                error = metric_f(GT, NN)
                metrics_result[cur_col][metric_name] = error
                # Training errors
                GT = gt[cur_col][train_ids].values
                NN = nn[cur_col][train_ids].values
                error = metric_f(GT, NN)
                metrics_result[cur_col][F"{metric_name}_training"] = error
                # Validation errors
                GT = gt[cur_col][val_ids].values
                NN = nn[cur_col][val_ids].values
                error = metric_f(GT, NN)
                metrics_result[cur_col][F"{metric_name}_validation"] = error
                # Test errors
                GT = gt[cur_col][test_ids].values
                NN = nn[cur_col][test_ids].values
                error = metric_f(GT, NN)
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
