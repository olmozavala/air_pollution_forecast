import numpy as np
from datetime import datetime, timedelta


def getEvenIndexForSplit(tot_num_pts, num_splits):
    """This util function generates the start indexes for each partition
    that will contain an almost even number of points"""

    split_size = int(np.floor(tot_num_pts/num_splits))
    extra_points = tot_num_pts % num_splits

    prev_indx = 0
    next_indx = 0
    output_indexes = np.zeros((num_splits,2), dtype=int)

    curr_split = 0
    while next_indx < tot_num_pts:
        prev_indx = next_indx
        if extra_points > 0:
            next_indx = prev_indx + split_size + 1
            extra_points -= 1
        else:
            next_indx = prev_indx + split_size
        output_indexes[curr_split] = [prev_indx, next_indx]
        curr_split+=1

    return output_indexes


def getStringDates(start_date, num_hours, date_format="%Y_%m_%d %H:%M:%S"):
    str_dates = [(start_date + timedelta(hours=i)).strftime(date_format) for i in num_hours]
    return str_dates


# Test
if __name__== '__main__':
    # print(getEvenIndexForSplit(114,33))
    print(getStringDates(datetime.now(), range(24)))

