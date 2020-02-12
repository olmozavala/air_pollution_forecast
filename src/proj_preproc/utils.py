import numpy as np
from datetime import datetime, timedelta


def getEvenIndexForSplit(tot_num_pts, num_splits):
    """This util function generates the start indexes for each partition
    that will contain an almost even number of points"""

    next_indx = 0
    output_indexes = np.zeros((num_splits, 2), dtype=int)
    curr_split = 0

    if tot_num_pts > num_splits: # This is the normal behaviour
        split_size = int(np.floor(tot_num_pts/num_splits))
        extra_points = tot_num_pts % num_splits

        while next_indx < tot_num_pts:
            prev_indx = next_indx
            if extra_points > 0:
                next_indx = prev_indx + split_size + 1
                extra_points -= 1
            else:
                next_indx = prev_indx + split_size
            output_indexes[curr_split] = [min(prev_indx, tot_num_pts-2), min(next_indx, tot_num_pts-1)]
            curr_split+=1

        return output_indexes
    else:  # In this case we apply nearest neighbor basically
        split_size = (tot_num_pts-1) / num_splits
        for i in range(1, num_splits+1):
            prev_indx = int(np.floor(i * split_size))
            next_indx = prev_indx + 1
            output_indexes[curr_split] = [prev_indx, next_indx]
            curr_split += 1
        return output_indexes


def getStringDates(start_date, num_hours, date_format="%Y_%m_%d %H:%M:%S"):
    str_dates = [(start_date + timedelta(hours=i)).strftime(date_format) for i in num_hours]
    return str_dates


# Test
if __name__== '__main__':
    # print(getEvenIndexForSplit(114,33))
    print(getStringDates(datetime.now(), range(24)))

