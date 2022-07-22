import numpy as np

def getQuadrantsAsString(tot_num_quadrants):
    each_dim_size = (int(np.sqrt(tot_num_quadrants)))
    return F"{each_dim_size}_{each_dim_size}"