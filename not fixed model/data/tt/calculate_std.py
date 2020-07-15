#!/usr/bin/env python3
## import libraries
from scipy.stats import norm
import numpy as np
import pandas as pd
import sys

# specify
name = sys.argv[1]

#csv_path = './independent/' + name + '_model_error_distribution.csv'
csv_path = './all/' + name + '_all_data_model_error_distribution.csv'
#csv_path = './bidir all/' + name + '_bidir_all_data_model_error_distribution.csv'

tmp = np.array(pd.read_csv(csv_path, header=None))
(mu, sigma) = norm.fit(tmp)

df = pd.DataFrame(data=np.array([mu, sigma]))
#data_name = './independent/std/' + name + '_std.csv'
data_name = './all/std/' + name + '_std.csv'
#data_name = './bidir all/std/' + name + '_std.csv'

df.to_csv(data_name, header=0, index=0)
