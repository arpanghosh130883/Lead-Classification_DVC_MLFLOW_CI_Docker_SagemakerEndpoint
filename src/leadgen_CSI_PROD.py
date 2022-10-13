
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#Creating a function for calculating CSI

def calculate_csi(expected, actual, buckets=10, axis=0):

    expected_array = expected.to_numpy()
    actual_array = actual.to_numpy()
    

    def scale_range (input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input


    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

    breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        

    expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
    actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

    def sub_csi(e_perc, a_perc):
        if a_perc == 0:
            a_perc = 0.0001
        if e_perc == 0:
            e_perc = 0.0001

        value = (e_perc - a_perc) * np.log(e_perc / a_perc)
        return(value)

    csi_value = np.sum(sub_csi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))

    return(csi_value)

    if len(expected.shape) == 1:
        csi_values = np.empty(len(expected.shape))
    else:
        csi_values = np.empty(expected.shape[axis])

    for i in range(0, len(csi_values)):
        if len(csi_values) == 1:
            csi_values = csi(expected, actual, buckets)
        elif axis == 0:
            csi_values[i] = csi(expected[:,i], actual[:,i], buckets)
        elif axis == 1:
             csi_values[i] = csi(expected[i,:], actual[i,:], buckets)
                
    return csi_values  


def calculate_result(expected_array, actual_array, buckets=10, axis=0):
    csi_values = []
    for col in expected_array:
        train_series = expected_array[col]
        test_series = actual_array[col]
        csi = np.round(calculate_csi(train_series, test_series, buckets=10, axis=1),5)
        csi_values.append(csi)
    result = pd.DataFrame(    {'Feature': expected_array.columns.tolist(),
         'Characteristic Stability Index (CSI)': csi_values
        })
    return result
    

# Extracting Traing ad Prod data for monitoring

train_data = pd.read_csv('artifacts/train_val_test_data/x_train.csv', error_bad_lines=False, index_col=0)
# Note:  Create a table as per the attached shared x_train data csv and read data from that table as train data

prod_data = pd.read_csv('artifacts/train_val_test_data/x_test.csv', error_bad_lines=False, index_col=0)
#  Note: Read only the  independent features fead into the model in Production 

csi = calculate_result(train_data,prod_data, buckets=10, axis=1)
print(csi)
