import numpy as np
import pandas as pd

#defining a custome fucntion for producing the range
def scale_range (input, min, max):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input

#y_test_preds = np.load('artifacts/predicted_values/predicted_test.npy')
# y_train = np.array(pd.read_csv('artifacts/train_val_test_data/y_train.csv'))
y_test = np.array(pd.read_csv('artifacts/train_val_test_data/predicted_test.csv'))

buckets = 10
raw_breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
breakpoints = scale_range(raw_breakpoints, np.min(y_test), np.max(y_test))

# y_test is a numpy array of the prediction output of test data set, to be load and stored as static numpy array using loadtxt form numpy . Predicted_test.csv Data shared in csv file and one time load


#Building the Train and Prod data basis the scoring parameter into buckets
train_count = np.histogram(y_test, breakpoints)[0]
prod_count = np.histogram(XXXX, breakpoints)[0]

#Converting into Pandas Dataframe
df = pd.DataFrame({'Bucket': np.arange(1, 11), 'Breakpoint Value':breakpoints[1:], 'Train Count':train_count, 'Prod Count':prod_count})
df['Train Percent'] = df['Train Count'] / len(y_test)
df['Prod Percent'] = df['Prod Count'] / len(XXXX)

# Where XXXX is the numpy array format of classification output of New Leads

#Exception handling for division by zero error

# df['Prod Percent'].loc[df['Prod Percent'] == 0] = 0.001
# df[df['Prod Percent'] == 0]['Prod Percent'] = 0.001
[0.001 if x==0 else x for x in df['Prod Percent']]


#PSI Calculation
df['PSI'] = (df['Prod Percent'] - df['Train Percent']) * np.log(df['Prod Percent'] / df['Train Percent'])

print(np.sum(df['PSI']))

#Interpretation
#PSI < 0.1: no significant population change

#PSI < 0.2: moderate population change

#PSI >= 0.2: significant population change