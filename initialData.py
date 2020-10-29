import matplotlib.dates
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import re
import statistics as st

dataset = pd.read_csv(r'C:\Users\Peter\Downloads\EdgeProductivityAdvisorDemo.csv')
#dataset.TimeStamp = dataset.TimeStamp.apply(lambda x: datetime.strptime(x,'%m/%d/%Y %H:%M'))
dataset = dataset.iloc[::-1]
dataset = dataset.reset_index()
dataset['DatesSplit'] = dataset['TimeStamp'].apply(lambda x: re.split('\s', x))
dataset['Date'] = dataset.DatesSplit.apply(lambda x: x[0])
dataset['Time'] = dataset.DatesSplit.apply(lambda x: x[1])
dataset.TimeStamp = dataset.TimeStamp.apply(lambda x: datetime.strptime(x,'%m/%d/%Y %H:%M'))


dates_list = list(set(dataset['Date']))
drilling_date = '10/21/2020'
dataset_subset = dataset[dataset['Date'] == drilling_date]

dataset_subset['TimeDiff'] = dataset_subset.TimeStamp.diff()
dataset_subset['TimeDiff'] = dataset_subset['TimeDiff']-pd.to_timedelta(dataset_subset['TimeDiff'].dt.days, unit = 'd')
dataset_subset['deltaTime'] = dataset_subset['TimeDiff'].apply(lambda x: x.seconds/60)
dataset_subset = dataset_subset.drop(columns = ['DatesSplit', 'TimeDiff'])
# dataset.Time = dataset.Time.apply(lambda x: datetime.strptime(x,'%H:%M'))

# plt.plot(dataset['TimeStamp'],dataset['Drill String Length (ft)'], 'o')
# plt.title('All Drilling Data')
# plt.gcf().autofmt_xdate()
# plt.show()



mode = st.mode(dataset_subset['deltaTime'])

def replaceBreaks(deltaTime):
    if deltaTime > 10:
        deltaTime = mode
    return deltaTime

dataset_subset['deltaTime'] = dataset_subset['deltaTime'].apply(replaceBreaks)




# Subset

plt.plot(dataset_subset['TimeStamp'],dataset_subset['Drill String Length (ft)'], 'o')
plt.title(str(drilling_date) + ' Drilling Data')
plt.gcf().autofmt_xdate()
# plt.show()

dataset_subset['deltaLength'] = dataset_subset['Drill String Length (ft)'].diff()
dataset_subset['ROP'] = (dataset_subset['deltaLength']/dataset_subset['deltaTime'])*60







print('Done')