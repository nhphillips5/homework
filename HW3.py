# 1. Transform the "mexico_weather.csv" data into the "clean_mexico_weather.csv"
# data. Use your cleaned dataset to make a plot of the daily average minimum and
# maximum temperatures (tmin and tmax). Put both tmin and tmax on the same axis.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

weather = pd.read_csv('mexico_weather.csv')

#print(weather.head(10))
#print(weather.info())
#print(weather.loc[0].head(10))
weather.replace([999,9999], np.nan, inplace = True)
#print(weather.head(10))

# Parse the info form the first column
col1 = weather.iloc[:,0]

station = [str(x)[0:11] for x in col1]
year = [int(str(x)[11:15]) for x in col1]
month = [int(str(x)[15:17]) for x in col1]
variable = [str(x)[17:] for x in col1]

new_weather = pd.DataFrame()
new_weather['id'] = station
new_weather['year'] = year
new_weather['month'] = month
new_weather['element'] = variable

#print(new_weather.head())
#print(new_weather.info())
#print(weather.info())

weather = pd.concat([new_weather, weather], axis=1, join='outer')
#weather.info()
#weather.head(30)

melted_weather = pd.melt(weather, id_vars = ['id','year','month','element','0'],var_name = 'day', value_name = 'variable')
#print(melted_weather.head(10))

# get rid of the zero column
melted_weather.drop(columns = ['0'], inplace = True)
#print(melted_weather.head(10))

melted_weather['day'] = melted_weather['day'].apply(lambda x: int(re.findall(r'\d+',str(x))[0]))

weather = melted_weather.pivot_table(index = ['year','month','day'], columns = 'element', values = 'variable')
weather.reset_index(drop = False, inplace = True)

print(weather.head(10))

#plot the daily average tmin and tmax
avg_tmax = weather.groupby(['month', 'day'])['TMAX'].mean().plot()
avg_tmin = weather.groupby(['month', 'day'])['TMIN'].mean().plot()

# 2. Use the pandas function read_html to get the tables on the following website:
url = 'https://en.wikipedia.org/wiki/List_of_United_States_tornadoes_from_January_to_March_2020#January'

tornadoes = pd.read_html(url)


# a. How many tables are on this page?

print(len(tornadoes[:]))

# There are 32 tables at that URL

# b. Make a dataset of all the tables that have 10 columns (these are the tables
#     with tornado information)


#print(tornadoes[:])

wind_list = list()

for i in range(0,32):
    if len(tornadoes[i].columns) == 10:
        wind_list.append(tornadoes[i])

torn_df = pd.concat(wind_list)


# c. Delete the rows in the dataset where the "Start Coordinate" is missing

torn_df['Start Coord.'].head(10)

torn_df.dropna(axis = 0, subset = ['Start Coord.'], inplace = True)

# d. Make a plot of the Start Coordinate with longitude on the x-axis and
#     latitude on the y-axis

latitude = torn_df['Start Coord.'].apply(lambda x: float(re.findall('\d+\.\d+', x)[0]))
longitude = torn_df['Start Coord.'].apply(lambda x: float(re.findall('\d+\.\d+', x)[1]))
longitude = -longitude

plt.scatter(longitude,latitude)

# e. Get summary statistics (min, Q1, median, Q3, maximum, mean, standard
#     deviation) for the time that a tornado is active (using the "Time" column),
#     the path length, and the maximum width.

# Time
pattern1 = '[^a-zA-z\d:]'
torn_df['start'] = torn_df['Time (UTC)'].apply(lambda x: re.split(pattern1, x)[0])
torn_df['end'] = torn_df['Time (UTC)'].apply(lambda x: re.split(pattern1, x)[-1])

start = pd.to_datetime(torn_df['start'],format = '%H:%M')
end = pd.to_datetime(torn_df['end'],format = '%H:%M')

torn_df['passed_time'] = ((end - start).apply(lambda x: x.seconds))

torn_df['passed_time'].describe()

# Path Length

torn_df['Path length'] = pd.to_numeric(torn_df['Path length'].apply(lambda x: x.split('mi')[0].strip()), errors = 'coerce')

torn_df['Path length'].describe()

# Max Width 
def split_width(x):
    if str(x).lower != 'unknown':
        return str(x).split()[0]
    else:
        return(x)

torn_df['Max width'] = pd.to_numeric(torn_df['Max width'].apply(split_width), errors = 'coerce')

torn_df['Max width'].describe()

