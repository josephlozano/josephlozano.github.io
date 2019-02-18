---
title: "Bay Area Rapid Transit - Data Analysis"
layout: post
date: 2019-01-06
tag: posting
projects: true
hidden: true # don't count this post in blog pagination
description: "Binary Classification Project"
category: blog
author: josephlozano
externalLink: false
---

# Business Objective: 
Analyze data and answer questions from San Francisco Bay Area Rapid Trasit (BART).

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

## Data import


```python
dtypes = {'Origin':'object','Destination':'object','Throughput':'int64'}
```


```python
rider2016 = pd.read_csv('date-hour-soo-dest-2016.csv', dtype=dtypes, parse_dates=['DateTime'])
```


```python
rider2016.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9971582 entries, 0 to 9971581
    Data columns (total 4 columns):
    Origin         object
    Destination    object
    Throughput     int64
    DateTime       datetime64[ns]
    dtypes: datetime64[ns](1), int64(1), object(2)
    memory usage: 304.3+ MB



```python
rider2016.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Origin</th>
      <th>Destination</th>
      <th>Throughput</th>
      <th>DateTime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12TH</td>
      <td>12TH</td>
      <td>1</td>
      <td>2016-01-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12TH</td>
      <td>16TH</td>
      <td>1</td>
      <td>2016-01-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12TH</td>
      <td>24TH</td>
      <td>4</td>
      <td>2016-01-01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12TH</td>
      <td>ASHB</td>
      <td>4</td>
      <td>2016-01-01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12TH</td>
      <td>BALB</td>
      <td>2</td>
      <td>2016-01-01</td>
    </tr>
  </tbody>
</table>
</div>




```python
rider2017 = pd.read_csv('date-hour-soo-dest-2017.csv', dtype=dtypes, parse_dates=['DateTime'])
```


```python
rider2017.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3313625 entries, 0 to 3313624
    Data columns (total 4 columns):
    Origin         object
    Destination    object
    Throughput     int64
    DateTime       datetime64[ns]
    dtypes: datetime64[ns](1), int64(1), object(2)
    memory usage: 101.1+ MB



```python
rider2017.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Origin</th>
      <th>Destination</th>
      <th>Throughput</th>
      <th>DateTime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12TH</td>
      <td>19TH</td>
      <td>1</td>
      <td>2017-01-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12TH</td>
      <td>24TH</td>
      <td>2</td>
      <td>2017-01-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12TH</td>
      <td>BAYF</td>
      <td>1</td>
      <td>2017-01-01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12TH</td>
      <td>CIVC</td>
      <td>5</td>
      <td>2017-01-01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12TH</td>
      <td>COLS</td>
      <td>2</td>
      <td>2017-01-01</td>
    </tr>
  </tbody>
</table>
</div>




```python
stationInfo = pd.read_csv('station_info.csv')
```


```python
stationInfo.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 46 entries, 0 to 45
    Data columns (total 4 columns):
    Abbreviation    46 non-null object
    Description     46 non-null object
    Location        46 non-null object
    Name            46 non-null object
    dtypes: object(4)
    memory usage: 1.5+ KB



```python
stationInfo.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Abbreviation</th>
      <th>Description</th>
      <th>Location</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12TH</td>
      <td>1245 Broadway, Oakland CA 94612&lt;br /&gt;12th St. ...</td>
      <td>-122.271450,37.803768,0</td>
      <td>12th St. Oakland City Center (12TH)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16TH</td>
      <td>2000 Mission Street, San Francisco CA 94110&lt;br...</td>
      <td>-122.419694,37.765062,0</td>
      <td>16th St. Mission (16TH)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19TH</td>
      <td>1900 Broadway, Oakland CA 94612&lt;br /&gt;19th Stre...</td>
      <td>-122.268602,37.808350,0</td>
      <td>19th St. Oakland (19TH)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>24TH</td>
      <td>2800 Mission Street, San Francisco CA 94110&lt;br...</td>
      <td>-122.418143,37.752470,0</td>
      <td>24th St. Mission (24TH)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ASHB</td>
      <td>3100 Adeline Street, Berkeley CA 94703&lt;br /&gt;As...</td>
      <td>-122.270062,37.852803,0</td>
      <td>Ashby (ASHB)</td>
    </tr>
  </tbody>
</table>
</div>




```python
combinedRider = pd.concat([rider2016,rider2017],ignore_index=True)
```


```python
combinedRider.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 13285207 entries, 0 to 13285206
    Data columns (total 4 columns):
    Origin         object
    Destination    object
    Throughput     int64
    DateTime       datetime64[ns]
    dtypes: datetime64[ns](1), int64(1), object(2)
    memory usage: 405.4+ MB



```python
pd.set_option('display.float_format', '{:.2f}'.format)
```

**Describe rider data to get overview of data.**


```python
combinedRider.describe(include='all')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Origin</th>
      <th>Destination</th>
      <th>Throughput</th>
      <th>DateTime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>13285207</td>
      <td>13285207</td>
      <td>13285207.00</td>
      <td>13285207</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>46</td>
      <td>46</td>
      <td>nan</td>
      <td>10962</td>
    </tr>
    <tr>
      <th>top</th>
      <td>POWL</td>
      <td>POWL</td>
      <td>nan</td>
      <td>2017-04-21 17:00:00</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>424161</td>
      <td>415710</td>
      <td>nan</td>
      <td>1847</td>
    </tr>
    <tr>
      <th>first</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>nan</td>
      <td>2016-01-01 00:00:00</td>
    </tr>
    <tr>
      <th>last</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>nan</td>
      <td>2017-05-03 23:00:00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>12.77</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>33.51</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>10.00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1826.00</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



**Create additional date features to answer questions.**


```python
combinedRider['Month'] = combinedRider.DateTime.apply(lambda x: x.month)
combinedRider['Day'] = combinedRider.DateTime.apply(lambda x: x.day)
combinedRider['Year'] = combinedRider.DateTime.apply(lambda x: x.year)
```


```python
combinedRider['DayOfWeek'] = combinedRider.DateTime.apply(lambda x: x.strftime('%A'))
```


```python
combinedRider['TimeOfDay'] = combinedRider.DateTime.apply(lambda x: x.hour)
```


```python
combinedRider.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Origin</th>
      <th>Destination</th>
      <th>Throughput</th>
      <th>DateTime</th>
      <th>Month</th>
      <th>Day</th>
      <th>Year</th>
      <th>DayOfWeek</th>
      <th>TimeOfDay</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12TH</td>
      <td>12TH</td>
      <td>1</td>
      <td>2016-01-01</td>
      <td>1</td>
      <td>1</td>
      <td>2016</td>
      <td>Friday</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12TH</td>
      <td>16TH</td>
      <td>1</td>
      <td>2016-01-01</td>
      <td>1</td>
      <td>1</td>
      <td>2016</td>
      <td>Friday</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12TH</td>
      <td>24TH</td>
      <td>4</td>
      <td>2016-01-01</td>
      <td>1</td>
      <td>1</td>
      <td>2016</td>
      <td>Friday</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12TH</td>
      <td>ASHB</td>
      <td>4</td>
      <td>2016-01-01</td>
      <td>1</td>
      <td>1</td>
      <td>2016</td>
      <td>Friday</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12TH</td>
      <td>BALB</td>
      <td>2</td>
      <td>2016-01-01</td>
      <td>1</td>
      <td>1</td>
      <td>2016</td>
      <td>Friday</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Which BART station is the busiest?

Defining busyness as the origin's throughput, the MONT station is the busiest with 16,107,314 throughput.


```python
originThroughput = combinedRider.groupby('Origin')['Throughput'].sum().reset_index()
```


```python
originThroughput.sort_values('Throughput',ascending=False).iloc[0]
```




    Origin            MONT
    Throughput    16107314
    Name: 25, dtype: object




```python
originThroughput.sort_values('Throughput',ascending=True).iloc[0]
```




    Origin         WSPR
    Throughput    81397
    Name: 45, dtype: object




```python
originThroughput.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Throughput</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>46.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3689505.70</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3547610.15</td>
    </tr>
    <tr>
      <th>min</th>
      <td>81397.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1811522.75</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2761226.00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3656062.50</td>
    </tr>
    <tr>
      <th>max</th>
      <td>16107314.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(15,8))
sns.barplot('Throughput', 'Origin', data=originThroughput)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1ed61c14438>




![png](/assets/images/BART_data_analysis/output_29_1.png)


## What is the least popular BART route?

Defining popularity as the routes's average throughput, the **ORIN-WDUB**, **WDUB-NCON**, and **LAFY-CAST** are tied for the least popular with with an average throughput of 1.08.


```python
routeThroughput = combinedRider.groupby(['Origin','Destination'])['Throughput'].mean().reset_index()
```


```python
routeThroughput.sort_values('Throughput',ascending=True)[:5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Origin</th>
      <th>Destination</th>
      <th>Throughput</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1377</th>
      <td>ORIN</td>
      <td>WDUB</td>
      <td>1.08</td>
    </tr>
    <tr>
      <th>2005</th>
      <td>WDUB</td>
      <td>NCON</td>
      <td>1.08</td>
    </tr>
    <tr>
      <th>2107</th>
      <td>WSPR</td>
      <td>SBRN</td>
      <td>1.08</td>
    </tr>
    <tr>
      <th>973</th>
      <td>LAFY</td>
      <td>CAST</td>
      <td>1.08</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>WDUB</td>
      <td>LAFY</td>
      <td>1.09</td>
    </tr>
  </tbody>
</table>
</div>



## When is the best time to go to SF from Berkeley if you want to find a seat?

Defining the best time to find a seat as the minimum average throughput, 4 am is the best time to go to SF from Berkeley.


```python
combinedData = combinedRider.merge(stationInfo,
                                   left_on='Origin',right_on='Abbreviation')
```


```python
combinedData = combinedData.merge(stationInfo,
                                  left_on='Destination', right_on='Abbreviation',
                                  suffixes=('_Orig','_Dest'))
```


```python
stationInfo.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Abbreviation</th>
      <th>Description</th>
      <th>Location</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12TH</td>
      <td>1245 Broadway, Oakland CA 94612&lt;br /&gt;12th St. ...</td>
      <td>-122.271450,37.803768,0</td>
      <td>12th St. Oakland City Center (12TH)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16TH</td>
      <td>2000 Mission Street, San Francisco CA 94110&lt;br...</td>
      <td>-122.419694,37.765062,0</td>
      <td>16th St. Mission (16TH)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19TH</td>
      <td>1900 Broadway, Oakland CA 94612&lt;br /&gt;19th Stre...</td>
      <td>-122.268602,37.808350,0</td>
      <td>19th St. Oakland (19TH)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>24TH</td>
      <td>2800 Mission Street, San Francisco CA 94110&lt;br...</td>
      <td>-122.418143,37.752470,0</td>
      <td>24th St. Mission (24TH)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ASHB</td>
      <td>3100 Adeline Street, Berkeley CA 94703&lt;br /&gt;As...</td>
      <td>-122.270062,37.852803,0</td>
      <td>Ashby (ASHB)</td>
    </tr>
  </tbody>
</table>
</div>




```python
import re
```

**Use pattern to find city name between comma and 'CA'.**


```python
pattern = ',\s(.*)\sCA' 
```


```python
string = '2000 Mission Street, San Francisco CA 94110<br/>'
```


```python
re.search(pattern, string).group(1)
```




    'San Francisco'



**Create origin and destination features using regex pattern.**


```python
combinedData['City_Orig'] = combinedData['Description_Orig'].apply(lambda x: re.search(pattern, x).group(1))
```


```python
combinedData['City_Dest'] = combinedData['Description_Dest'].apply(lambda x: re.search(pattern, x).group(1))
```


```python
criteria0 = combinedData['City_Orig'] == 'Berkeley'
```


```python
criteria1 = combinedData['City_Dest'] == 'San Francisco'
```


```python
TimeDayThroughput = combinedData[(criteria0) & (criteria1)].groupby('TimeOfDay')['Throughput'].mean().reset_index()
```


```python
TimeDayThroughput.sort_values('Throughput').iloc[0]
```




    TimeOfDay    4.00
    Throughput   1.61
    Name: 4, dtype: float64



## Which day of the week is the busiest?

Defining busyness as the total throughput per day, Wednesday is the busiest with 30,645,415 total throughput.


```python
dayThroughput = combinedData.groupby(['DayOfWeek'])['Throughput'].sum().reset_index()
```


```python
dayThroughput.sort_values('Throughput',ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DayOfWeek</th>
      <th>Throughput</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>Wednesday</td>
      <td>30645415</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Tuesday</td>
      <td>30318375</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Thursday</td>
      <td>30025051</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Friday</td>
      <td>28350830</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Monday</td>
      <td>26938975</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Saturday</td>
      <td>13683238</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sunday</td>
      <td>9593228</td>
    </tr>
  </tbody>
</table>
</div>



## How many people take the BART late at night? 

Defining 'late at night' as midnight to 3am, the total throughput is 1,845,915 which is 1 percent of the total throughput of 169,555,112.


```python
def hour_descr(hour):
    if 0 <= hour <= 3:
        return 'late night'
    elif 4 <= hour <= 6:
        return 'early morning'
    elif 7 <= hour <= 9:
        return 'morning rush hour'
    elif 10 <= hour <= 15:
        return 'midday'
    elif 16 <= hour <= 18:
        return 'evening rush hour'
    elif 19 <= hour <= 23:
        return 'evening'
    else:
        return 'error'
```


```python
combinedData['Hour_Descr'] = combinedData['TimeOfDay'].apply(hour_descr)
```


```python
lateNightThroughput = combinedData.groupby([combinedData['Hour_Descr'] == 'late night'])['Throughput'].sum().reset_index()
```


```python
lateNightThroughput.columns = ['Late_At_Night','Total_Throughput']
```


```python
lateNightThroughput['Perc_Total'] = lateNightThroughput['Total_Throughput'] / lateNightThroughput['Total_Throughput'].sum()
```


```python
lateNightThroughput
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Late_At_Night</th>
      <th>Total_Throughput</th>
      <th>Perc_Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>167709197</td>
      <td>0.99</td>
    </tr>
    <tr>
      <th>1</th>
      <td>True</td>
      <td>1845915</td>
      <td>0.01</td>
    </tr>
  </tbody>
</table>
</div>


