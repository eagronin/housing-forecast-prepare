# Data Preparation and Cleaning

## Overview
This section first describes cleaning and processing of the data on single-family homes in Denver, CO.  It then turns to construction of new features that were subsequently used for predicting home values along with the existing features.  The new features include comparables-based valuation and forecaseted value using the history of prior sales.  The existing features were listed in the [previous section](https://eagronin.github.io/housing-forecast-acquire/).  Home values are subsequently predicted using a random forest model fitted to a training set and evaluated using a test set.  The model produced the test set R-squared of 0.92.  The subsequent sections discuss the steps that I took in more detail. 

Description of the data is provided in the [previous section](https://eagronin.github.io/housing-forecast-acquire/).

Estimation of the model and evaluation of its performance are discussed in the [next section](https://eagronin.github.io/housing-forecast-analyze/).

The analysis for this project was performed in Python.

## Data Cleaning and Processing 
The summary statistics of the raw dataset are reported below:  

```
                   count         mean          std        min          25%          50%          75%           max
id               15000.0  51762290.74  61908763.48  143367.00  10048022.75  25632408.50  51142218.75  3.209481e+08
zipcode          15000.0     80204.92         9.72   80022.00     80205.00     80206.00     80207.00  8.020900e+04
latitude         14985.0        39.74         0.02      39.61        39.73        39.75        39.76  3.989000e+01
longitude        14985.0      -104.96         0.04    -105.11      -104.98      -104.96      -104.94 -1.048300e+02
bedrooms         15000.0         2.71         0.90       0.00         2.00         3.00         3.00  1.500000e+01
bathrooms        15000.0         2.20         1.17       0.00         1.00         2.00         3.00  1.200000e+01
rooms            15000.0         6.16         1.96       0.00         5.00         6.00         7.00  3.900000e+01
squareFootage    15000.0      1514.50       830.64     350.00       986.00      1267.50      1766.25  1.090700e+04
lotSize          15000.0      5820.77      3013.28     278.00      4620.00      5950.00      6270.00  1.228390e+05
yearBuilt        14999.0      1929.52        29.94    1874.00      1907.00      1925.00      1949.00  2.016000e+03
lastSaleAmount   15000.0    405356.34    775699.85     259.00    194000.00    320000.00    463200.00  4.560000e+07
priorSaleAmount  11287.0    259435.01    337938.70       0.00    110000.00    210000.00    330240.00  1.600000e+07
estimated_value  15000.0    637162.55    504418.49  147767.00    398434.75    518357.50    687969.25  1.014531e+07
```

In order to show summary statistics for the dates of last and prior sales, I convert these features into date format:

```python
data.lastSaleDate = pd.to_datetime(data.lastSaleDate)
data.priorSaleDate = pd.to_datetime(data.priorSaleDate)
print(np.around(data[['lastSaleDate', 'priorSaleDate']].describe().transpose(), decimals=2).to_string())
```

Which results in the following output: 

```
               count unique                  top freq                first                 last
lastSaleDate   15000   4347  2014-03-03 00:00:00   19  1997-08-01 00:00:00  2017-07-21 00:00:00
priorSaleDate  11173   4475  2008-01-03 00:00:00   15  1967-05-10 00:00:00  2017-07-11 00:00:00
```

In order to prepare the data for the analysis, I evaluated accuracy of the data, imputed missing values and constructed new features. 

### 1.	Evaluation of data accuracy

This step fixes inaccuracies in the data.  It removes one duplicate record, an outlier in terms of lotSize (lotSize of 278 sq. feet) and 3 outliers in terms of lastSaleAmount ($45.6 million for each - a price not consistent with the other characteristics of these homes, such as square footage). Removal of these recors is performed using the code below:

```python
import pandas as pd

def check_data_accuracy(data):
    
    # check if there are multiple samples with the same address
    
    data.address.unique().size
    data[data.duplicated(subset = 'address', keep = False) == True]
    if 4762 in data.index:
        data.drop(data.index[4762], axis = 0, inplace = True)   # drop one duplicate
        
    # remove 3 records that appear to have incorrect lastSaleAmount
    data = data[data.lastSaleAmount != 45600000]
    
    # convert zipcode to string format
    data.zipcode = data.zipcode.astype(str)
    
    # remove one observation with lotSize < 500 sq. feet
    data = data[data.lotSize > 500]
        
    return data
```

### 2.	Imputing missing values
A look at the summary statistics above reveals that the most significant issue in the data is a large number of missing values for priorSaleDate and priorSaleAmount.   Further, there are also 1,296 zero values for priorSaleAmount.  

With respect to these zero values, it appears that the time period between the last and prior sale dates is substantially shorter when prior sale amount is zero (611 days on average) compared to when priorSaleAmount is not zero (2080 days on average).  It is possible that such prior sales never occurred and rather represent a stage in the process of the last sale.  If the same sale sometimes appears in the data both as the last sale and as the prior sale, we would expect sales dates and amounts to match between lastSaleDate and priorSaleDate for at least a small portion of the dataset.  There are indeed 155 samples in which priorSaleDate and priorSaleAmount are identical to lastSaleDate and lastSaleAmount, respectively.

Based on the above observations I assume that when priorSaleAmount is zero, there was only one sale.  I set priorSaleAmount and priorSaleDate to lastSaleAmount and lastSaleDate when priorSaleAmount is either zero or missing, or when priorSaleDate is missing.  I further assume that observations with identical lastSaleDate and priorSaleDate but different lastSaleAmount and priorSaleAmount also have only one sale, and change priorSaleAmount to lastSaleAmount.  In the records with priorSaleDate coming after lastSaleDate, I set priorSaleDate to lastSaleDate. Finally, I create a dummy variable taking value of 1 when prior sale exists and 0 otherwise.

The other features with missing values include latitude, longitude and yearBuilt.  I impute average latitude and longitude for the home’s zipcode when the coordinates are missing, and I set one missing value for yearBuilt to the average yearBuilt in the data.

The code implementing the above steps is shown below:

```python
import pandas as pd
import numpy as np

def impute_missing_values(data):
    
    # convert zipcode to string format
    data.zipcode = data.zipcode.astype(str)
    
    # change priorSaleAmount and priorSaleDate to lastSaleAmount and lastSaleDate, respectively, when priorSaleAmount is either zero or NAN
    data.priorSaleDate[(data.priorSaleAmount == 0) | (data.priorSaleAmount.isnull())] = data.lastSaleDate
    data.priorSaleAmount[(data.priorSaleAmount == 0) | (data.priorSaleAmount.isnull())] = data.lastSaleAmount
    # change priorSaleAmount and priorSaleDate to lastSaleAmount and lastSaleDate, respectively, when priorSaleDate is NAN
    data.priorSaleDate[data.priorSaleDate.isnull()] = data.lastSaleDate
    data.priorSaleAmount[data.priorSaleDate.isnull()] = data.lastSaleAmount
    
    # assume that an observation with identical lastSaleDate and priorSaleDate but different lastSaleAmount and priorSaleAmount did not have a prior sale (i.e., change priorSaleAmount to lastSaleAmount)
    data.priorSaleAmount[(data.priorSaleAmount != data.lastSaleAmount) & (data.priorSaleDate == data.lastSaleDate)] = data.lastSaleAmount
    
    # fix observations with inconsistent last sale date and prior sale date    
    data.priorSaleDate[data.lastSaleDate - data.priorSaleDate < pd.Timedelta('0D')] = data.lastSaleDate
    
    # create a dummy variable that equals 1 when prior sale occurred and zero otherwise
    data['priorSaleDummy'] = 1
    data.priorSaleDummy[data.priorSaleDate == data.lastSaleDate] = 0

    # assign average longitude and average latitude for the respective zipcode 
    # to homes for which these attributes are missing
    # calculate average for each zipcode:
    temp = data[['zipcode', 'latitude', 'longitude']]
    temp = temp.groupby('zipcode').mean()
    temp = temp.rename(columns = {'latitude': 'av_lat', 'longitude': 'av_lon'})
    temp = temp.reset_index(drop = False)
    data = data.merge(temp, on = 'zipcode', how = 'left')
    data.latitude[data.latitude.isnull()] = data.av_lat
    data.longitude[data.longitude.isnull()] = data.av_lon
    data.drop(['av_lat', 'av_lon'], axis = 1, inplace = True)
    # in case we want to delete data with missing latitude or longitude: 
    # data = data[(data.latitude.isnull() == False) & (data.longitude.isnull() == False)]
    
    # impute missing values in yearBuilt
    data.yearBuilt[data.yearBuilt.isnull()] = data.yearBuilt.mean()
    
    return data
```

The refined summary statistics after cleaning the data and imputing missing values are showned below:

```
                               count         mean          std        min          25%          50%          75%           max
id                           14995.0  51747046.96  61886948.72  143367.00  10048014.50  25632409.00  51142171.50  3.209481e+08
latitude                     14995.0        39.74         0.02      39.61        39.73        39.75        39.76  3.989000e+01
longitude                    14995.0      -104.96         0.04    -105.11      -104.98      -104.96      -104.94 -1.048300e+02
bedrooms                     14995.0         2.71         0.90       0.00         2.00         3.00         3.00  1.500000e+01
bathrooms                    14995.0         2.19         1.17       0.00         1.00         2.00         3.00  1.200000e+01
rooms                        14995.0         6.16         1.96       0.00         5.00         6.00         7.00  3.900000e+01
squareFootage                14995.0      1514.48       830.63     350.00       986.00      1268.00      1766.00  1.090700e+04
lotSize                      14995.0      5821.33      3013.40     832.00      4620.00      5950.00      6270.00  1.228390e+05
yearBuilt                    14995.0      1929.51        29.93    1874.00      1907.00      1925.00      1949.00  2.016000e+03
lastSaleAmount               14995.0    565579.91    602265.46     435.61    309350.55    449413.01    643237.79  2.923536e+07
priorSaleAmount              14995.0    522941.82    706421.18     183.33    260258.14    398366.98    609081.04  3.573380e+07
estimated_value              14995.0    637238.82    504478.69  147767.00    398465.50    518476.00    688162.00  1.014531e+07
priorSaleDummy               14995.0         0.64         0.48       0.00         0.00         1.00         1.00  1.000000e+00
rebuiltDummy                 14995.0         0.01         0.08       0.00         0.00         0.00         0.00  1.000000e+00
yearsBetweenSales            14995.0         3.68         5.86       0.00         0.00         0.99         5.26  4.988000e+01
annAppreciation              14995.0         0.10         0.18      -0.50         0.00         0.03         0.11  5.000000e-01
Dummy2012ForLastSaleAmount   14995.0         0.41         0.49       0.00         0.00         0.00         1.00  1.000000e+00
lastSaleAmountAfter2012      14995.0    242000.54    455360.62       0.00         0.00         0.00    427672.49  1.773222e+07
Dummy2012ForPriorSaleAmount  14995.0         0.16         0.36       0.00         0.00         0.00         0.00  1.000000e+00
priorSaleAmountAfter2012     14995.0     75165.38    312688.04       0.00         0.00         0.00         0.00  1.773222e+07
80203                        14995.0         0.03         0.16       0.00         0.00         0.00         0.00  1.000000e+00
80204                        14995.0         0.17         0.38       0.00         0.00         0.00         0.00  1.000000e+00
80205                        14995.0         0.24         0.43       0.00         0.00         0.00         0.00  1.000000e+00
80206                        14995.0         0.16         0.37       0.00         0.00         0.00         0.00  1.000000e+00
80207                        14995.0         0.27         0.44       0.00         0.00         0.00         1.00  1.000000e+00
80209                        14995.0         0.12         0.33       0.00         0.00         0.00         0.00  1.000000e+00
80123                        14995.0         0.01         0.11       0.00         0.00         0.00         0.00  1.000000e+00
```

### 3.	Constructing new features
I construct several new features and make adjustments to the existing features in order to enhance the model’s performance.  I discuss the steps that I took below.

First, while the history of sales prices is important for valuing a home, many sales in the data took place a long time ago.  Sale amounts are not adjusted for the appreciation in home value between lastSaleDate and present time.  The following example demonstrates how this issue can affect the model’s performance.  Suppose, for example, that home A is currently valued at $2 million and home B is currently valued at $1 million.  The data also shows that home A was last sold in 1990 for $600,000, while home B was last sold recently in 2018.  In this example, a model that uses lastSaleAmount as a feature may predict a lower value for home A even though this home is as twice as much more expensive than home B. 

To fix this timing issue, I adjust lastSaleAmount for the housing price appreciation between lastSaleDate and the present using S&P/Case-Shiller CO-Denver Home Price Index.  This index increased in value from 49.856 in December 1987 to 206.014 in December 2017.   This implies the annual appreciation rate of 4.84 percent (calculated as (206.014/49.856)^(1/30)-1).  Then, for example, if a home was last sold in 2010 (eight years ago), I adjust lastSaleAmount using the following formula: lastSaleAmount * 1.0484^8-1.  I perform the same adjustment for priorSaleAmount.

```python
data['priceAppreciation'] = 1.0484
data['lastSaleAmount'] = data.lastSaleAmount * data.priceAppreciation.pow(2018 - data.lastSaleDate.dt.year)
data['priorSaleAmount'] = data.priorSaleAmount * data.priceAppreciation.pow(2018 - data.priorSaleDate.dt.year)
data.drop(['priceAppreciation'], axis = 1, inplace = True)
```

Second, because homes are typically valued using information from sales of comparable homes, I created a valuation measure using estimated_value (or Zestimate) of comparable homes.  When the model was trained, each home in the training data has been valued using the three most comparable homes in the training data (i.e., the average of estimated_values for the three homes).  When the model was tested, each home in the test data has been compared to the three most comparable homes in the training set (not in the test set, because estimated_value of comparable homes cannot be used as a predictor of value at the test stage).  I used geographical proximity, the number of bedrooms and the number of bathrooms to assess comparability.  Figuratively speaking, this valuation approach is identical in spirit to an implementation of the K Nearest Neighbors algorithm.

The implementation of the valuation approach using comparable homes is shown below:

```python
import pandas as pd
import numpy as np
from math import sin, cos, sqrt, atan2, radians
    
train['lat'] = train.latitude.apply(lambda i: radians(i))
train['lon'] = train.longitude.apply(lambda i: radians(i))
train['estimated_value_sqft'] = train.estimated_value / train.squareFootage
train['lastSaleAmount_sqft'] = train.lastSaleAmount / train.squareFootage

def zestimate_comps(idTargetHome, sqftTargetHome, lonTargetHome, latTargetHome, bedTargetHome, bathTargetHome):

    # select comparables and compute an estimate of home value based on zestimate (estimated_value) of comparables

    #print('before: ', train.shape)
    train_cp = train   
    #print('after: ', train_cp.shape)

    R = 6373.0        # approximate radius of earth in km

    latTargetHome = radians(latTargetHome)
    lonTargetHome = radians(lonTargetHome)

    dlon = train.lon - lonTargetHome
    dlat = train.lat - latTargetHome

    a = dlat.apply(lambda i: sin(i / 2)**2) + cos(latTargetHome) * train.lat.apply(lambda i: cos(i)) * dlon.apply(lambda i: sin(i / 2)**2)  
    c = a.apply(lambda i: 2 * atan2(sqrt(i), sqrt(1 - i)))

    distance = R * c

    train_cp['distance'] = distance

    # calculate estimate of home value based on comps' zestimate
    # create an index by adding points for shorter distance from the home being valued, identical number of 
    # bedrooms and bathrooms 
    train_cp['ind'] = 0
    train_cp.ind[train_cp.distance <= 1] = train_cp.ind + 1
    train_cp.ind[train_cp.distance <= 2] = train_cp.ind + 1
    train_cp.ind[train_cp.bedrooms == bedTargetHome] = train_cp.ind + 1
    train_cp.ind[train_cp.bathrooms == bathTargetHome] = train_cp.ind + 1
    train_cp.ind = train_cp.ind.max() - train_cp.ind    
    train_cp = train_cp.sort_values(by = ['ind', 'distance'], ascending = True)
    #print(train_cp[['ind', 'distance']].iloc[:3,:])
    train_cp = train_cp.iloc[:4,:]
    train_cp = train_cp[train_cp.id != idTargetHome]       # exclude the home being value from the set of potential comparables
    zestimateCompsValue = train_cp.estimated_value_sqft.mean() * sqftTargetHome
    #print(train_cp[['id', 'ind', 'distance']])

    return zestimateCompsValue
    
X['zestCompVal'] = np.nan
X.zestCompVal = X.apply(lambda x: zestimate_comps(x.id, x.squareFootage, x.longitude, x.latitude, x.bedrooms, x.bathrooms), axis = 1)

```

I have also created a similar valuation metric which uses lastSaleAmount (adjusted to the present value of the home using the S&P Case Shiller index as shown above) instead of estimated_value:

```python
def sold_comps(idTargetHome, sqftTargetHome, lonTargetHome, latTargetHome, bedTargetHome, bathTargetHome):

    # select comparables and compute an estimate of home value based on lastSaleAmount of comparables

    #print('before: ', train.shape)
    train_cp = train[train.id != idTargetHome]   # exclude the home being valued from the set of potential comparables
    #print('after: ', train_cp.shape)

    R = 6373.0        # approximate radius of earth in km

    latTargetHome = radians(latTargetHome)
    lonTargetHome = radians(lonTargetHome)

    dlon = train.lon - lonTargetHome
    dlat = train.lat - latTargetHome

    a = dlat.apply(lambda i: sin(i / 2)**2) + cos(latTargetHome) * train.lat.apply(lambda i: cos(i)) * dlon.apply(lambda i: sin(i / 2)**2)  
    c = a.apply(lambda i: 2 * atan2(sqrt(i), sqrt(1 - i)))

    distance = R * c

    train_cp['distance'] = distance

    # calculate value estimate based on comps' lastSaleAmount if sold within last 3.5 years
    # create an index by adding points for shorter distance from the home being valued, identical number of 
    # bedrooms and bathrooms and shorter time elapsed since lastSaleDate
    train_cp['ind'] = 0
    train_cp.ind[(train_cp.distance <= 1) & (train_cp.lastSaleDate.dt.year > 2014)] = train_cp.ind + 1
    train_cp.ind[(train_cp.distance <= 2) & (train_cp.lastSaleDate.dt.year > 2014)] = train_cp.ind + 1
    train_cp.ind[(train_cp.bedrooms == bedTargetHome) & (train_cp.lastSaleDate.dt.year > 2014)] = train_cp.ind + 1
    train_cp.ind[(train_cp.bathrooms == bathTargetHome) & (train_cp.lastSaleDate.dt.year > 2014)] = train_cp.ind + 1
    train_cp.ind[train_cp.lastSaleDate.dt.year > 2015] = train_cp.ind + 1
    train_cp.ind[train_cp.lastSaleDate.dt.year > 2016] = train_cp.ind + 1
    train_cp.ind = train_cp.ind.max() - train_cp.ind    
    train_cp = train_cp.sort_values(by = ['ind', 'distance'], ascending = True)
    #print(train_cp[['ind', 'distance']].iloc[:3,:])
    train_cp = train_cp.iloc[:4,:]
    train_cp = train_cp[train_cp.id != idTargetHome]       # exclude the home being value from the set of potential comparables
    soldCompsValue = train.lastSaleAmount_sqft.mean() * sqftTargetHome

    return soldCompsValue

X['soldCompVal'] = np.nan
X.soldCompVal = X.apply(lambda x: sold_comps(x.id, x.squareFootage, x.longitude, x.latitude, x.bedrooms, x.bathrooms), axis = 1)
```

Third, I construct the annualized home price appreciation between priorSaleDate and lastSaleDate.  This feature has a potential to capture the variation in home price appreciation across homes.  

However, this measure is noisy because, for example, purchasing a fixer upper, refurbishing it and selling one year later at a substantially higher price would result in a home price appreciation that is not representative of home price appreciation over the lifetime of that home.

The code that implements the construction of home price appreciation is shown below: 

```python
# calculate values per sq. foot
data['estimated_value_sqft'] = data.estimated_value / data.squareFootage
data['lastSaleAmount_sqft'] = data.lastSaleAmount / data.squareFootage
data['priorSaleAmount_sqft'] = data.priorSaleAmount / data.squareFootage

data['yearsBetweenSales'] = (data.lastSaleDate - data.priorSaleDate).dt.days/365
data['annAppreciation'] = (data.lastSaleAmount_sqft / data.priorSaleAmount_sqft).pow(1/data.yearsBetweenSales) - 1

# smooth the annualized appreciation rate as sales that occur within a few months of each other result in large variation of annualized appreciation rate; it is unlikely that home value increases or declines at an annualized rate of more than 50 percent
data.annAppreciation[data.annAppreciation > 0.5] = 0.5
data.annAppreciation[data.annAppreciation < -0.5] = -0.5
```

Fourth, in order to account for the explosive growth in housing prices in Denver after 2012 (a pattern observed in other cities as well), I added a dummy variable that takes the value of one if lastSaleDate is after 2012, and zero otherwise.   I also added such a dummy for priorSaleDate and the interactions of these dummies with lastSaleAmount and priorSaleAmount, respectively:

```python
data['Dummy2012ForLastSaleAmount'] = 0
data.Dummy2012ForLastSaleAmount[data.lastSaleDate.dt.year > 2012] = 1
data['lastSaleAmountAfter2012'] = data.lastSaleAmount * data.Dummy2012ForLastSaleAmount

data['Dummy2012ForPriorSaleAmount'] = 0
data.Dummy2012ForPriorSaleAmount[data.priorSaleDate.dt.year > 2012] = 1
data['priorSaleAmountAfter2012'] = data.priorSaleAmount * data.Dummy2012ForPriorSaleAmount
```

Fifth, I create a dummy for homes that were rebuilt after the last sale.  These are homes with yearBuilt greater than the year of lastSaleDate.  Identifying such homes is important because their estimated_value could be considerably higher than the last sale price:

```python
data[data.yearBuilt > data.lastSaleDate.dt.year].shape
data['rebuiltDummy'] = 0
data.rebuiltDummy[data.yearBuilt > data.lastSaleDate.dt.year] = 1
```

Sixth, there is a considerable variation in home prices across zipcodes.  Therefore, I added a dummy variable for each zipcode:

```python
zipcode = pd.get_dummies(data.zipcode)

# drop zipcodes with a very small number of home sales
z1 = ['80203', '80204', '80205', '80206', '80207', '80209', '80123']
z2 = zipcode.columns
z_intersection = [value for value in z1 if value in z2]
zipcode = zipcode[z_intersection]
data = pd.concat([data, zipcode], axis = 1)
```

Finally, I remove the features that are not going to be used as explanatory variables in fitting the model:

```python
data.drop(['address', 'city', 'state', 'lastSaleAmount_sqft', 'priorSaleAmount_sqft', 'estimated_value_sqft'], axis = 1, inplace = True)
```

Next step:  [Analysis](https://eagronin.github.io/housing-forecast-analyze/).
