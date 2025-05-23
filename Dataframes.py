import pandas as pd
from datetime import datetime
import numpy as np

class ImgTable:
    def __init__(self, dataframe):
        self.df = dataframe
    
    def addImage(self, filename, url, source_url, date_time):
        new_image = pd.DataFrame({'Image Filename': [filename], 'URL': [url], 'Source URL': [source_url], 'Download Date And Time': [date_time]})
        self.df = pd.concat([new_image,self.df.loc[:]], ignore_index=True)
        return
    
    def storeTable(self):
        self.df.to_csv('res/ImageData.csv', index=False)

    def getValue(self, url, column):
        return self.df.loc[self.df['URL'] == url][column].values[0]

class URLTable:
    def __init__(self, dataframe):
        self.df = dataframe

    def addURL(self, url, site, inheritance_coefficient):
        if site != None:
            inherited_weight = int(self.getValue(site, 'Weight') * inheritance_coefficient)
            depth = self.getValue(site, 'Depth') + 1
        else:
            inherited_weight = 0
            depth = 0

        new_url =  pd.DataFrame({'URL': [url], 'Weight': [inherited_weight-depth], 'Total Images': [0], 'Depth': [depth], 'Estimated Refresh Rate': [600], 'Estimated Refresh Time': [datetime.now().strftime("%d/%m/%Y %H:%M:%S")], 'Last Refresh': [pd.NaT], 'Change History': ['']})
        self.df = pd.concat([new_url,self.df.loc[:]], ignore_index=True)
        return
    
    def updateURL(self, url, weight, decay_coefficient):
        history = str(self.getValue(url, 'Change History')).replace('n','').replace('N','').replace('a','')[0:9] #Retrieves the success of the 9 most recent refreshes
        if weight > 0:
            change_history = "1" + history[0:9] #Records whether new images were obtained this refresh, storing up to 10 refreshes
        else:
            change_history = "0" + history[0:9]

        current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        historical_success = str(self.getValue(url, 'Change History')).count('1') * 0.1
        upper_bound = 0.7
        lower_bound = 0.3

        estimated_refresh_rate = calcRefreshTime(self, url, historical_success, upper_bound, lower_bound)
        estimated_refresh_time = pd.to_datetime(current_time) + pd.to_timedelta(self.getValue(url, 'Estimated Refresh Rate'), unit='s') #Saves closest refresh time
        total_images = self.getValue(url, 'Total Images') + weight #Adds number of new images to total images (Done prior to new weight calculation)
        if historical_success == 0 or historical_success > lower_bound:
            calc_weight = int(self.getValue(url, 'Weight') * decay_coefficient) + weight
        else:
            calc_weight = self.getValue(url, 'Weight') + weight

        self.df.loc[self.df['URL'] == url, ['Weight', 'Total Images', 'Estimated Refresh Rate', 'Estimated Refresh Time', 'Last Refresh', 'Change History']] = [int(calc_weight), total_images, estimated_refresh_rate, estimated_refresh_time, current_time, change_history]
        return

    def storeTable(self):
        self.df.to_csv('res/URLData.csv', index=False)
    
    def getValue(self, url, column):
        return self.df.loc[self.df['URL'] == url][column].values[0]

class ResultsTable:
    def __init__(self, dataframe):
        self.df = dataframe
    
    def addResult(self, crawler_type, parallel_method, workers, decay_coefficient, inheritance_coefficient, crawl_time, total_images, total_scraped):
        new_data_point = pd.DataFrame({'Crawler Type': [crawler_type], 'Parallel Method': [parallel_method], 'Workers': [workers], 'Decay Coefficient': [decay_coefficient], 'Inheritance Coefficient': [inheritance_coefficient], 'Time': [crawl_time], 'Total Images': [total_images], 'Total Scraped': [total_scraped]})
        self.df = pd.concat([new_data_point,self.df.loc[:]], ignore_index=True)
        return
    
    def storeTable(self):
        self.df.to_csv('res/ResultsData.csv', index=False)

def createImageTable():
    image_columns = {'Image Filename': pd.Series(dtype='str'), 'URL': pd.Series(dtype='str'), 'Source URL': pd.Series(dtype='str'), 'Download Date And Time': pd.Series(dtype='datetime64[ns]')}
    image_table = pd.DataFrame(data=image_columns)
    image_table.to_csv('res/ImageData.csv', index=False)
    return image_table

def createURLTable():
    url_columns = {'URL': pd.Series(dtype='str'), 'Weight': pd.Series(dtype='int'), 'Total Images': pd.Series(dtype='int'), 'Depth': pd.Series(dtype='int'), 'Estimated Refresh Rate': pd.Series(dtype='int'), 
                   'Estimated Refresh Time': pd.Series(dtype='datetime64[ns]'), 'Last Refresh': pd.Series(dtype='datetime64[ns]'), 'Change History': pd.Series(dtype='str')}
    url_table = pd.DataFrame(data=url_columns)
    url_table.to_csv('res/URLData.csv', index=False)
    return url_table

def createResultsTable():
    results_columns = {'Crawler Type': pd.Series(dtype='str'), 'Parallel Method': pd.Series(dtype='str'), 'Workers': pd.Series(dtype='int'), 'Decay Coefficient': pd.Series(dtype='float'), 'Inheritance Coefficient': pd.Series(dtype='float'), 'Time': pd.Series(dtype='int'), 'Total Images': pd.Series(dtype='int'), 'Total Scraped': pd.Series(dtype='int')}
    results_table = pd.DataFrame(data=results_columns)
    results_table.to_csv('res/ResultsData.csv', index=False)
    return results_table

def loadImageData():
    datetimeColumns = ['Download Date And Time']
    df = pd.read_csv('res/ImageData.csv', dtype = {'Image Filename': str, 'URL': str, 'Source URL': str}, parse_dates=datetimeColumns)
    return df

def loadURLData():
    datetimeColumns = ['Estimated Refresh Time', 'Last Refresh']
    df = pd.read_csv('res/URLData.csv', dtype={'URL': str, 'Weight': int, 'Total Images': int, 'Depth': int, 'Estimated Refresh Rate': int, 'Change History': str}, parse_dates=datetimeColumns)
    return df

def loadResultsData():
    df = pd.read_csv('res/ResultsData.csv', dtype = {'Crawler Type': str, 'Parallel Method': str,'Workers': int, 'Decay Coefficient': float, 'Inheritance Coefficient': float, 'Time': int, 'Total Images': int})
    return df

def calcRefreshTime(self, url, historical_success, upper_bound, lower_bound):
    current_refresh_time = self.getValue(url, 'Estimated Refresh Rate')
    if len(str(self.getValue(url, 'Change History'))) == 10:
        historical_success = str(self.getValue(url, 'Change History')).count('1') * 0.1
        refresh_change = ((1 - (historical_success/upper_bound)) * unitStepFunction(historical_success, upper_bound) + (1 - (historical_success/lower_bound)) * unitStepFunction(lower_bound, historical_success)) * current_refresh_time
        new_refresh_time = current_refresh_time + refresh_change
    else:
        new_refresh_time = current_refresh_time
    return new_refresh_time

def unitStepFunction(x, y):
    z = x - y
    if z > 0:
        return 1
    else:
        return 0