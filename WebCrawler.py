import re
import urllib.request
import os
import sys
import time
import heapq
import cv2
import logging as lg
import tensorflow as tf
import pandas as pd
import threading
import multiprocessing
#import requests

from Dataframes import *
from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from urllib.parse import urlparse, urljoin
from keras.models import load_model
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from fake_useragent import UserAgent

class RecordScheduler(threading.Timer): #Class for calling the record results function at the specified intervals
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)

# def processArguments():
#     timeframes = ["d", "day", "w", "week", "m", "month", "y", "year"]
#     valid_settings = ["noseeds", "images", "queries", "focused", "multiprocessing", "scraped"]
#     print(isinstance(int(command_args[0]), int))
#     try:
#         if isinstance(int(command_args[0]), int) and int(command_args[0]) > 0:
#             end_time = command_args[0]
#         if len(command_args) >= 0:
#             if isinstance(int(command_args[1]), int) and int(command_args[1]) > 0:
#                 max_workers = command_args[1]
#             for setting in command_args:
#                 if isinstance(setting, str):
#                     setting = setting.lower()
#                     if setting in timeframes:
#                         timeframe = setting
#                         continue
#                     if setting in valid_settings:
#                         if setting == "noseeds":
#                             use_seeds = False
#                         if setting == "queries":
#                             use_queries = True
#                         if setting == "images":
#                             use_images = True
#                         if setting == "focused":
#                             use_incremental = False
#                         if setting == "multiprocessing":
#                             use_threading = False
#                         if setting == "scraped":
#                             show_scraped = True
#     except:
#         print("Invalid runtime or workers. Please runtime in seconds (int) as the first argument and pass the number of workers (int) the second argument.")
#         exit()

def initialiseFiles():
    if use_seeds == True:
        f = open("res/SeedURLs.txt", "r")
        for seed in f:
            seed_urls.append(seed.replace('\n', ''))
        f.close()
    f = open("res/Blacklist.txt", "r")
    for url in f:
        blacklist.append(url.replace('\n', ''))
    f.close()
    return seed_urls, blacklist

def initialiseWebDriver(i):
    options = Options()
    ua = UserAgent() 
    userAgent = ua.random
    options.add_argument(f'user-agent={userAgent}') 
    options.add_argument('--headless') #Run selenium minimised
    options.add_argument('--incognito') #Run in incognito to ensure no privileges exist for any domains
    options.add_experimental_option('excludeSwitches', ['enable-logging']) #Remove DevTools listening messages
    lg.getLogger('WDM').setLevel(lg.NOTSET)
    driver = webdriver.Chrome(options=options)
    if use_queries == True and i == 0: #Initialise first web driver with google consent enabled
        driver.get("https://www.google.com") 
        time.sleep(10)
        driver.find_element(by=By.ID, value='L2AGLb').click()
    return driver

def initialiseSeedUrls():
    disasters = ['Earthquake', 'Landslide', 'Volcanic Eruption', 'Flood', 'Hurricane', 'Tornado', 'Blizzard', 'Tsunami', 'Cyclone', 'Wildfire']

    if seed_urls == [] or use_queries == True:
        for disaster in disasters:
            link = "https://www.google.com/search?q=" + disaster + " " + QUERY
            if timeframe == "":
                pass
            elif timeframe == 'day' or timeframe == 'd':
                link = link + PAST_DAY
            elif timeframe == 'week' or timeframe == 'w':
                link = link + PAST_WEEK
            elif timeframe == 'month' or timeframe == 'm':
                link = link + PAST_MONTH
            elif timeframe == 'year' or timeframe == 'y':
                link = link + PAST_YEAR
            seed_urls.append(link)
            if use_images == True:
                link = link + IMAGE
                seed_urls.append(link)

    for seed in seed_urls:
        if blacklistCheck(seed) == True:
            continue
        if (url_table.df['URL'].isin([seed]).any() == False): #Check if link is in the table already, if not, add the url
            if validateUrl(seed) == True:
                url_table.addURL(seed, None, INHERITANCE_COEFFICIENT)
            else:
                lg.exception("Invalid Seed URL")

def loadTables():
    if os.path.exists('res/ImageData.csv'):
        data = loadImageData()
        image_table = ImgTable(data)
    else:
        table = createImageTable()
        image_table = ImgTable(table)

    if os.path.exists('res/URLData.csv'):
        data = loadURLData()
        url_table = URLTable(data)
    else:
        table = createURLTable()
        url_table = URLTable(table)
    
    if os.path.exists('res/ResultsData.csv'):
        data = loadResultsData()
        results_table = ResultsTable(data)
    else:
        table = createResultsTable()
        results_table = ResultsTable(table)
    return image_table, url_table, results_table

def storeData():
    image_table.storeTable()
    url_table.storeTable()
    results_table.storeTable()

def disasterCheck(path):
    try: #Catches any corrupted images
        img = cv2.imread(path)
        resized_img = tf.image.resize(img, (256,256))
        yhat = disasterModel(np.expand_dims(resized_img/255, 0)) #.predict() function generates tensorflow tracing errors

        if yhat > 0.5: 
            return False #Predicted class is not a disaster
        else:
            return True #Predicted class is a disaster
    except Exception as e: 
        lg.exception('Issue with image {}'.format(path))
        return False #Removes invalid images from database

def blacklistCheck(url):
    for section in blacklist: #Skip urls with blacklisted strings
        if section in url:
            return True
            
def extractImages(img_urls, site):
    weight = 0
    for url in img_urls:
        if blacklistCheck(url) == True:
            continue
        filename = re.search(r'/([\w_-]+[.](jpeg|jpg|png))', url) #Find the image title from the url, between the .png, .jpg or .jpeg string and the last '/'
        if not filename or image_table.df['URL'].isin([url]).any():
            lg.info("URL has an incorrect or unwanted format or image url has already been scraped")
            continue #Avoid scraping url again
        current_time = datetime.now()
        filename = current_time.strftime("%d-%m-%YT%H-%M-%S.%f") + "." + str(filename.group(1).split(".", 1)[-1])
        path = "res/Images/" + filename
        try:
            urllib.request.urlretrieve(url, path)
            if os.path.exists(path):
                if disasterCheck(path) == True:
                    image_table.addImage(filename, url, site, current_time.strftime("%d-%m-%Y %H:%M:%S"))
                    weight += 1
                else:
                    try:
                        os.remove(path)
                    except:
                        lg.exception("Path does not exist: " + path)
        except (urllib.error.URLError) as err:
            lg.error("Failed to download {} due to {}".format(url, err))
            os.remove(path)
            if err.code() in [403, 406]:
                blacklist.append(urlparse(url).netloc)
            blacklist.append(path)
    return weight

def findImagePath(img_tag):
    try:
        return img_tag['src']
    except:
        try:
            return img_tag['data-src']
        except:
            try:
                return img_tag['data-srcset']
            except:
                try:
                    return img_tag['data-fallback-src']
                except:
                    pass

def validateUrl(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def focusedCrawler():
    new_urls = url_table.df.loc[url_table.df['Last Refresh'].isnull()] #Retrive rows where a refresh has not occured
    best_urls = new_urls.sort_values(by=['Weight'], ascending=False).head(max_workers)
    ranked_urls = [url for url in best_urls['URL']]
    return ranked_urls

def crawlModule(site, driver):
    # s = requests.Session()
    # retries = Retry(total=3, backoff_factor=0.5,)
    # s.mount('http://', HTTPAdapter(max_retries=retries))

    # try:
    #     response = s.get(site, headers={"User-Agent": "Mozilla"})
    # except:
    #     lg.exception("Failed URL request")
    #     return []
    lg.info("Thread is starting")
    driver.get(site)
    time.sleep(5) #Wait for page to load
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    #Initalise lists for scraped URLs
    img_urls = [] 
    href_urls = []
    img_tags = soup.find_all('img') #Find all image tags in the html scraped
    try:
        img_urls = [urljoin(site, findImagePath(img)) for img in img_tags] #Store all src attributes (url links) from img tags
        #print(img_urls)
    except:
        lg.exception("No src attribute")

    href_tags = soup.find_all('a', href=True) #Find all href tags in the html scraped
    try:
        href_urls = [urljoin(site, link['href']) for link in href_tags] #Store all href attributes (urls) from href tags
        #print(href_urls)
    except:
        lg.exception("No href attribute")

    weight = extractImages(img_urls, site)
    if show_scraped == True:
        print("Weight: " + str(weight) + " Scraped: " + site)
    url_table.updateURL(site, weight, DECAY_COEFFICIENT)

    for link in href_urls: #Check for new URLs to add to the URL table (Done after scraped url is evaluated to ensure correct weight is inherited)
        if blacklistCheck(link) == True:
            continue
        if link.startswith(('http://', 'https://')):
            if 'google' in link: #Strip google url redirect
                link = link.replace('https://www.google.com/url?q=', '')
                link = link.split("&sa=U", 1)[0]
            if url_table.df['URL'].isin([link]).any() == False: #Check if link is in the table already, if not, add the url
                url_table.addURL(link, site, INHERITANCE_COEFFICIENT)
    lg.info("Thread is Ending")
    return

def rankingModule():
    global url_queue
    best_urls = pd.DataFrame(url_table.df.sort_values(by=['Weight'], ascending=False).head(QUEUE_SIZE))
    #Drop urls that have been searched and did not return anything (Assuming the url is 'dead')
    best_urls = best_urls.drop(best_urls[(best_urls["Weight"] == 0) & (best_urls["Last Refresh"].notnull())].index.tolist())
    ranked_urls = [url for url in best_urls['URL']]
    queue_urls = [url[1] for url in url_queue]
    new_urls = set(ranked_urls) - set(queue_urls)
    purge_urls = set(queue_urls) - set(ranked_urls)

    if len(purge_urls) != 0: #Purge if purge url exists and empty list will purge entire list
        for i in range(len(url_queue)):
            if url_queue[i][1] in purge_urls:
                url_queue[i] = (2147483647, "") #Ensures purge items has the worst priority
        url_queue = sorted(url_queue)
        del url_queue[-(len(purge_urls)):] #Delete purge entries in url queue
        heapq.heapify(url_queue)

    if len(new_urls) != 0: #Ensures a null url is not pushed to the heap
        for url in new_urls:
            heapq.heappush(url_queue, (0, url))
            # priority = pd.to_datetime(url_table.getValue(url, 'Estimated Refresh Time')) - pd.to_datetime(datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
            # heapq.heappush(url_queue, (priority.total_seconds(), url))
    updatePriorities()
    return

def updateModule():
    for _ in range(len(url_queue)):
        site = heapq.heappop(url_queue)[1]
        updated_priority = pd.to_datetime(url_table.getValue(site, 'Estimated Refresh Time')) - pd.to_datetime(datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
        if updated_priority.total_seconds() <= 0:
            return site
        else:
            heapq.heappush(url_queue, (updated_priority.total_seconds(), site))
            updatePriorities()
    if url_queue == []:
        return None
    for i in range(max_workers):
        if focusedCrawler()[i] not in sites:
            return focusedCrawler()[i] #url queue is full and no estimated refresh time has passed so search a new url with the highest weight

def updatePriorities():
    for i in range(len(url_queue)):
        try:
            priority = pd.to_datetime(url_table.getValue(url_queue[i][1], 'Estimated Refresh Time')) - pd.to_datetime(datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
        except:
            print(url_queue[i])
        url_queue[i] = (priority.total_seconds(), url_queue[i][1])
    heapq.heapify(url_queue)

def recordResults():
    global crawl_time
    if use_incremental == True:
        crawler_type = 'Incremental'
    else:
        crawler_type = 'Focused'
    if use_threading == True:
        parallel_method = 'Threading'
    else:
        parallel_method = 'Multiprocessing'
    total_images = url_table.df['Total Images'].sum()
    results_table.addResult(crawler_type, parallel_method, max_workers, DECAY_COEFFICIENT, INHERITANCE_COEFFICIENT, crawl_time, total_images, total_scraped)
    crawl_time = crawl_time + 300

if __name__ == "__main__":
    PAST_DAY = "&tbs=qdr:d"
    PAST_WEEK = "&tbs=qdr:w"
    PAST_MONTH = "&tbs=qdr:m"
    PAST_YEAR = "&tbs=qdr:y"
    IMAGE = "&tbm=isch"
    QUERY = "Disaster Site Aerial Drone Images"

    #Default settings
    DECAY_COEFFICIENT = 0.9
    INHERITANCE_COEFFICIENT = 0.1
    QUEUE_SIZE = 1000
    use_seeds = True
    use_queries = False
    use_images = True
    use_incremental = True
    use_threading = True
    show_scraped = True
    max_workers = 8
    timeframe = "d"
    counter = 0
    total_scraped = 0
    end_time = 46500

    # command_args = sys.argv[1:] #Initialise setting from input
    # if len(command_args) > 0:
    #     processArguments()

    lg.basicConfig(filename="res/log.txt",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%d/%m/%YT%H:%M:%S')

    seed_urls = []
    blacklist = []
    url_queue = []
    try: 
        seed_urls, blacklist = initialiseFiles()
        image_table, url_table, results_table = loadTables()
        disasterModel = load_model('res/DisasterModel.h5')
    except:
        lg.error("Invalid initialised files")
        print("Invalid initialised files")
        exit()
    initialiseSeedUrls()

    drivers = [initialiseWebDriver(i) for i in range(max_workers)] #Initialise and add web drivers for every other thread
    
    if use_incremental == True:
        rankingModule() #Initialise url queue

    crawl_time = 0
    recordResults()
    record_scheduler = RecordScheduler(300, recordResults)
    record_scheduler.start()

    while (len(url_queue) != 0 or use_incremental == False) and crawl_time < end_time:
        sites = []
        if use_incremental == True:
            rankingModule()
            for _ in range(max_workers):
                site = updateModule()
                if site == None:
                    break
                sites.append(site)
        else:
            sites = focusedCrawler()
        if use_threading == True:
            with ThreadPoolExecutor(max_workers=max_workers+1) as executor: #+1 worker to account for result recording
                executor.map(crawlModule, sites, drivers)
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                executor.map(crawlModule, sites, drivers)
        total_scraped += len(sites)
        if use_incremental == True and url_queue == []: 
            rankingModule()

        if counter % 10 == 0:
            storeData()
        if counter % 20 == 0:
            print(url_table.df.sort_values(by=['Weight'], ascending=False).head(10))
        counter += 1
    storeData()
    [driver.quit() for driver in drivers]
    print("Url Queue is Empty or Time Condition Met")