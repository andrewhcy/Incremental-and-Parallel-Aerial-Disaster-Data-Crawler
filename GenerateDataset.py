from Dataframes import *
import os
import pandas as pd
import shutil
import sys
import time
from datetime import datetime, timedelta

def loadImageTable():
    if os.path.exists('res/ImageData.csv'):
        data = loadImageData()
        image_table = ImgTable(data)
    else:
        print("Image Data file does not exist in the res directory")
    return image_table

if __name__ == "__main__":
    command_args = sys.argv[1:]

    timeframe = 1
    try:
        if len(command_args) > 0:
            if isinstance(command_args[0], float) and command_args[0] > 0:
                timeframe = command_args[0]
    except:
        print("Invalid Timeframe. Please pass time in days as a float as the first argument.")
        exit()
        
    current_time = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    source_directory = "res/Images/"
    target_directory = "res/Datasets/" + str(current_time) + " " + str(timeframe) + "days"
    os.mkdir(target_directory)

    image_table = loadImageTable()
    image_freshness = pd.to_datetime(current_time) - timedelta(days=timeframe)
    image_freshness = image_freshness.strftime("%d-%m-%Y %H-%M-%S")
    mask = (image_table.df['Download Date And Time'] > datetime.strptime(image_freshness, "%d-%m-%Y %H-%M-%S")) &(image_table.df['Download Date And Time'] <= datetime.strptime(current_time, "%d-%m-%Y %H-%M-%S"))
    target_images = image_table.df.loc[mask]
    #print(target_images)
    copy_images = [filename for filename in target_images['Image Filename']]
    
    for image in copy_images:
        try:
            shutil.copy((source_directory + image), target_directory)
        except:
            print("Image not found")