# ECM2414-Software-Development-CA
PebbleGame Application 13 October 2021 -11 November 2021

FILE CONTENTS
================================================================================================================================================================================================================================
-Introduction
-Requirements
-Usage
-Known Issues
-Details

INTRODUCTION
================================================================================================================================================================================================================================
A web crawler that efficiently gathers raw aerial drone data of disaster sites available within the world wide web to be coalesced in a dataset that can be used to train AI for disaster response activities.

REQUIREMENTS
================================================================================================================================================================================================================================
beautifulsoup4==4.10.0
fake_useragent==1.1.3
keras==2.12.0
matplotlib==3.6.0
numpy==1.22.3
opencv_python==4.7.0.72
pandas==1.5.0
selenium==4.9.0
skimage==0.0
tensorflow==2.12.0
tensorflow_intel==2.12.0
tqdm==4.64.1
urllib3==1.26.8

USAGE
================================================================================================================================================================================================================================
Implementation Working Directory
python WebCrawler.py 
python GenerateDataset.py days

The Image Classifier is run without any arguements
Implementation/res/Generate Image Classifier/ Working Directory
python ImageClassifier.py

NOTE
================================================================================================================================================================================================================================
No image data used in this project is imported on github due to filesize.

KNOWN ISSUES
================================================================================================================================================================================================================================
Corrupt JPEG data: error messages from opening scraped images are printed to the terminal and are uncatchable since libjpeg error handling is not overwritten. This cannot be fixed unless specific parameters in the OpenCV package are manually edited. However, this issue has no detrimental effects on the program.

DETAILS
================================================================================================================================================================================================================================
Authors: Andrew Hin Chong Yau
License: GNU General Public License v3.0
Date: 23/04/2023
