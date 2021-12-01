# Yuvi_Chess_Trainer

This repository contains code and necessary files to run the Yuvi Chess Trainer. 

Yuvi Chess Trainer is an interactive chess training program that allows players to play multiplayer chess games, play against the chess computer Yuvi, or analyze their games to discover where they may improve. Yuvi uses a custom chess evaluation engine trained off of hundreds of thousands of chess positions to generate accurate evaluations that power its analysis.

Instructions and Notes for Running Yuvi Chess Trainer:

First, install necessary dependencies:

Install Numpy: pip3 install numpy
Install PyTorch: pip3 install torch torchvision
Install Pandas: pip3 install pandas
Install Joblib: pip3 install joblib
Install Tensorflow: pip3 install "tensorflow>=2.0.0"

Run the Following File:
TermProjectFinalMainFile.py

This file can be found through --> Application --> TermProjectFinalMainFile.py

When running file, ensure that all images and other.py files are within the same folder directory. 

IMPORTANT: In the TermProjectFinalMainFile.py, the app inializer function contains a section that loads the chess evaluation engine ml model. Please specify the directory in which the model is located on your computer. The file path currently chosen corresponds to where it is located on my computer. 

Note: the application takes a few moments to load as it is loading the machine learning model.

When in the application:
- Press 'r' to restart the application. This will lead you back to the home screen. 
- Press 'f' to flip the board and view different board orientations. 

Aside from the Application Folder you will find the Data and Models Folder. 

Data Folder
- This contains the csv file with chess positions the custom evaluation engine was trained off of. 

Models Folder
- This contains different chess evaluation engine model files that can be loaded into the application. model_cpu2 is the one you should load as this is the most accurate model. 
- You will also find the buildingMlModel.py file. This walks through how the chess evaluation engine ml model was compiled and trained. It includes all relevant citations and sources as well. 
