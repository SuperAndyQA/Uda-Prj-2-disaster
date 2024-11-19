# Uda-Prj-2-disaster
## Overview
This is Project 2 - Disaster Response in Udacity Data Scientist Nano-degree, in which Twitter responses from CSV files are extracted, transformed and loaded into SQL Lite DB, before NLP/Machine Learning pipelines are implemented to tokenize, extract features, model and evaluate to give insight about nature of responses in disaster and support decision making actions.
## File Repository
+---app
|  \---templates
|          <> go.html
|          <> master.html
+---data
|          categories.csv
|          DisasterResponse.db
|          messages.csv
|          process_data.py
\---models
|          train_classifier.py
|---README.md

## How to run
Repository comprises of different folders for different purposes:
1. data: this folder contains raw data files (categories.csv, messages.csv) and process_data.py that run ETL code to transform the data of Twitter messages to load into DisasterResponse.db file (SQL Lite).
To run the the ETL process:
- First, you need navigate into data folder using following command
"cd data"
- 
- 
