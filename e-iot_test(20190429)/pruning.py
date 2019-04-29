#-*- coding: utf-8 -*-

"""
    Copyright (C) 2019, Korea Electronics Technology Institute (KETI)
    Author: Changwoo Kim / cwkim@keti.re.kr / +82-10-9536-0610
    pruning.py : pruning raw data of SQL DB
"""

# Ignore the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# Data manipulation, visualization and useful functions
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format
pd.options.display.max_rows = 50
pd.options.display.max_columns = 40
import numpy as np
from tqdm import tqdm
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Necessary libraries
from multiprocessing import Process, Queue
import sys, os, time, logging, pprint
from logging import handlers
from datetime import datetime
import re, ast, json
# For Machbase DB
from machbaseAPI.machbaseAPI import machbase
# For Influx DB
from influxdb import InfluxDBClient

# machbase_connect: Remote access to Machbase DB (in KETI server)
def machbase_connect(id='SYS', pw='MANAGER', ip='127.0.0.1', port=5656):
    try:
        mclient = machbase()
        if mclient.open(ip, id, pw, port) is 0:
            print("Fatal error: " + mclient.result())
            # logging.error("Fatal error: {}".format(str(mclient.result())))
            return 0
        print("DB Connection: {}".format(str(mclient.isConnected())))
        # logging.info("DB Connection: {}".format(str(db.isConnected())))
        return mclient
    except Exception as err:
        logging.error("Failed to connect to Machbase DB {}".format(err))

# influx_connect: Remote access to Influx DB (in KETI server)
def influx_connect(id='keti', pw='keti1234', ip='127.0.0.1', port=8086):
    try:
        iclient = InfluxDBClient(ip, port, id, pw)
        return iclient
    except Exception as err:
        logging.error("Failed to connect to Influx DB {}".format(err))
        
# inserting_point: writing point as json data of ONP_DISP in Influx DB
def inserting_point(influxdb, json_data):
    db_name = "KETI"
    measurement_name = "ONP_DISP"
    json_body = [{
            'measurement' : measurement_name,
            'time' : datetime.strptime(json_data["TIMESTAMP"][:19], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%dT%H:%M:%SZ"),
            'tags' : { 'region':'asia-seoul' },
            'fields' : json_data
            }]

    if db_name in influxdb.get_list_database():
        influxdb.switch_database(db_name)
        # INSERT documents of MongoDB
        influxdb.write_points(json_body)
    else:
        # CREATE DATABASE <NAME>
        influxdb.create_database(db_name)
        influxdb.switch_database(db_name)
        # INSERT documents of MongoDB
        influxdb.write_points(json_body)
        
# Access to Machbase DB
machbase_db = machbase_connect(id='SYS', pw='MANAGER', ip='127.0.0.1', port=5656)

# Averaging each values during 10 sec
# 2018-11-01 ~ 2019-03-31
calendar = {'201811': ["2018-11-01 00:00:00", "2018-11-30 23:59:59"],
            '201812': ["2018-12-01 00:00:00", "2018-12-31 23:59:59"],
            '201901': ["2019-01-01 00:00:00", "2019-01-31 23:59:59"],
            '201902': ["2019-02-01 00:00:00", "2019-02-28 23:59:59"],
            '201903': ["2019-03-01 00:00:00", "2019-03-31 23:59:59"]
           }
LIMIT_ROWS = '1'

c_keys = list(calendar.keys())
for i in range(len(c_keys)):
    start_date = calendar.get(c_keys[i])[0]
    end_date = calendar.get(c_keys[i])[-1]
    query = ("SELECT DATE_TRUNC('SECOND', TIMESTAMP, 10) TIMESTAMP, "
         "AVG(DC_OUT_S) DC_OUT_S, AVG(WP_SPD_S) WP_SPD_S, AVG(WP_SPD_M) WP_SPD_M, AVG(DISP_GAP) DISP_GAP, AVG(DISP_DIL) DISP_DIL, "
         "AVG(WP_IN_S) WP_IN_S, AVG(HS_STM) HS_STM, AVG(HS_TMPT_M) HS_TMPT_M, AVG(HS_TMPT_O) HS_TMPT_O, "
         "AVG(DISP_PWR_M) DISP_PWR_M, AVG(WP_LOAD_B) WP_LOAD_B, AVG(WP_LOAD_T) WP_LOAD_T, AVG(DISP_PWR_S) DISP_PWR_S, "
         "AVG(DC1_LEV) DC1_LEV, AVG(PULP_TMPT_M) PULP_TMPT_M, AVG(DC_OUT_M) DC_OUT_M, AVG(DC2_LEV) DC2_LEV, "
         "AVG(DISP_DIL_S) DISP_DIL_S, AVG(HS_TMPT_S) HS_TMPT_S, AVG(DISP_ENG) DISP_ENG, AVG(DISP_VIB) DISP_VIB, "
         "AVG(DISP_DIL_M) DISP_DIL_M, AVG(WP_IN_M) WP_IN_M "
         "FROM (SELECT /*+ ROLLUP(ONP_DISP , SEC) */ TIMESTAMP, "
         "DC_OUT_S, WP_SPD_S, WP_SPD_M, DISP_GAP, DISP_DIL, WP_IN_S, HS_STM, HS_TMPT_M, HS_TMPT_O, DISP_PWR_M, "
         "WP_LOAD_B, WP_LOAD_T, DISP_PWR_S, DC1_LEV, PULP_TMPT_M, DC_OUT_M, DC2_LEV, DISP_DIL_S, HS_TMPT_S, DISP_ENG, "
         "DISP_VIB, DISP_DIL_M, WP_IN_M "
         "FROM ONP_DISP "
         "WHERE TIMESTAMP BETWEEN TO_DATE('{0}') AND TO_DATE('{1}'))"
         "GROUP BY TIMESTAMP ORDER BY TIMESTAMP;").format(start_date, end_date)
#          "GROUP BY TIMESTAMP ORDER BY TIMESTAMP LIMIT {2};").format(start_date, end_date, LIMIT_ROWS)

    if machbase_db.execute(query) is 0:
        print("Machbase DB Execute Error !!!")
#     logging.warn("Machbase DB Execute Error !!!")
    res = machbase_db.result()
    list_res = json.loads('['+res+']')

    # Inserting json data to Influx DB
    influx_db = influx_connect(id='keti', pw='keti1234', ip='127.0.0.1', port=8086)
    for point in list_res:
        # Coverting data type from string to float without 'TIMESTAMP'
        keys = [k for k in point.keys()]
        for key in keys:
            if not 'TIMESTAMP' == key:
                point[key] = float(point[key])
        inserting_point(influxdb=influx_db, json_data=point)
    print("Saved data between "+str(start_date)+" and "+str(end_date))
    
# if __name__ == '__main__':
#     # Logging the history
#     logging.basicConfig(filename='pruning.log', level=logging.INFO,
#                         format='%(asctime)s - %(message)s',
#                         datefmt='%Y-%m-%d %H:%M:%S')
#     LOG_MAX_BYTES = 10*1024*1024    # 10 MB
#     handlers.RotatingFileHandler(filename='pruning.log', maxBytes=LOG_MAX_BYTES, backupCount=5)
