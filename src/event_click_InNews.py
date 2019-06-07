# -*- coding: utf-8 -*-
"""
Created on Sat May 25 13:35:42 2019

@author: Gulaer
"""

import json
import pandas as pd
import mmap
from tqdm import tqdm
from flatten_json import flatten
import argparse
from datetime import datetime, timedelta

parser = argparse.ArgumentParser(description='')
parser.add_argument("-b", dest="begin_day", type=str, help="event begin day, ex. 2019,3,21")
parser.add_argument("-e", dest="end_day", type=str, help="event end day, ex. 2019,5,20")
args = parser.parse_args()

def get_event_date(begin_day,end_day):
    bd = begin_day.split(',') 
    bd = [int(x) for x in bd]
    ed = end_day.split(',') 
    ed = [int(x) for x in ed]
    begin_date = datetime(bd[0],bd[1],bd[2]).date()
    end_date = datetime(ed[0],ed[1],ed[2]).date()
    eventdate = []
    a_day = timedelta(days=1)
    while begin_date <= end_date:
        eventdate.append(str(begin_date))
        begin_date += a_day
    return eventdate

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

def invalid_raw_click_event(event_json):
    if ("page.item.id" not in event_json.keys()) or event_json["page.item.id"] == '':
        return True
    if ("eds_id" not in event_json.keys()) or event_json["eds_id"] == '':
        return True
    if ("event_timestamp" not in event_json.keys()) or event_json["event_timestamp"] == '':
        return True
    
    return False

def read_events_JSON(path, names='all'):
    print('Reading from {} ...'.format(path))
    names = names if names != 'all' else set()
    events = []
    with open(path, 'r', encoding="utf-8") as f:
        for line in tqdm(f, total=get_num_lines(path)):
            event = flatten(json.loads(line), separator='.')
            if invalid_raw_click_event(event):
                continue
            names = names if names != 'all' else event.keys()
            filterd_event = {k:v for k,v in event.items() if k in names}

            events.append(filterd_event)
    
    events = pd.DataFrame(events)
    return events

def convert_type_to_category(data_f, inplace=False):
    # base on ratio of unique items to convert datatype and reduce memory
    # https://www.dataquest.io/blog/pandas-big-data/
    df = data_f if inplace else data_f.copy()
    for col in df.columns:
        ratio =  len(df[col].unique()) / len(df[col])
        print("Unique ratio of attribute \"{}\" = {}".format(col, ratio))
        if ratio < 0.5:
            print('Convert column \'{}\' to dtype \'category\''.format(col))
            df[col] = df[col].astype('category')
    return df

def remove_extreme_users(events, min_click, max_click):
    """
    Args:
        events - pandas  DataFrane
                 click events need to remove extreme users that we will not give personalized recommendations
        min_click - int minimum number of click per day
        max_click - int maximum number of click per day
        
    Return:
        New DataFrame of cleaned click events
    """
    user_clicked_number = events.groupby('cookie_mapping.et_token').size()
    common_user = user_clicked_number[user_clicked_number > min_click]
    common_user = common_user[common_user < max_click]
    return events[events['cookie_mapping.et_token'].isin(common_user.index)]

if __name__ == '__main__':
    
    begin_day = args.begin_day
    end_day = args.end_day
    eventdate = get_event_date(begin_day,end_day)
    
    events_path_list = ['../data/ettoday_event_{}.json'.format(x)
                         for x in eventdate] 
    
    attributes = ['page.item.id','eds_id','event_timestamp']
    for i, events_path in enumerate(events_path_list):
        datas = read_events_JSON(events_path, names=attributes)
        datas = convert_type_to_category(datas)
        datas = datas.rename(columns = {"page.item.id": "news_id", 
                                        "eds_id":"user_id", 
                                        "event_timestamp": "event_time"}) 
    
        pkl_path = '../data/event-'+eventdate[i]+'.pkl'
        datas.to_pickle(pkl_path)
    