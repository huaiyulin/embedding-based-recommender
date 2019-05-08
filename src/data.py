# -*- coding: utf-8 -*-
import json
import pandas as pd
import json
from flatten_json import flatten

news_id    = 'page.item.id'
user_id    = 'cookie_mapping.et_token'
event_time = 'event_timestamp'
check_list = [news_id,user_id,event_time]

f = open('../data/ettoday_event_2019-03-09.json','r')

i = 0
while True:
    i +=1
    line = f.readline()
    event = flatten(json.loads(line), separator='.')
    if all([ x in event.keys() for x in check_list]):
        print(line)
        print(event[user_id],event[news_id],event[event_time])
