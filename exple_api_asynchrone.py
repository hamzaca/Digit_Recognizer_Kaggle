#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 03:32:41 2020

@author: hamza
"""



import requests
import json

import time


start_time = time.time()

#file_name = str(input(' image_path : ') )
file_name = "four.png"
data = {"image_path" : file_name}
response = requests.post('http://127.0.0.1:5000/', json=data)




print(" predicted digit  : " + str(response.json()))

print("time  = " + str(time.time()  - start_time) + " seconds " )



