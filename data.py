#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv

with open("All_Data1.csv", "rb") as f:
    r = csv.reader(f)
    data = list(r)

clean_data, samples, target  = [], [], []

for d in range (1,len(data)):
    if ((data[d].count("NULL") +data[d].count("#N/A")+ data[d].count("PrivacySuppressed")) <= (0.25 * len(data[d])) and \
    data[d][-1] not in ["NULL", "#N/A", "PrivacySuppressed"]):
        curr = []
        for i in range (1, len(data[d])-1):
            if data[d][i] in ["NULL", "#N/A", "PrivacySuppressed"]:
                 curr.append(-1)
            else: curr.append(float(data[d][i]))

        samples.append(curr[0:-1])
        target.append(int(data[d][-1])/15000 if int(data[d][-1])/15000 < 8 else 8)

print ([[i*15000, target.count(i)] for i in range(0,8)])