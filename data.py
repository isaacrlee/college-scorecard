#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv

with open("All_Data1.csv", "rb") as f:
    r = csv.reader(f)
    data = list(r)

income, samples, target  = [], [], []

for d in range (1,len(data)):
    if ((data[d].count("NULL") +data[d].count("#N/A")+ data[d].count("PrivacySuppressed")) <= (0.25 * len(data[d])) and \
    data[d][-1] not in ["NULL", "#N/A", "PrivacySuppressed"]):
        curr = []
        for i in range (1, len(data[d])-1):
            if data[d][i] in ["NULL", "#N/A", "PrivacySuppressed"]:
                 curr.append(-1)
            else: curr.append(float(data[d][i]))

        samples.append(curr[0:-1])
        income.append(float((data[d][-1])))
        # if int(data[d][-1]) <20000: target.append(0)
        # elif int(data[d][-1]) <40000: target.append(1)
        # elif int(data[d][-1]) <60000: target.append(2)
        # elif int(data[d][-1]) <80000: target.append(3)
        # else: target.append(4)

        if int(data[d][-1]) <=23800: target.append(0)
        elif int(data[d][-1]) <=30600: target.append(1)
        elif int(data[d][-1]) <=38000: target.append(2)
        else: target.append(3)

print ([[target.count(i)] for i in range(0,4)])