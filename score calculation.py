'''
Iterate over all powertrain configurations

'''

import numpy as np
import matplotlib.pyplot as plt
import sys, os
import pandas as pd

results = pd.read_excel('powertrains.xlsx', sheet_name='Simulation')
motor_name = results['Motor_Name'].values
engine_name = results['ICE_Name'].values
acc_time = results['Acceleration_Time'].values
lap_time = results['Lap_Time'].values
lap_no = results['Lap_Number'].values

lap_no = np.clip(lap_no,a_min=0,a_max = 44)
acc_score = 10+15+75*(np.min(acc_time)/acc_time)
maxavg = 105/44
endurance_score = 35+52.5+262.5*(lap_no/np.max(lap_no))*((maxavg/lap_time-1)/(maxavg/np.min(lap_time)-1))
total_score = acc_score + endurance_score

from openpyxl import load_workbook

book = load_workbook('data\\powertrains.xlsx')
writer = pandas.ExcelWriter('data\\powertrains.xlsx', engine='openpyxl') 
writer.book = book

## ExcelWriter for some reason uses writer.sheets to access the sheet.
## If you leave it empty it will not know that sheet Main is already there
## and will create a new sheet.

writer.sheets = dict((ws.title, ws) for ws in book.worksheets)

data_filtered.to_excel(writer, "Main", cols=['Diff1', 'Diff2'])

writer.save()

df = pd.DataFrame({'Motor_Name': configs[:,0],'ICE_Name': configs[:,1],'Acceleration_Time': result[:,0],'Lap_Time': result[:,1],'Lap_Number': result[:,2]})
writer = pd.ExcelWriter('powertrains.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Simulation')
writer.save()
print("Simulation saved.")

print('finished!')



