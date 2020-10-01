import csv
import pandas as pd
import matplotlib.pyplot as plt
import os

dir_files = os.listdir(os.getcwd())

dfs = []

for filename in dir_files:
    if filename != 'run_results.py' and filename != 'compare_fetch':
        dfs.append(pd.read_csv(filename))

big_frame = pd.concat(dfs, ignore_index=True)
# big_frame['Value'].plot.line()
# plt.show()


big_frame[big_frame['Value'] > 0].sort_values()


print(len(big_frame[big_frame['Value'] > 0])/len(big_frame)*100)
print(f'Mean rewards: {big_frame[big_frame["Value"] > 0].mean().Value:.20f}')
print(f'Min reward: {big_frame[big_frame["Value"] > 0].min().Value:.20f}')
print(f'Max reward: {big_frame[big_frame["Value"] > 0].max().Value:.20f}')
print(len(big_frame))
