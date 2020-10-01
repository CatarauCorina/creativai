import csv
import pandas as pd
import matplotlib.pyplot as plt
import os

dir_files = os.listdir(os.getcwd())

dfs = []

for idx, filename in  enumerate(dir_files):
    print(filename)
    if filename != 'comp_fetch.py':
        big_frame = pd.read_csv(filename)
        big_frame = big_frame[:250]
        # if filename == 'run-fetch_dqn_conv_network_832_cuda_0.999_0.9_0.05_200-tag-Reward_episode.csv':
        #     big_frame = big_frame[:250]
        print(len(big_frame[big_frame['Value'] > 0]) / len(big_frame) * 100)
        print(f'Mean rewards: {big_frame[big_frame["Value"] > 0].mean().Value:.20f}')
        print(f'Min reward: {big_frame[big_frame["Value"] > 0].min().Value:.20f}')
        print(f'Max reward: {big_frame[big_frame["Value"] > 0].max().Value:.20f}')
        big_frame['Value'].plot.line()
        plt.show()

# big_frame = pd.concat(dfs, ignore_index=True)
# big_frame['Value'].plot.line()
# plt.show()

# print(len(big_frame[big_frame['Value'] > 0])/len(big_frame)*100)
# print(f'Mean rewards: {big_frame[big_frame["Value"] > 0].mean().Value:.20f}')
# print(f'Min reward: {big_frame[big_frame["Value"] > 0].min().Value:.20f}')
# print(f'Max reward: {big_frame[big_frame["Value"] > 0].max().Value:.20f}')
# print(len(big_frame))
