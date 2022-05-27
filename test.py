import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('./Data/ID#31127011_2WProsthUse/VirtualGameData/VirtualArmGames_-2_4_2021_16_34_19.csv')

# print(data.head())

pre_ramp_speed = data['PreRampSpeed']
post_ramp_speed = data['PostRampSpeed']
motion_class = data['class']

# plt.hist(pre_ramp_speed, bins=13, density=False, histtype='step')
# plt.title("Pre Ramp Speed")
# plt.xlabel("Speed Percentages")
# plt.ylabel("Counts")
# plt.show()

pre_counts = pre_ramp_speed.value_counts()
pre_array = pre_counts.to_numpy().reshape(-1,1)
print(pre_counts)
print(pre_array)
