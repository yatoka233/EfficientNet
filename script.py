import math
import numpy as np
import os

import matplotlib.pyplot as plt

# lrf = 0.01
# lf = lambda x: ((1 + math.cos(x * math.pi / 5)) / 2) * (1 - lrf) + lrf
#
#
# y = []
# x = np.arange(5)
#
# for i in x:
#     y.append(lf(i))
#
#
# plt.plot(x, y)
# plt.show()
# print(y)


path = r'E:\Agent\Paper'
dir = []
def findall(path):
    for i in os.listdir(path):
        full_path = os.path.join(path, i)
        if os.path.isdir(full_path):
            dir.append(full_path)
            findall(full_path)
        else:
            print(full_path)

findall(path)

print('dirs:')
for i in dir:
    print(i)