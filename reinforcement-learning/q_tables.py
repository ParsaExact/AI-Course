import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
df = pd.read_csv("my_data.csv")
snake1 = df["snake_1"]
snake2 = df["snake_2"]
avrg1 = []
avrg2 = []
for i in range(len(snake1)//100):
    avrg1.append(np.mean(snake1[i:i+100]))
    avrg2.append(np.mean(snake1[i:i+100]))

plt.subplot(1, 2, 1)
plt.plot(range(len(avrg1)), avrg1)
plt.xlabel("episode")
plt.ylabel("snake 1 reward")
plt.subplot(1, 2, 2)
plt.plot(range(len(avrg2)), avrg2)
plt.xlabel("episode")
plt.ylabel("snake 2 reward")
plt.show()
with open('Qtabel1.pickle', 'rb') as f:
    data1 = pickle.load(f)

with open('Qtabel2.pickle', 'rb') as f:
    data2 = pickle.load(f)

# Save as a .npy file
np.save('s1_qtble.npy', data1)
np.save('s2_qtble.npy', data2)