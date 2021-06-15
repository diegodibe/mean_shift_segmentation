from imageSegmentation import scale
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

results = pd.read_csv('mean_shift_evaluation.csv').drop(['Unnamed: 0'], axis=1)
print(results)
chunk = results.loc[results['radius'] == 0.1]
print(chunk)
chunk_3d = chunk.loc[chunk['dim_5'] == False]
chunk_5d = chunk.loc[chunk['dim_5'] == True]
print(chunk_3d)
c_fixed = results.loc[results['c'] == 4]
c_fixed_3d = c_fixed.loc[c_fixed['dim_5'] == False]
c_fixed_5d = c_fixed.loc[c_fixed['dim_5'] == True]

fig, ax = plt.subplots(2, 2)
fig.suptitle(f'visualization of the parameters')
ax[0, 0].set_xlabel('radius')
ax[0, 0].xaxis.set_label_position('top')
ax[0, 0].set_ylabel('clusters')
ax[0, 0].set_ylim(-8, 300)
ax[0, 0].plot(c_fixed_3d['radius'], c_fixed_3d['clusters'], label='3D')
ax[0, 0].plot(c_fixed_5d['radius'], c_fixed_5d['clusters'], label='5D')
ax[0, 0].legend()

ax[1, 0].set_xlabel('c')
ax[1, 0].set_ylabel('clusters')
ax[1, 0].plot(chunk_3d['c'], chunk_3d['clusters'], label='3D')
ax[1, 0].plot(chunk_5d['c'], chunk_5d['clusters'], label='5D')
ax[1, 0].legend()

ax[0, 1].set_xlabel('radius')
ax[0, 1].xaxis.set_label_position('top')
ax[0, 1].set_ylabel('time')
ax[0, 1].plot(c_fixed_3d['radius'], c_fixed_3d['time'], label='3D')
ax[0, 1].plot(c_fixed_5d['radius'], c_fixed_5d['time'], label='5D')
ax[0, 1].legend()

ax[1, 1].set_xlabel('c')
ax[1, 1].set_ylabel('time')
ax[1, 1].plot(chunk_3d['c'], chunk_3d['time'], label='3D')
ax[1, 1].plot(chunk_5d['c'], chunk_5d['time'], label='5D')
ax[1, 1].legend()
plt.savefig('images/parameters.jpg')
plt.show()

