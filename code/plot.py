import numpy as np
import matplotlib.pyplot as plt

T1_data_512 = np.load('Tree1-Data-512.npy')
T1_data_256 = np.load('Tree1-Data-256.npy')
T1_data_128 = np.load('Tree1-Data-128.npy')

T1_error_512 = np.load('Tree1-Error-512.npy')
T1_error_256 = np.load('Tree1-Error-256.npy')
T1_error_128 = np.load('Tree1-Error-128.npy')

T2_data_512 = np.load('Tree2-Data-512.npy')
T2_data_256 = np.load('Tree2-Data-256.npy')
T2_data_128 = np.load('Tree2-Data-128.npy')

T2_error_512 = np.load('Tree2-Error-512.npy')
T2_error_256 = np.load('Tree2-Error-256.npy')
T2_error_128 = np.load('Tree2-Error-128.npy')


axis = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])

fig1 = plt.figure(figsize=(8, 10), dpi=300)
plt.errorbar(axis, T1_data_512, yerr=T1_error_512, elinewidth=1, marker='o', color='black', linestyle='--', label='D=512')
plt.errorbar(axis+0.2, T1_data_256, yerr=T1_error_256, elinewidth=1, marker='o', color='red', linestyle='--',label='D=256')
plt.errorbar(axis+0.4, T1_data_128, yerr=T1_error_128, elinewidth=1, marker='o', color='blue', linestyle='--',label='D=128')
plt.xlabel('Number of Constituents', fontsize=10)
plt.ylabel('E(a)', fontsize=10)
plt.title('Comparison for Tree 1', fontsize=10)
plt.xlim([0,17])
plt.xticks(fontsize = 8)
plt.yticks(fontsize = 8)
plt.legend(fontsize='small', loc='upperleft')
fig1.set_size_inches(7,5)
fig1.savefig('test11')

axis = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14])

fig1 = plt.figure(figsize=(8, 10), dpi=300)
plt.errorbar(axis, T2_data_512, yerr=T2_error_512, elinewidth=1, marker='o', color='black', linestyle='--',label='D=512')
plt.errorbar(axis+0.2, T2_data_256, yerr=T2_error_256, elinewidth=1, marker='o', color='red', linestyle='--',label='D=256')
plt.errorbar(axis+0.4, T2_data_128, yerr=T2_error_128, elinewidth=1, marker='o', color='blue', linestyle='--',label='D=128')
plt.xlabel('Number of Constituents', fontsize=10)
plt.ylabel('E(a)', fontsize=10)
plt.title('Comparison for Tree 2', fontsize=10)
plt.xlim([0,15])
plt.xticks(fontsize = 8)
plt.yticks(fontsize = 8)
plt.legend(fontsize='small', loc='upperleft')
fig1.set_size_inches(7,5)
fig1.savefig('test22')
