from sed_trans import *
import csv
import numpy as np
import matplotlib.pyplot as plt


D50 = 0.0061
rho_sediment=1048.0

L = 100.0
number_nodes =201
dx = L/(number_nodes-1)

Q = np.genfromtxt('Q.csv', delimiter=',')
A = np.genfromtxt('A.csv', delimiter=',')
Z = np.genfromtxt('Z.csv', delimiter=',')
Zbed = np.genfromtxt('Zbed.csv', delimiter=',')

h = Z - Zbed
u = Q/A

x = np.zeros(number_nodes)
Qbed_star = np.zeros(number_nodes)

for i in range(len(x)):
        x[i] = i*dx
        Qbed_star[i] = get_unit_bed_load(h[i],u[i], D50,rho_sediment )

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.plot(x,Qbed_star)
plt.show()