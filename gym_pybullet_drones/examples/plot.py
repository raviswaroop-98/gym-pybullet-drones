import numpy as np
import matplotlib.pyplot as plt


default = "/Users/raviswarooprayavarapu/Documents/gym-pybullet-drones/gym_pybullet_drones/examples/results"
path = default+"/save-hover-ddpg-kin-rpm-11.23.2022_23.49.26"
mod = path + "/evaluations.npz"
data = np.load(mod)

#x = data['timesteps']
y = data['results'].flatten()
x = np.arange(0,len(y))
plt.plot(x,y)
plt.show()
