from utils import *
import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
import scipy

# 1) import data from .mat file
# what I have: a path to a .mat file
# what I want: every variable as a separate np array from the mat file
path_to_mat_file = 'dataForSebastian_SF_VTAoptoData.mat'
data_dict = import_data_from_mat(path_to_mat_file)
BBB1_data = data_dict['Bdata'][0]
Cdata1_data = data_dict['Cdata'][0]

BBB1_data_smoothed = smooth(BBB1_data, window_len=10, window='hanning')
Cdata1_data_smoothed = smooth(Cdata1_data, window_len=1, window='hanning')

# 2) plot the BB data
plt.plot(BBB1_data, "k-")
plt.plot(BBB1_data_smoothed, "r--")
plt.ylabel('BBB')
plt.show()

# 3) preprocess data for pysindy
# what I have: every variable as a separate np array from the mat file
# what we want: we want a variable x and a variable y, where x is time and y is data
period = 100
offset = 000
offset_x = 0 + offset
offset_y = 0 + offset
normalize = 0.05
y = BBB1_data_smoothed[0:period]
y_data_org = BBB1_data[0+offset_y:period+offset_y] * normalize
x_data_org = Cdata1_data[0+offset_x:period+offset_x] * normalize
y_data = BBB1_data_smoothed[0+offset_y:period+offset_y] * normalize
x_data = Cdata1_data_smoothed[0+offset_x:period+offset_x] * normalize
x = x_data_org
y = y_data_org
dx = Cdata1_data[1+offset:period+1+offset] - Cdata1_data[0+offset:period+offset]
dy = BBB1_data[1+offset:period+1+offset] - BBB1_data[0+offset:period+offset]
# t = np.arange(len(y))

# 26 minutes total
# smooth BBB to about 0.7 seconds
# time step equals 0.07 seconds

t = np.linspace(0, 1, period)
# x = 3 * np.exp(-2 * t)
# y = 0.5 * np.exp(t)

plt.plot(t,y_data_org, "k-")
plt.plot(t,x_data_org, "b-")
plt.plot(t,y_data, "k--")
plt.plot(t,x_data, "b--")
plt.plot(t,x, "r-")
plt.plot(t,y, "g-")
plt.ylabel('Y')
plt.show()

# feed data to pysindy
X = np.stack((x, y), axis=-1)

# Initialize two libraries
#poly_library = ps.PolynomialLibrary(degree=2)
#fourier_library = ps.FourierLibrary()
# Initialize this generalized library, all the work hidden from the user!
#generalized_library = ps.GeneralizedLibrary([poly_library, fourier_library])

# run pysindy
model = ps.SINDy(feature_names=["x", "y"])
#model = ps.SINDy(feature_library=generalized_library,feature_names=["x", "y"])
model.fit(X, t=t)
# show results from pysindy
model.print()

# print the fit of the model
x0 = x[0]
y0 = y[0]
t0 = t[0]
t_test = t
sim = model.simulate([x0, y0], t=t_test)

plt.figure()
plt.plot(t0, x0, "ro", label="Initial condition", alpha=0.6, markersize=8)
plt.plot(t, x, "b", label="Original data for x", alpha=0.4, linewidth=4)
plt.plot(t_test, sim[:, 0], "k--", label="SINDy model", linewidth=3)
plt.xlabel("t")
plt.ylabel("x")
plt.plot(t,y)
plt.legend()
plt.show()

data_dict2 = import_data_from_mat('oldDataForSebastian_SF_VTAoptoData.mat')
y2 = data_dict2['Bdata'][0][200:200+period] * normalize
x2 = data_dict['Cdata'][0][200:200+period] * normalize


plt.figure()
plt.plot(t,y)
plt.plot(t,y2)
plt.show()

plt.figure()
plt.plot(t,x)
plt.plot(t,x2)
plt.show()

plt.figure()
plt.plot(t,x)
plt.plot(t,y)
plt.show()

plt.figure()
plt.plot(t,x2)
plt.plot(t,y2)
plt.show()

plt.figure()
plt.plot(t,x)
plt.plot(t,x2)
plt.plot(t,y)
plt.plot(t,y2)
plt.show()

plt.figure()
plt.plot(t,x_data)
plt.plot(t,y_data)
plt.show()

'''
plt.figure()
plt.plot(x,y)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
plt.figure()
plt.plot(x,dx)
plt.show()
plt.figure()
plt.plot(y,dy)
plt.show()
plt.figure()
plt.plot(dx,y)
plt.show()
plt.figure()
plt.plot(dy,x)
plt.show()
plt.figure()
plt.plot(dy,dx)
plt.show()
print(scipy.stats.pearsonr(y,dy))
print(scipy.stats.pearsonr(x,dx))
'''

print(scipy.stats.pearsonr(x,x2))
print(scipy.stats.pearsonr(y,y2))