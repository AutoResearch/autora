import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import cnnsimple.plot_utils as plot_utils

window = plot_utils.DebugWindow()

for phase in np.linspace(0, 10*np.pi, 500):
    window.update(phase, 0)


# x = np.linspace(0, 6*np.pi, 100)
# y = np.sin(x)
#
# plt.ion()
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# line1, = ax.plot(x, y, 'r-')
# plt.draw()
#
# for phase in np.linspace(0, 10*np.pi, 500):
#     line1.set_ydata(np.sin(x + phase))
#     plt.draw()
#     plt.pause(0.02)
#
# plt.ioff()
# plt.show()