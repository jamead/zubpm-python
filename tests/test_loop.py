import matplotlib.pyplot as plt
import numpy as np
import time

plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111)

while True:
    data = np.random.rand(100)
    line1, = ax.plot(data)
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(1)
    plt.cla()     # clear previous data
