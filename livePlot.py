import matplotlib.pyplot as plt
import matplotlib.animation as animation

#Initialize plot
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111) #why 111?
ax.set_title("Live plot of model prediction for gesture activation")
ax.set_xlabel("sample #")
ax.set_ylabel("Gesture activation prediction")
ax.set_ylim([0, 1])
plot, = ax.plot(list(range(100)), [0 for _ in range(100)], 'b-')



# plt.show()

def update_plot(y):
    plot.set_ydata(y)
    fig.canvas.draw() 
    fig.canvas.flush_events() 