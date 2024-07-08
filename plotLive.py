import matplotlib.pyplot as plt
import matplotlib.animation as animation

#Initialize plot
plt.ion()
fig, ax = plt.subplots()
ax.set_title("Live plot of model prediction for gesture activation")
ax.set_xlabel("sample #")
ax.set_ylabel("Gesture activation prediction")
ax.set_ylim([0, 1])
plot, = ax.plot(list(range(100)), [0 for _ in range(100)], 'b-', animated=True)

plt.pause(0.1)
bg = fig.canvas.copy_from_bbox(fig.bbox)
ax.draw_artist(plot)
fig.canvas.blit(fig.bbox)


def update_plot(y):
    fig.canvas.restore_region(bg)
    plot.set_ydata(y)
    ax.draw_artist(plot)
    fig.canvas.blit(fig.bbox)
    fig.canvas.flush_events()