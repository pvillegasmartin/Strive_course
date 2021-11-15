import numpy as np, math, matplotlib.patches as patches
from matplotlib import pyplot as plt
from matplotlib import animation

# Create figure
fig = plt.figure()
ax = fig.gca()

# Axes labels and title are established
ax = fig.gca()
ax.set_xlabel('x')
ax.set_ylabel('y')

ax.set_ylim(-2, 2)
ax.set_xlim(-2, 2)
plt.gca().set_aspect('equal', adjustable='box')

x = np.linspace(-1, 1, 20)
y = np.linspace(-1, 1, 20)
dx = np.zeros(len(x))
dy = np.zeros(len(y))

for i in range(len(x)):
    dx[i] = math.sin(x[i])
    dy[i] = math.cos(y[i])

vector_x1=3
vector_y1=3
patch = patches.Arrow(0, 0, vector_x1, vector_y1)
angle=45*np.pi/180
magnitude = 2

def init():
    ax.add_patch(patch)
    return patch,


def animate(t):
    global patch

    ax.patches.remove(patch)

    patch = plt.Arrow(0, 0, dx[t], dy[t])
    ax.add_patch(patch)

    return patch,


anim = animation.FuncAnimation(fig, animate,
                               init_func=init,
                               interval=20,
                               blit=False)

plt.show()