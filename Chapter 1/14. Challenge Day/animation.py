
from matplotlib.pyplot import subplot
from matplotlib import animation
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import numpy as np, math, matplotlib.patches as patches

vector_x1 =2
vector_y1 =3
angle =45
fig,ax = plt.subplots()
patch = patches.Arrow(0, 0, vector_x1, vector_y1)
angle_new=angle*np.pi/180
magnitude = np.sqrt(vector_x1**2 + vector_y1**2)
angles = np.linspace(np.arctan(vector_y1/vector_x1),np.arctan(vector_y1/vector_x1)+angle_new,100)
def init():
    ax.add_patch(patch)
    return patch,


def animate(t):
    global patch

    ax.patches.remove(patch)

    patch = plt.Arrow(0, 0, magnitude*np.cos(angles[t]),magnitude*np.sin(angles[t]))
    ax.add_patch(patch)

    return patch,


anim = animation.FuncAnimation(fig, animate,
                                    init_func=init,
                                    interval=20,
                                    blit=False)


ax.set_xlim(-10,10)
ax.set_ylim(-10,10)
plt.show()