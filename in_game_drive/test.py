import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import ImageGrab
import numpy as np
import time
time.sleep(3)
# screen = np.array(ImageGrab.grab(bbox=(0,40, 1280, 740)))
screen = np.array(ImageGrab.grab(bbox=(20,630, 200, 710)))

plt.imshow(screen)

plt.show()