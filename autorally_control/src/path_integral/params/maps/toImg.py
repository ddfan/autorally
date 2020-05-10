#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import copy

filename = 'ccrf_costmap_09_29_2017'
data = np.load('ccrf_costmap_09_29_2017.npz')

X_cen_smooth = data['X_cen_smooth']
Y_cen_smooth = data['Y_cen_smooth']
W_cen_smooth = data['W_cen_smooth']
X_out = data['X_out']
Y_out = data['Y_out']
X_in = data['X_in']
Y_in = data['Y_in']
X_cen = data['X_cen']
Y_cen = data['Y_cen']
xBounds = data['xBounds']
yBounds = data['yBounds']
pixelsPerMeter = data['pixelsPerMeter']
channel0 = data['channel0']
channel1 = data['channel1']
channel2 = data['channel2']
channel3 = data['channel3']
filterChannel = data['filterChannel']

x_min = xBounds[0]
x_max = xBounds[1]
y_min = yBounds[0]
y_max = yBounds[1]

width = int((x_max - x_min) * pixelsPerMeter)
height = int((y_max - y_min) * pixelsPerMeter)


obstacles = copy.copy(channel0)
obstacle_min_val = 1.0
obstacle_max_val = 11.2
obstacles[obstacles > obstacle_max_val] = 0.0
obstacles[obstacles < obstacle_min_val] = 0.0
obstacles[obstacles >= obstacle_min_val] = 1.0
obstacles = obstacles * 255

desiredPixelsPerMeter = 1.0 / 0.02
desiredWidth = int((x_max - x_min) * desiredPixelsPerMeter)
desiredHeight = int((y_max - y_min) * desiredPixelsPerMeter)

obstacles = np.reshape(obstacles, (height, width))
img = Image.fromarray(obstacles)
img = img.resize((desiredWidth, desiredHeight))

# img.save('ccrf_costmap_09_29_2017.gif',format='gif')
# imgplot = plt.imshow(img)
# plt.show()


from skimage import measure

track = copy.copy(channel0)
track_max_val = 0.06
track[track > track_max_val] = 1.0
track[track <= track_max_val] = 0.0
track = track * 255

track = np.reshape(track, (height, width))
contours = measure.find_contours(track, 0.8, fully_connected='low')

# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(track, cmap=plt.cm.gray)

for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

plt.show()

waypoints = contours[0]

waypoints = waypoints / pixelsPerMeter

print(waypoints)

np.save('ccrf_costmap_09_29_2017_waypoints.npy', waypoints)
