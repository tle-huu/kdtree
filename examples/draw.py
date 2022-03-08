import numpy as np
import matplotlib.pyplot as plt
import statistics
import sys

class Node:
        def __init__(self, value=None, left=None, right=None):
                self.left = left
                self.right = right
                self.value = value


points = []
gold = None
nn = None

with open("kdtree.txt") as f:
        line = f.readline()
        while line:
                if line[:4] == "gold":
                        line = line[5:].split(",")
                        gold = (float(line[0]), float(line[1]))
                elif line[:2] == "nn":
                        line = line[4:].split(",")
                        nn = (float(line[0]), float(line[1]))
                elif line[:4] != "null":
                        line = line.split(",")
                        points.append((float(line[0]), float(line[1])))
                line = f.readline()

# Plotting
fig, ax = plt.subplots()
plt.scatter([p[0] for p in points], [p[1] for p in points], color = 'black', zorder = 10)
plt.scatter([gold[0]], [gold[1]], color = 'red', zorder = 11)
plt.scatter([nn[0]], [nn[1]], color = 'green', zorder = 11)
c = plt.Circle((gold[0], gold[1]), np.linalg.norm(np.array(gold) - np.array(nn)), fill = False, color = 'y', zorder = 14)
ax.add_patch(c)
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.gca().set_aspect('equal', adjustable = 'box')
plt.show()
