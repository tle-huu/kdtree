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
neighbors = []
radius = 0

with open("kdtree.txt") as f:
        line_number = 1
        line = f.readline().strip()
        while line:
                if line[0] == "#":
                    if line[2:] == "points":
                        line = f.readline().strip()
                        line_number += 1
                        while line and line[0] != "#":
                            line = line.split(",")
                            points.append((float(line[0]), float(line[1])))
                            line = f.readline().strip()
                            line_number += 1
                    elif line[2:] == "gold":
                        line = f.readline().strip()
                        line_number += 1
                        line = line.split(",")
                        gold = (float(line[0]), float(line[1]))
                        line = f.readline().strip()
                        line_number += 1
                    elif line[2:] == "nn":
                        line = f.readline().strip()
                        line_number += 1
                        line = line.split(",")
                        nn = (float(line[0]), float(line[1]))
                        line = f.readline().strip()
                        line_number += 1
                    elif line[2:] == "neighbors":
                        line = f.readline().strip()
                        line_number += 1
                        radius = float(line)
                        line = f.readline().strip()
                        line_number += 1
                        while line and line[0] != "#":
                            line = line.split(",")
                            neighbors.append((float(line[0]), float(line[1])))
                            line = f.readline().strip()
                            line_number += 1
                    else:
                        print(f"Parsing Error: unknown line {line_number}: '{line.strip()}'")
                        sys.exit(1)

# Plotting
fig, ax = plt.subplots()
fig.suptitle('Neighbors Search with Kdtree')
ax.set_title(f"Radius = {radius}")
# points
plt.scatter([p[0] for p in points], [p[1] for p in points], color = 'black', zorder = 3)


# neighbors
plt.scatter([p[0] for p in neighbors], [p[1] for p in neighbors], color = 'yellow', zorder = 11)
d = plt.Circle((gold[0], gold[1]), radius, fill = False, color = 'y', zorder = 14)
ax.add_patch(d)

# gold point
plt.scatter([gold[0]], [gold[1]], color = 'red', zorder = 11)

# nearest neighbor
plt.scatter([nn[0]], [nn[1]], color = 'green', zorder = 11)

# Circle whose radius is distance to nn
c = plt.Circle((gold[0], gold[1]), np.linalg.norm(np.array(gold) - np.array(nn)), fill = False, color = 'green', zorder = 14)
ax.add_patch(c)
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.gca().set_aspect('equal', adjustable = 'box')
plt.show()
