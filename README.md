# Kdtree

`Kdtree` is a C++ implementation of a [kdtree](https://en.wikipedia.org/wiki/K-d_tree).

<p align="center">
<img src=https://github.com/tle-huu/kdtree/wiki/images/kdtree_nn.png width=512 height=auto>
</p>

## Description

* Simple
* Lightweight
* Static Kdtree
* Header only
* Nearest neighbor search
* k-Nearest neighbors search
* Neighbors search within a radius

## Documentation

The Kdtree needs to be provided a GetCoord function along with the object type and the dimension.
The GetCoord function is meant to access the n-th coordinate of the objects.

## Examples code

An example of code is provided in examples/.

The program generates some random 2D points and creates a Kdtree from them.
It then generates a random 2D point and searches for its nearest neighbors and its neighbors within a radius.

### How to build
```
$ git clone https://github.com/tle-huu/kdtree.git
$ cd kdtree
$ mkdir build
$ cd build
$ cmake ..
$ make
```

### How to run
```
$ cd build/examples
$ ./find_nn
$ python3 draw.py
```
- find_nn
    - Generate random 2D points and create a kdtree.
    - Generate a random point whose neighbors we want to find
    - Dump Kdtree points, the random point, and its neighbors in a file 'kdtree.txt'
- draw.py
    - Parse 'kdtree.txt' and plot the points and found neighbors

## License

Distributed under the MIT License.
