# K-d tree

`Kdtree` is a C++ implementation of a [kdtree](https://en.wikipedia.org/wiki/K-d_tree).

<p align="center">
<img src=https://github.com/tle-huu/kdtree/wiki/images/kdtree_nn.png width=512 height=auto>
</p>

## Examples code
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
