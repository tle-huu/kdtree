#include <algorithm>
#include <fstream>
#include <iostream>

#include "Kdtree.h"

struct Point
{
        float x;
        float y;
};

Point make_random_point()
{
        return Point{
            (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * 10.0f,
            (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) *
                10.0f};
}

int main()
{
        srand(static_cast<unsigned>(time(0)));
        std::ofstream file("kdtree.txt");
        // Tree from the Wikipedia article
        // std::vector<Point> points = {{2,3}, {5,4}, {9,6}, {4,7}, {8,1}, {7,2}
        // };

        std::vector<Point> points;
        for (size_t i = 0; i < 50; ++i)
        {
                points.emplace_back(make_random_point());
                file << points[i].x << "," << points[i].y << '\n';
        }

        auto get_coord = [](const Point& p, int axis) {
                if (axis == 0) return p.x;
                return p.y;
        };
        kdtree::Kdtree<Point, decltype(get_coord)> tree(points, get_coord);

        Point gold = make_random_point();
        const Point& nn = tree.find_nn(gold);

        file << "gold: " << gold.x << "," << gold.y << '\n';
        file << "nn: " << nn.x << "," << nn.y << '\n';

        return 0;
}
