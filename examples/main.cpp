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
        //};

        // Generate random points
        file << "# points\n";
        std::vector<Point> points;
        for (size_t i = 0; i < 550; ++i)
        {
                points.emplace_back(make_random_point());
                file << points[i].x << "," << points[i].y << '\n';
        }

        // Define function to grab object's coordinates
        auto get_coord = [](const Point& p, int axis)
        {
                if (axis == 0) return p.x;
                return p.y;
        };

        // Create Kdtree
        kdtree::Kdtree<Point, decltype(get_coord)> tree(points, get_coord);

        // Generate point whose neighbors we want to find
        file << "# gold\n";
        Point gold = make_random_point();
        file << gold.x << "," << gold.y << '\n';

        // Find the nearest neighbor
        file << "# nn\n";
        const Point& nn = tree.find_nn(gold);
        file << nn.x << "," << nn.y << '\n';

        // Find all the neighbors within 'radius' distance
        float radius = 2.0;
        file << "# neighbors\n";
        file << radius << '\n';
        std::vector<size_t> neighbors = tree.find_neighbors(gold, radius);
        for (auto&& ind : neighbors)
        {
                auto& neigh = points[ind];
                file << neigh.x << "," << neigh.y << '\n';
        }

        return 0;
}
