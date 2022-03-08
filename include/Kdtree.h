#pragma once

#include <algorithm>
#include <cassert>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

namespace kdtree
{
// TODO: Make this a template parameter
static constexpr int DIMENSION = 2;

template <typename T, typename GetCoord, typename Float = float,
          typename Compare = std::less<Float>>
class Kdtree
{
        static_assert(std::is_arithmetic_v<Float>);
        static_assert(
            std::is_convertible_v<std::invoke_result_t<GetCoord, const T&, int>,
                                  Float>,
            "GetCoord must be a callable whose signature is Float(const "
            "T&, int)");
        static_assert(
            std::is_convertible_v<
                std::invoke_result_t<Compare, const Float&, const Float&>,
                bool>,
            "GetMedian must be a callable whose signature is bool(const T&, "
            "const "
            "T&))");

public:
        Kdtree(const std::vector<T>& values,
               const GetCoord& get_coord = GetCoord(),
               const Compare& comp = Compare())
            : values_(values), comp_(comp), get_coord_(get_coord)
        {
                std::vector<size_t> indices(values.size());
                std::iota(std::begin(indices), std::end(indices), 0);
                root_ = make_kd_tree_from_vector(indices, 0, values.size());
        }

        const T& find_nn(const T& element)
        {
                size_t best = std::numeric_limits<size_t>::max();
                Float best_dist = std::numeric_limits<Float>::max();

                find_nn_aux(root_.get(), element, 0, best, best_dist);
                return values_[best];
        }

private:
        struct Node
        {
                size_t index;
                std::unique_ptr<Node> left = nullptr;
                std::unique_ptr<Node> right = nullptr;
        };

        const std::vector<T>& values_;
        Compare comp_;
        GetCoord get_coord_;
        std::unique_ptr<Node> root_ = nullptr;

        std::unique_ptr<Node> make_kd_tree_from_vector(
            std::vector<size_t>& indices, size_t i, size_t j, int depth = 0)
        {
                if (i == j) return nullptr;

                int axis = depth % DIMENSION;
                size_t median_index = get_median(indices, i, j, axis);

                std::unique_ptr<Node> new_node = std::make_unique<Node>(
                    Node{median_index, nullptr, nullptr});

                size_t median = (j + i) / 2;
                new_node->left =
                    make_kd_tree_from_vector(indices, i, median, depth + 1);
                new_node->right =
                    make_kd_tree_from_vector(indices, median + 1, j, depth + 1);
                return new_node;
        }

        size_t get_median(std::vector<size_t>& indices, size_t i, size_t j,
                          int axis)
        {
                auto m = indices.begin() + (j + i) / 2;
                std::nth_element(
                    indices.begin() + static_cast<std::ptrdiff_t>(i), m,
                    indices.begin() + static_cast<std::ptrdiff_t>(j),
                    [this, axis](size_t ind_a, size_t ind_b) {
                            const T& a = values_[ind_a];
                            const T& b = values_[ind_b];
                            return comp_(get_coord_(a, axis),
                                         get_coord_(b, axis));
                    });
                return indices[(j + i) / 2];
        }

        void find_nn_aux(const Node* node, const T& element, int depth,
                         size_t& best, Float& best_dist)
        {
                if (node == nullptr) return;

                const T& value = values_[node->index];

                Float current_dist = squared_distance(element, value);

                if (current_dist < best_dist)
                {
                        best_dist = current_dist;
                        best = node->index;
                }

                int axis = depth % DIMENSION;

                // This if else could be avoided with the use of abs
                if (get_coord_(element, axis) < get_coord_(value, axis))
                {
                        find_nn_aux(node->left.get(), element, depth + 1, best,
                                    best_dist);
                        if (get_coord_(value, axis) <
                            best_dist + get_coord_(element, axis))
                        {
                                find_nn_aux(node->right.get(), element,
                                            depth + 1, best, best_dist);
                        }
                }
                else
                {
                        find_nn_aux(node->right.get(), element, depth + 1, best,
                                    best_dist);
                        if (get_coord_(element, axis) <
                            best_dist + get_coord_(value, axis))
                        {
                                find_nn_aux(node->left.get(), element,
                                            depth + 1, best, best_dist);
                        }
                }
        }

        Float squared_distance(const T& a, const T& b)
        {
                Float distance = 0;
                for (int d = 0; d < DIMENSION; ++d)
                {
                        distance += (get_coord_(a, d) - get_coord_(b, d)) *
                                    (get_coord_(a, d) - get_coord_(b, d));
                }
                return distance;
        }
};

}  // namespace kdtree
