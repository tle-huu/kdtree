#pragma once

#include <algorithm>
#include <cassert>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

namespace kdtree
{
template <typename T, int DIMENSION, typename GetCoord, typename Float = float>
class Kdtree
{
    static_assert(std::is_arithmetic_v<Float>);
    static_assert(std::is_convertible_v<
                      std::invoke_result_t<GetCoord, const T&, int>, Float>,
                  "GetCoord must be a callable whose signature is Float(const "
                  "T&, int)");

public:
    Kdtree(const std::vector<T>& values, const GetCoord& get_coord = GetCoord())
        : values_(values), get_coord_(get_coord)
    {
        std::vector<size_t> indices(values.size());
        std::iota(std::begin(indices), std::end(indices), 0);
        root_ = make_kd_tree_from_vector(indices, 0, values.size());
    }

    // Returns a const reference to the nearest neighbor
    const T& find_nn(const T& element) const noexcept
    {
        size_t best = std::numeric_limits<size_t>::max();
        Float best_dist = std::numeric_limits<Float>::max();

        find_nn_aux(root_.get(), element, /* depth */ 0, best, best_dist);
        return values_[best];
    }

    // Returns the indices of the k nearest neighbors
    std::vector<size_t> find_knn(const T& element, int k) const noexcept
    {
        assert(k > 0);

        KnnMaxHeap neighbors(k);
        Float best_dist = std::numeric_limits<Float>::max();

        find_knn_aux(root_.get(), element, /* depth */ 0, best_dist, neighbors);

        std::vector<size_t> ret;
        ret.reserve(neighbors.size());
        for (auto& p : neighbors)
        {
            ret.push_back(p.second);
        }
        return ret;
    }

    // Returns the indices of the nearest neighbors within 'radius' distance
    std::vector<size_t> find_neighbors(const T& element,
                                       Float radius) const noexcept
    {
        std::vector<size_t> neighbors;
        find_neighbors_aux(root_.get(), element, radius * radius,
                           /* depth */ 0, neighbors);
        return neighbors;
    }

private:
    struct Node
    {
        size_t index;
        std::unique_ptr<Node> left = nullptr;
        std::unique_ptr<Node> right = nullptr;
    };

    // Bounded max heap
    template <typename U>
    class MaxHeap
    {
    public:
        using iterator = typename std::vector<U>::iterator;
        using const_iterator = typename std::vector<U>::const_iterator;

        // -------------------------------------------------------------
        // MaxHeap Constructor
        // -------------------------------------------------------------
        MaxHeap() = delete;
        MaxHeap(size_t capacity) : capacity_(capacity)
        {
            assert(capacity > 0);
            values_.reserve(capacity);
        }

        // ------------------------------------------------------------
        // MaxHeap Member Mutators
        // ------------------------------------------------------------
        void push(const U& value) noexcept
        {
            if (values_.size() == capacity_ && value < values_[0])
            {
                values_[0] = value;
                bubble_down(0);
            }
            else if (values_.size() < capacity_)
            {
                values_.push_back(value);
                std::push_heap(values_.begin(), values_.end());
            }
        }

        // ------------------------------------------------------------
        // MaxHeap Member Accessors
        // ------------------------------------------------------------
        const U& head() const noexcept { return values_[0]; }

        size_t size() noexcept { return values_.size(); }

        size_t capacity() noexcept { return values_.capacity(); }

        // Returns an `iterator` to the beginning of the inlined vector.
        iterator begin() noexcept { return values_.begin(); }

        // Returns a `const_iterator` to the beginning of the inlined
        // vector.
        const iterator begin() const noexcept { return values_.begin(); }

        // Returns an `iterator` to the end of the inlined vector.
        iterator end() noexcept { return values_.end(); }

        // Returns a `const_iterator` to the end of the inlined vector.
        const iterator end() const noexcept { return values_.end(); }

    private:
        size_t left(size_t k) const noexcept { return 2 * k + 1; }

        size_t right(size_t k) const noexcept { return 2 * k + 2; }

        void bubble_down(size_t index) noexcept
        {
            while (left(index) < values_.size())
            {
                size_t largest = index;
                if (values_[largest] < values_[left(index)])
                    largest = left(index);
                if (right(index) < values_.size() &&
                    values_[largest] < values_[right(index)])
                    largest = right(index);
                if (largest != index)
                {
                    std::swap(values_[index], values_[largest]);
                    index = largest;
                    continue;
                }
                break;
            }
        }

        size_t capacity_;
        std::vector<U> values_;
    };
    using KnnMaxHeap = MaxHeap<std::pair<Float, size_t>>;

    std::unique_ptr<Node> make_kd_tree_from_vector(std::vector<size_t>& indices,
                                                   size_t i, size_t j,
                                                   int depth = 0)
    {
        if (i == j) return nullptr;

        int axis = depth % DIMENSION;
        size_t median_index = get_median(indices, i, j, axis);

        std::unique_ptr<Node> new_node =
            std::make_unique<Node>(Node{median_index, nullptr, nullptr});

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
        std::nth_element(indices.begin() + static_cast<std::ptrdiff_t>(i), m,
                         indices.begin() + static_cast<std::ptrdiff_t>(j),
                         [this, axis](size_t ind_a, size_t ind_b)
                         {
                             const T& a = values_[ind_a];
                             const T& b = values_[ind_b];
                             return get_coord_(a, axis) < get_coord_(b, axis);
                         });
        return indices[(j + i) / 2];
    }

    void find_nn_aux(const Node* node, const T& element, int depth,
                     size_t& best, Float& best_dist) const
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
        auto [next, other_child] =
            get_coord_(element, axis) < get_coord_(value, axis)
                ? std::pair<Node*, Node*>{node->left.get(), node->right.get()}
                : std::pair<Node*, Node*>{node->right.get(), node->left.get()};
        find_nn_aux(next, element, depth + 1, best, best_dist);
        if (squared(std::abs(get_coord_(element, axis) -
                             get_coord_(value, axis))) < best_dist)
        {
            find_nn_aux(other_child, element, depth + 1, best, best_dist);
        }
    }

    void find_knn_aux(const Node* node, const T& element, int depth,
                      Float& best_dist, KnnMaxHeap& neighbors) const noexcept
    {
        if (node == nullptr) return;

        const T& value = values_[node->index];
        Float current_dist = squared_distance(element, value);
        neighbors.push({current_dist, node->index});
        if (current_dist < best_dist)
        {
            best_dist = current_dist;
        }

        int axis = depth % DIMENSION;
        auto [next, other_child] =
            get_coord_(element, axis) < get_coord_(value, axis)
                ? std::pair<Node*, Node*>{node->left.get(), node->right.get()}
                : std::pair<Node*, Node*>{node->right.get(), node->left.get()};
        find_knn_aux(next, element, depth + 1, best_dist, neighbors);
        if (neighbors.size() < neighbors.capacity() ||
            squared(std::abs(get_coord_(element, axis) -
                             get_coord_(value, axis))) < neighbors.head().first)
        {
            find_knn_aux(other_child, element, depth + 1, best_dist, neighbors);
        }
    }

    void find_neighbors_aux(const Node* node, const T& element,
                            Float squared_radius, int depth,
                            std::vector<size_t>& neighbors) const noexcept
    {
        if (node == nullptr) return;

        const T& value = values_[node->index];
        Float current_dist = squared_distance(element, value);
        if (current_dist < squared_radius)
        {
            neighbors.emplace_back(node->index);
        }

        int axis = depth % DIMENSION;
        auto [next, other_child] =
            get_coord_(element, axis) < get_coord_(value, axis)
                ? std::pair<Node*, Node*>{node->left.get(), node->right.get()}
                : std::pair<Node*, Node*>{node->right.get(), node->left.get()};
        find_neighbors_aux(next, element, squared_radius, depth + 1, neighbors);

        if (squared(std::abs(get_coord_(element, axis) -
                             get_coord_(value, axis))) < squared_radius)
        {
            find_neighbors_aux(other_child, element, squared_radius, depth + 1,
                               neighbors);
        }
    }

    Float squared_distance(const T& a, const T& b) const noexcept
    {
        Float distance = 0;
        for (int d = 0; d < DIMENSION; ++d)
        {
            distance += (get_coord_(a, d) - get_coord_(b, d)) *
                        (get_coord_(a, d) - get_coord_(b, d));
        }
        return distance;
    }

    Float squared(Float number) const noexcept { return number * number; }

    const std::vector<T>& values_;
    GetCoord get_coord_;
    std::unique_ptr<Node> root_ = nullptr;
};

}  // namespace kdtree
