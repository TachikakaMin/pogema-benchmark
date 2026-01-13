#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <queue>
#include <random>
#include <cstring>
#include <unordered_map>
#include <mutex>

namespace py = pybind11;

constexpr int NOT_FOUND_PATH = 2048;

class LazyDistanceMap {
public:
    int height, width;
    std::vector<float> map_data;  // Flattened map
    std::unordered_map<int, std::vector<int>> cache;  // goal_key -> distance array
    std::mutex cache_mutex;

    LazyDistanceMap() : height(0), width(0) {}

    void init(py::array_t<float> map_array) {
        auto map = map_array.unchecked<2>();
        height = map.shape(0);
        width = map.shape(1);

        map_data.resize(height * width);
        for (int x = 0; x < height; ++x) {
            for (int y = 0; y < width; ++y) {
                map_data[x * width + y] = map(x, y);
            }
        }
        cache.clear();
    }

    // BFS from goal to compute distances
    std::vector<int> bfs_from_goal(int gx, int gy) {
        std::vector<int> dist(height * width, NOT_FOUND_PATH);

        if (gx < 0 || gx >= height || gy < 0 || gy >= width) {
            return dist;
        }
        if (map_data[gx * width + gy] != 0) {
            return dist;
        }

        dist[gx * width + gy] = 0;
        std::queue<std::pair<int, int>> q;
        q.push({gx, gy});

        const int dx[] = {-1, 1, 0, 0};
        const int dy[] = {0, 0, -1, 1};

        while (!q.empty()) {
            auto [cx, cy] = q.front();
            q.pop();
            int curr_dist = dist[cx * width + cy];

            for (int d = 0; d < 4; ++d) {
                int nx = cx + dx[d];
                int ny = cy + dy[d];

                if (nx >= 0 && nx < height && ny >= 0 && ny < width) {
                    int nidx = nx * width + ny;
                    if (map_data[nidx] == 0 && dist[nidx] == NOT_FOUND_PATH) {
                        dist[nidx] = curr_dist + 1;
                        q.push({nx, ny});
                    }
                }
            }
        }

        return dist;
    }

    int get_distance(int ax, int ay, int gx, int gy) {
        if (ax < 0 || ax >= height || ay < 0 || ay >= width) {
            return NOT_FOUND_PATH;
        }

        int goal_key = gx * width + gy;

        // Check cache
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            auto it = cache.find(goal_key);
            if (it != cache.end()) {
                return it->second[ax * width + ay];
            }
        }

        // Compute and cache
        auto dist = bfs_from_goal(gx, gy);
        int result = dist[ax * width + ay];

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            cache[goal_key] = std::move(dist);
        }

        return result;
    }

    // Get distance array for a goal (for batch operations)
    const std::vector<int>& get_distance_array(int gx, int gy) {
        int goal_key = gx * width + gy;

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            auto it = cache.find(goal_key);
            if (it != cache.end()) {
                return it->second;
            }
        }

        auto dist = bfs_from_goal(gx, gy);

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            cache[goal_key] = std::move(dist);
            return cache[goal_key];
        }
    }
};

// Construct input features
py::array_t<float> construct_features(
    py::array_t<float> map_data,
    py::array_t<int64_t> agent_locations,
    py::array_t<int64_t> goal_locations,
    LazyDistanceMap& distance_map,
    int feature_dim,
    const std::string& feature_type,
    unsigned int seed = 0
) {
    auto map = map_data.unchecked<2>();
    auto agents = agent_locations.unchecked<2>();
    auto goals = goal_locations.unchecked<2>();

    int height = map.shape(0);
    int width = map.shape(1);
    int agent_num = agents.shape(0);

    // Create output array
    py::array_t<float> output({feature_dim, height, width});
    auto out = output.mutable_unchecked<3>();

    // Initialize to zero
    std::memset(output.mutable_data(), 0, feature_dim * height * width * sizeof(float));

    // Channel 0: Map obstacles
    for (int x = 0; x < height; ++x) {
        for (int y = 0; y < width; ++y) {
            out(0, x, y) = map(x, y);
        }
    }

    // Channel 1: Agent positions (marked with agent_id starting from 1)
    for (int i = 0; i < agent_num; ++i) {
        int x = agents(i, 0);
        int y = agents(i, 1);
        if (x >= 0 && x < height && y >= 0 && y < width) {
            out(1, x, y) = i + 1;
        }
    }

    // Channel 2: Goal positions
    for (int i = 0; i < agent_num; ++i) {
        int x = goals(i, 0);
        int y = goals(i, 1);
        if (x >= 0 && x < height && y >= 0 && y < width) {
            out(2, x, y) = i + 1;
        }
    }

    if (feature_dim >= 4) {
        // Channel 3: Distance to goal
        for (int i = 0; i < agent_num; ++i) {
            int ax = agents(i, 0);
            int ay = agents(i, 1);
            int gx = goals(i, 0);
            int gy = goals(i, 1);

            if (ax >= 0 && ax < height && ay >= 0 && ay < width) {
                out(3, ax, ay) = distance_map.get_distance(ax, ay, gx, gy);
            }
        }
    }

    if (feature_dim >= 6 && feature_type == "gradient") {
        // Channels 4 & 5: Gradient directions
        std::mt19937 gen(seed);
        std::uniform_int_distribution<> choice3(-1, 1);
        std::uniform_int_distribution<> choice2(0, 1);

        for (int i = 0; i < agent_num; ++i) {
            int ax = agents(i, 0);
            int ay = agents(i, 1);
            int gx = goals(i, 0);
            int gy = goals(i, 1);

            if (ax < 0 || ax >= height || ay < 0 || ay >= width) continue;

            // Get distance array for this goal (cached)
            const auto& dist_array = distance_map.get_distance_array(gx, gy);

            int current_dist = dist_array[ax * width + ay];

            // Check 4 directions using cached array
            int left_dist = (ax > 0) ? dist_array[(ax - 1) * width + ay] : NOT_FOUND_PATH;
            int right_dist = (ax < height - 1) ? dist_array[(ax + 1) * width + ay] : NOT_FOUND_PATH;
            int up_dist = (ay < width - 1) ? dist_array[ax * width + (ay + 1)] : NOT_FOUND_PATH;
            int down_dist = (ay > 0) ? dist_array[ax * width + (ay - 1)] : NOT_FOUND_PATH;

            // Check if other agents are blocking
            for (int j = 0; j < agent_num; ++j) {
                if (i == j) continue;
                int ox = agents(j, 0);
                int oy = agents(j, 1);

                if (ox == ax - 1 && oy == ay) left_dist = NOT_FOUND_PATH;
                if (ox == ax + 1 && oy == ay) right_dist = NOT_FOUND_PATH;
                if (ox == ax && oy == ay + 1) up_dist = NOT_FOUND_PATH;
                if (ox == ax && oy == ay - 1) down_dist = NOT_FOUND_PATH;
            }

            int delta_left = left_dist - current_dist;
            int delta_right = right_dist - current_dist;
            int delta_up = up_dist - current_dist;
            int delta_down = down_dist - current_dist;

            // Compute gradient x (left/right)
            float dx = 0;
            if (delta_left > 0 && delta_right > 0) {
                dx = 0;
            } else if (delta_left >= 0 && delta_right < 0) {
                dx = 1;
            } else if (delta_left < 0 && delta_right >= 0) {
                dx = -1;
            } else if (delta_left < 0 && delta_right < 0) {
                dx = choice2(gen) == 0 ? -1 : 1;
            } else if (delta_left == 0 && delta_right == 0) {
                dx = choice3(gen);
            } else if (delta_left == 0 && delta_right > 0) {
                dx = choice2(gen) == 0 ? 0 : -1;
            } else if (delta_left > 0 && delta_right == 0) {
                dx = choice2(gen) == 0 ? 0 : 1;
            } else {
                dx = choice2(gen) == 0 ? -1 : 1;
            }

            // Compute gradient y (down/up)
            float dy = 0;
            if (delta_down > 0 && delta_up > 0) {
                dy = 0;
            } else if (delta_down >= 0 && delta_up < 0) {
                dy = 1;
            } else if (delta_down < 0 && delta_up >= 0) {
                dy = -1;
            } else if (delta_down < 0 && delta_up < 0) {
                dy = choice2(gen) == 0 ? -1 : 1;
            } else if (delta_down == 0 && delta_up == 0) {
                dy = choice3(gen);
            } else if (delta_down == 0 && delta_up > 0) {
                dy = choice2(gen) == 0 ? -1 : 0;
            } else if (delta_down > 0 && delta_up == 0) {
                dy = choice2(gen) == 0 ? 0 : 1;
            } else {
                dy = choice2(gen) == 0 ? -1 : 1;
            }

            out(4, ax, ay) = dx;
            out(5, ax, ay) = dy;
        }
    }

    return output;
}

PYBIND11_MODULE(mapf_features_cpp, m) {
    m.doc() = "C++ acceleration for Foundation MAPF feature construction";

    py::class_<LazyDistanceMap>(m, "DistanceMap")
        .def(py::init<>())
        .def("compute", &LazyDistanceMap::init, "Initialize with map data (lazy computation)")
        .def("get_distance", &LazyDistanceMap::get_distance, "Get distance between two points")
        .def_readonly("height", &LazyDistanceMap::height)
        .def_readonly("width", &LazyDistanceMap::width);

    m.def("construct_features", &construct_features,
          "Construct input features for Foundation MAPF",
          py::arg("map_data"),
          py::arg("agent_locations"),
          py::arg("goal_locations"),
          py::arg("distance_map"),
          py::arg("feature_dim"),
          py::arg("feature_type"),
          py::arg("seed") = 0);

    m.attr("NOT_FOUND_PATH") = NOT_FOUND_PATH;
}
