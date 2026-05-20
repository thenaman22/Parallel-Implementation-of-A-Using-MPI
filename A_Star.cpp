#include "mpi.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <list>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr int INF = std::numeric_limits<int>::max() / 4;
constexpr int ROOT_RANK = 0;

enum Tags {
  TAG_NODE_UPDATE = 100
};

// Grid cells are stored in a flat vector using row * dim + col indexing.
struct Grid {
  int dim = 0;
  std::vector<int> cells;

  bool in_bounds(int row, int col) const {
    return row >= 0 && row < dim && col >= 0 && col < dim;
  }

  bool valid_node(int node) const {
    return node >= 0 && node < static_cast<int>(cells.size());
  }

  bool passable(int node) const {
    return valid_node(node) && cells[node] == 1;
  }

  int node(int row, int col) const {
    return row * dim + col;
  }

  std::pair<int, int> coord(int node_id) const {
    return {node_id / dim, node_id % dim};
  }

  std::vector<int> neighbors(int node_id) const {
    std::vector<int> result;
    result.reserve(4);

    const auto [row, col] = coord(node_id);

    if (in_bounds(row - 1, col)) {
      const int up = node(row - 1, col);
      if (passable(up)) {
        result.push_back(up);
      }
    }
    if (in_bounds(row + 1, col)) {
      const int down = node(row + 1, col);
      if (passable(down)) {
        result.push_back(down);
      }
    }
    if (in_bounds(row, col - 1)) {
      const int left = node(row, col - 1);
      if (passable(left)) {
        result.push_back(left);
      }
    }
    if (in_bounds(row, col + 1)) {
      const int right = node(row, col + 1);
      if (passable(right)) {
        result.push_back(right);
      }
    }

    return result;
  }
};

struct QueueItem {
  int f = INF;
  int g = INF;
  int node = -1;
};

struct QueueCompare {
  bool operator()(const QueueItem& lhs, const QueueItem& rhs) const {
    if (lhs.f != rhs.f) {
      return lhs.f > rhs.f;
    }
    if (lhs.g != rhs.g) {
      return lhs.g > rhs.g;
    }
    return lhs.node > rhs.node;
  }
};

struct NodeUpdate {
  int node = -1;
  int g = INF;
  int parent = -1;
};

struct PendingNodeSend {
  NodeUpdate msg;
  MPI_Request request = MPI_REQUEST_NULL;
};

struct ParallelResult {
  std::vector<int> path;
  int goal_cost = INF;
  bool found = false;
  bool path_valid = false;
};

using OpenQueue = std::priority_queue<QueueItem, std::vector<QueueItem>, QueueCompare>;

int to_node(int row, int col, int dim) {
  return row * dim + col;
}

std::pair<int, int> to_coord(int node, int dim) {
  return {node / dim, node % dim};
}

int heuristic(int node, int goal, const Grid& grid) {
  const auto [node_row, node_col] = to_coord(node, grid.dim);
  const auto [goal_row, goal_col] = to_coord(goal, grid.dim);
  return std::abs(goal_row - node_row) + std::abs(goal_col - node_col);
}

// Deterministically assigns each node to one MPI rank.
int owner(int node, int nproc) {
  constexpr double A = (std::sqrt(5.0) - 1.0) / 2.0;
  const double x = static_cast<double>(node) * A;
  const double fractional = x - std::floor(x);
  int result = static_cast<int>(std::floor(static_cast<double>(nproc) * fractional));
  if (result < 0) {
    result = 0;
  }
  if (result >= nproc) {
    result = nproc - 1;
  }
  return result;
}

bool read_grid(const std::string& filename,
               int start_row,
               int start_col,
               int goal_row,
               int goal_col,
               Grid& grid,
               std::string& error) {
  if (filename.empty()) {
    error = "missing input filename";
    return false;
  }

  std::ifstream input(filename);
  if (!input) {
    error = "could not open file: " + filename;
    return false;
  }

  int dim = 0;
  if (!(input >> dim) || dim <= 0) {
    error = "invalid grid dimension";
    return false;
  }

  grid.dim = dim;
  grid.cells.assign(static_cast<std::size_t>(dim) * static_cast<std::size_t>(dim), 0);

  for (int row = 0; row < dim; ++row) {
    for (int col = 0; col < dim; ++col) {
      int value = -1;
      if (!(input >> value)) {
        error = "grid ended early while reading row " + std::to_string(row);
        return false;
      }
      if (value != 0 && value != 1) {
        error = "grid values must be 0 or 1";
        return false;
      }
      grid.cells[static_cast<std::size_t>(to_node(row, col, dim))] = value;
    }
  }

  int extra_value = 0;
  if (input >> extra_value) {
    error = "grid contains extra values after the expected " +
            std::to_string(dim * dim) + " cells";
    return false;
  }

  if (!grid.in_bounds(start_row, start_col) || !grid.in_bounds(goal_row, goal_col)) {
    error = "start or goal is out of bounds";
    return false;
  }

  const int start = grid.node(start_row, start_col);
  const int goal = grid.node(goal_row, goal_col);
  if (!grid.passable(start)) {
    error = "start node is blocked";
    return false;
  }
  if (!grid.passable(goal)) {
    error = "goal node is blocked";
    return false;
  }

  return true;
}

bool relax_node(int node,
                int tentative_g,
                int parent_node,
                int goal,
                const Grid& grid,
                std::unordered_map<int, int>& g_cost,
                std::unordered_map<int, int>& parent,
                OpenQueue& open) {
  // We only keep strictly better g-costs. Lower-priority duplicates stay in the
  // heap and get discarded later when they become stale.
  const auto current = g_cost.find(node);
  if (current != g_cost.end() && tentative_g >= current->second) {
    return false;
  }

  g_cost[node] = tentative_g;
  if (parent_node >= 0) {
    parent[node] = parent_node;
  } else {
    parent.erase(node);
  }

  open.push({tentative_g + heuristic(node, goal, grid), tentative_g, node});
  return true;
}

void discard_stale_frontier(OpenQueue& open, const std::unordered_map<int, int>& g_cost) {
  while (!open.empty()) {
    const QueueItem item = open.top();
    const auto current = g_cost.find(item.node);
    // priority_queue has no decrease-key, so old entries are expected.
    if (current == g_cost.end() || current->second != item.g) {
      open.pop();
      continue;
    }
    break;
  }
}

void prune_completed_sends(std::list<PendingNodeSend>& pending_sends) {
  for (auto it = pending_sends.begin(); it != pending_sends.end();) {
    int completed = 0;
    MPI_Test(&it->request, &completed, MPI_STATUS_IGNORE);
    if (completed != 0) {
      it = pending_sends.erase(it);
    } else {
      ++it;
    }
  }
}

void wait_for_pending_sends(std::list<PendingNodeSend>& pending_sends) {
  for (auto& pending : pending_sends) {
    MPI_Wait(&pending.request, MPI_STATUS_IGNORE);
  }
  pending_sends.clear();
}

void send_node_update_safely(int recipient,
                             const NodeUpdate& msg,
                             MPI_Datatype node_update_type,
                             std::list<PendingNodeSend>& pending_sends,
                             long long& sent_updates) {
  // Keep the message storage alive until MPI reports the send request complete.
  pending_sends.push_back({msg, MPI_REQUEST_NULL});
  PendingNodeSend& pending = pending_sends.back();
  MPI_Isend(&pending.msg,
            1,
            node_update_type,
            recipient,
            TAG_NODE_UPDATE,
            MPI_COMM_WORLD,
            &pending.request);
  ++sent_updates;
}

void drain_incoming_messages(int rank,
                             int nproc,
                             int goal,
                             const Grid& grid,
                             MPI_Datatype node_update_type,
                             std::unordered_map<int, int>& g_cost,
                             std::unordered_map<int, int>& parent,
                             OpenQueue& open,
                             long long& received_updates) {
  while (true) {
    int available = 0;
    MPI_Status status{};
    MPI_Iprobe(MPI_ANY_SOURCE, TAG_NODE_UPDATE, MPI_COMM_WORLD, &available, &status);
    if (available == 0) {
      break;
    }

    NodeUpdate update;
    MPI_Recv(&update,
             1,
             node_update_type,
             status.MPI_SOURCE,
             status.MPI_TAG,
             MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    ++received_updates;

    // Every update should already be routed to the owning rank, but this check
    // keeps the local state honest if a bad message ever appears.
    if (owner(update.node, nproc) != rank) {
      continue;
    }

    relax_node(update.node, update.g, update.parent, goal, grid, g_cost, parent, open);
  }
}

std::vector<int> reconstruct_path(const std::unordered_map<int, int>& parent,
                                  int start,
                                  int goal) {
  if (start == goal) {
    return {start};
  }

  std::vector<int> reversed_path;
  reversed_path.push_back(goal);

  int current = goal;
  const std::size_t max_steps = parent.size() + 1;

  while (current != start) {
    const auto it = parent.find(current);
    if (it == parent.end()) {
      return {};
    }
    current = it->second;
    reversed_path.push_back(current);
    if (reversed_path.size() > max_steps + 1) {
      return {};
    }
  }

  std::reverse(reversed_path.begin(), reversed_path.end());
  return reversed_path;
}

int path_cost(const std::vector<int>& path) {
  if (path.empty()) {
    return INF;
  }
  return static_cast<int>(path.size()) - 1;
}

std::vector<int> seq_aStar(int start, int goal, const Grid& grid) {
  OpenQueue open;
  std::unordered_map<int, int> g_cost;
  std::unordered_map<int, int> parent;

  // The sequential run is the correctness baseline for the MPI version.
  relax_node(start, 0, -1, goal, grid, g_cost, parent, open);

  while (true) {
    discard_stale_frontier(open, g_cost);
    if (open.empty()) {
      break;
    }

    const QueueItem current = open.top();
    open.pop();

    if (current.node == goal) {
      return reconstruct_path(parent, start, goal);
    }

    for (const int neighbor : grid.neighbors(current.node)) {
      relax_node(neighbor, current.g + 1, current.node, goal, grid, g_cost, parent, open);
    }
  }

  return {};
}

std::vector<int> flatten_parent_map(const std::unordered_map<int, int>& parent) {
  std::vector<int> flat;
  flat.reserve(parent.size() * 2);
  for (const auto& entry : parent) {
    flat.push_back(entry.first);
    flat.push_back(entry.second);
  }
  return flat;
}

ParallelResult parallel_aStar(int start,
                              int goal,
                              const Grid& grid,
                              int nproc,
                              int rank,
                              MPI_Datatype node_update_type) {
  OpenQueue open;
  std::unordered_map<int, int> g_cost;
  std::unordered_map<int, int> parent;
  std::list<PendingNodeSend> pending_sends;

  long long sent_updates = 0;
  long long received_updates = 0;

  int local_best_goal = INF;
  int global_best_goal = INF;

  // Only the owner of the start node seeds the distributed frontier.
  if (rank == owner(start, nproc)) {
    relax_node(start, 0, -1, goal, grid, g_cost, parent, open);
  }

  while (true) {
    prune_completed_sends(pending_sends);
    drain_incoming_messages(rank,
                            nproc,
                            goal,
                            grid,
                            node_update_type,
                            g_cost,
                            parent,
                            open,
                            received_updates);
    discard_stale_frontier(open, g_cost);

    if (!open.empty() && open.top().f < global_best_goal) {
      const QueueItem current = open.top();
      open.pop();

      if (current.node == goal) {
        // Store the best complete path cost seen by this rank; the collective
        // min below makes it visible to the rest of the search.
        if (current.g < local_best_goal) {
          local_best_goal = current.g;
        }
      } else {
        for (const int neighbor : grid.neighbors(current.node)) {
          const int tentative_g = current.g + 1;
          if (tentative_g >= global_best_goal) {
            continue;
          }

          const int target_rank = owner(neighbor, nproc);
          if (target_rank == rank) {
            // Fast path: keep the update local when we already own the node.
            relax_node(neighbor,
                       tentative_g,
                       current.node,
                       goal,
                       grid,
                       g_cost,
                       parent,
                       open);
          } else {
            send_node_update_safely(target_rank,
                                    {neighbor, tentative_g, current.node},
                                    node_update_type,
                                    pending_sends,
                                    sent_updates);
          }
        }
      }
    }

    prune_completed_sends(pending_sends);

    int reduced_best_goal = INF;
    MPI_Allreduce(&local_best_goal,
                  &reduced_best_goal,
                  1,
                  MPI_INT,
                  MPI_MIN,
                  MPI_COMM_WORLD);
    global_best_goal = reduced_best_goal;

    discard_stale_frontier(open, g_cost);
    const int local_active = (!open.empty() && open.top().f < global_best_goal) ? 1 : 0;

    int any_active = 0;
    MPI_Allreduce(&local_active, &any_active, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    long long local_message_counts[2] = {sent_updates, received_updates};
    long long global_message_counts[2] = {0, 0};
    MPI_Allreduce(local_message_counts,
                  global_message_counts,
                  2,
                  MPI_LONG_LONG,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    // Terminate only after every process is idle and every node update has been received.
    if (any_active == 0 && global_message_counts[0] == global_message_counts[1]) {
      break;
    }
  }

  wait_for_pending_sends(pending_sends);

  ParallelResult result;
  result.goal_cost = global_best_goal;
  result.found = global_best_goal < INF;
  result.path_valid = !result.found;

  const std::vector<int> local_parent_data = flatten_parent_map(parent);
  const int local_parent_count = static_cast<int>(local_parent_data.size());

  std::vector<int> counts;
  std::vector<int> displacements;
  std::vector<int> gathered_parent_data;

  if (rank == ROOT_RANK) {
    counts.resize(nproc, 0);
  }

  MPI_Gather(&local_parent_count,
             1,
             MPI_INT,
             rank == ROOT_RANK ? counts.data() : nullptr,
             1,
             MPI_INT,
             ROOT_RANK,
             MPI_COMM_WORLD);

  if (rank == ROOT_RANK) {
    displacements.resize(nproc, 0);
    int total = 0;
    for (int i = 0; i < nproc; ++i) {
      displacements[i] = total;
      total += counts[i];
    }
    gathered_parent_data.resize(total);
  }

  MPI_Gatherv(local_parent_data.empty() ? nullptr : local_parent_data.data(),
              local_parent_count,
              MPI_INT,
              rank == ROOT_RANK ? gathered_parent_data.data() : nullptr,
              rank == ROOT_RANK ? counts.data() : nullptr,
              rank == ROOT_RANK ? displacements.data() : nullptr,
              MPI_INT,
              ROOT_RANK,
              MPI_COMM_WORLD);

  if (rank == ROOT_RANK && result.found) {
    // Path reconstruction is centralized after termination so we do not depend
    // on parent updates racing the search itself.
    std::unordered_map<int, int> merged_parent;
    merged_parent.reserve(gathered_parent_data.size() / 2 + 1);

    for (std::size_t i = 0; i + 1 < gathered_parent_data.size(); i += 2) {
      merged_parent[gathered_parent_data[i]] = gathered_parent_data[i + 1];
    }

    result.path = reconstruct_path(merged_parent, start, goal);
    result.path_valid = !result.path.empty() && path_cost(result.path) == result.goal_cost;
  }

  return result;
}

MPI_Datatype create_node_update_type() {
  NodeUpdate sample;
  MPI_Datatype node_update_type = MPI_DATATYPE_NULL;

  int block_lengths[3] = {1, 1, 1};
  MPI_Aint offsets[3];
  MPI_Aint base = 0;
  MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};

  MPI_Get_address(&sample, &base);
  MPI_Get_address(&sample.node, &offsets[0]);
  MPI_Get_address(&sample.g, &offsets[1]);
  MPI_Get_address(&sample.parent, &offsets[2]);

  offsets[0] -= base;
  offsets[1] -= base;
  offsets[2] -= base;

  MPI_Type_create_struct(3, block_lengths, offsets, types, &node_update_type);
  MPI_Type_commit(&node_update_type);
  return node_update_type;
}

void print_path(std::ostream& out, const std::vector<int>& path, const Grid& grid) {
  if (path.empty()) {
    out << "NO PATH\n";
    return;
  }

  for (std::size_t i = 0; i < path.size(); ++i) {
    const auto [row, col] = grid.coord(path[i]);
    out << "(" << row << "," << col << ")";
    if (i + 1 != path.size()) {
      out << " ";
    }
  }
  out << '\n';
}

}  // namespace

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int nproc = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  std::string filename;
  std::vector<int> coords;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-f") {
      if (i + 1 >= argc) {
        if (rank == ROOT_RANK) {
          std::cerr << "Error: '-f' requires a filename\n";
        }
        MPI_Finalize();
        return -1;
      }
      filename = argv[++i];
      continue;
    }

    try {
      coords.push_back(std::stoi(arg));
    } catch (const std::exception&) {
      if (rank == ROOT_RANK) {
        std::cerr << "Error: invalid integer argument '" << arg << "'\n";
      }
      MPI_Finalize();
      return -1;
    }
  }

  if (coords.size() != 4) {
    if (rank == ROOT_RANK) {
      std::cerr << "Usage: " << argv[0] << " -f <filename> <x1> <y1> <x2> <y2>\n";
    }
    MPI_Finalize();
    return -1;
  }

  const int start_row = coords[0];
  const int start_col = coords[1];
  const int goal_row = coords[2];
  const int goal_col = coords[3];

  Grid grid;
  std::string error;
  if (!read_grid(filename, start_row, start_col, goal_row, goal_col, grid, error)) {
    if (rank == ROOT_RANK) {
      std::cerr << "Input error: " << error << '\n';
    }
    MPI_Finalize();
    return -1;
  }

  const int start = grid.node(start_row, start_col);
  const int goal = grid.node(goal_row, goal_col);

  std::vector<int> serial_path;
  double serial_time = 0.0;

  if (rank == ROOT_RANK) {
    const double serial_start = MPI_Wtime();
    serial_path = seq_aStar(start, goal, grid);
    serial_time = MPI_Wtime() - serial_start;
  }

  MPI_Datatype node_update_type = create_node_update_type();

  // Barriers keep the parallel timing window aligned across ranks.
  MPI_Barrier(MPI_COMM_WORLD);
  const double parallel_start = MPI_Wtime();
  ParallelResult parallel_result = parallel_aStar(start, goal, grid, nproc, rank, node_update_type);
  MPI_Barrier(MPI_COMM_WORLD);
  const double local_parallel_time = MPI_Wtime() - parallel_start;

  double parallel_time = 0.0;
  MPI_Reduce(&local_parallel_time,
             &parallel_time,
             1,
             MPI_DOUBLE,
             MPI_MAX,
             ROOT_RANK,
             MPI_COMM_WORLD);

  MPI_Type_free(&node_update_type);

  if (rank == ROOT_RANK) {
    const bool serial_found = !serial_path.empty();
    const int serial_cost = path_cost(serial_path);
    const bool parallel_found = parallel_result.found;
    const int parallel_cost = parallel_result.goal_cost;

    const bool correctness_pass =
        serial_found == parallel_found &&
        (!serial_found || (serial_cost == parallel_cost && parallel_result.path_valid));

    std::cout << "Serial path cost: ";
    if (serial_found) {
      std::cout << serial_cost << '\n';
    } else {
      std::cout << "NO PATH\n";
    }
    std::cout << "Serial path nodes: " << serial_path.size() << '\n';
    std::cout << "Serial path: ";
    print_path(std::cout, serial_path, grid);
    std::cout << "Serial time: " << serial_time << " seconds\n\n";

    std::cout << "Parallel path cost: ";
    if (parallel_found) {
      std::cout << parallel_cost << '\n';
    } else {
      std::cout << "NO PATH\n";
    }
    std::cout << "Parallel path nodes: " << parallel_result.path.size() << '\n';
    std::cout << "Parallel path: ";
    print_path(std::cout, parallel_result.path, grid);
    std::cout << "Parallel time: " << parallel_time << " seconds\n";

    if (parallel_time > 0.0) {
      std::cout << "Speedup: " << (serial_time / parallel_time) << '\n';
    } else {
      std::cout << "Speedup: inf\n";
    }

    std::cout << "Correctness: " << (correctness_pass ? "PASS" : "FAIL") << '\n';
    if (parallel_found && !parallel_result.path_valid) {
      std::cout << "Parallel path reconstruction failed validation.\n";
    }
  }

  MPI_Finalize();
  return 0;
}
