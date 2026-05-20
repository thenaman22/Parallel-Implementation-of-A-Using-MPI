# MPI A* Implementation Guide

This document explains what the code in `A_Star.cpp` does, how the sequential and parallel versions work, and why the MPI-based distributed version can outperform the sequential baseline on the right workloads.

## 1. What the Program Does

The program solves a shortest-path problem on a 2D grid:

- `1` means the cell is traversable
- `0` means the cell is blocked
- movement is allowed in four directions only:
  - up
  - down
  - left
  - right

Each cell is represented as a single integer node ID:

```text
node = row * dim + col
```

This lets the search logic work with a compact 1D representation while still converting back to `(row, col)` for output.

The program runs two versions of A*:

1. A sequential baseline on rank 0.
2. A distributed MPI version where each node is owned by one process according to a deterministic hash.

At the end, it compares the two results and reports whether the parallel implementation found the same optimal path cost.

## 2. Main Data Structures

### `Grid`

`Grid` stores:

- `dim`: grid width and height
- `cells`: flattened grid values

It also provides helper methods for:

- bounds checking
- testing whether a node is traversable
- converting between node IDs and coordinates
- generating neighbors

This replaces the older raw-pointer grid representation with a safer RAII-based structure using `std::vector<int>`.

### `QueueItem`

`QueueItem` represents an A* frontier entry:

- `f`: estimated total cost `f(n) = g(n) + h(n)`
- `g`: known path cost from start to this node
- `node`: node ID

The priority queue is ordered by the smallest `f`, then by `g`, then by `node`.

### `NodeUpdate`

`NodeUpdate` is the MPI message sent between ranks when one process discovers a better path to a node owned by another process.

It contains:

- `node`: the destination node being improved
- `g`: the new tentative path cost
- `parent`: the predecessor node on that path

### `PendingNodeSend`

`PendingNodeSend` stores:

- the message buffer
- the `MPI_Request`

This is important because the code uses `MPI_Isend`, and the message buffer must remain valid until the send completes.

### `ParallelResult`

`ParallelResult` packages the result of the distributed search:

- final path
- goal cost
- whether a path was found
- whether the reconstructed path was validated

## 3. Important Helper Functions

### `heuristic(node, goal, grid)`

This computes Manhattan distance:

```text
h(n) = abs(goal_row - node_row) + abs(goal_col - node_col)
```

This is admissible for four-direction movement on a grid with unit edge cost, so A* remains optimal.

### `owner(node, nproc)`

This assigns each node to exactly one MPI rank:

```text
owner(node) = floor(P * fractional_part(node * A))
A = (sqrt(5) - 1) / 2
```

This is the core of the distributed design. Instead of splitting the grid into rectangular regions, ownership is determined by hashing the node ID. That means:

- each node has exactly one authoritative owner
- each rank stores search state only for the nodes it owns
- work can be balanced more evenly than a naive spatial split in some maps

### `relax_node(...)`

This is the local A* update rule:

- if a newly discovered `g` is better than the current one, update it
- record the parent
- push a new frontier entry into the priority queue

The implementation allows duplicate entries in the priority queue. This is normal because `std::priority_queue` does not support decrease-key.

### `discard_stale_frontier(...)`

Because duplicate queue entries are allowed, some entries become stale after a better `g` cost is discovered later. This helper removes stale entries from the top of the heap before expansion.

This makes the implementation simpler and safer than trying to maintain a separate open-list set with strict queue/set synchronization.

## 4. How the Sequential Version Works

The sequential solver is `seq_aStar(...)`.

### Sequential flow

1. Insert the start node with `g = 0` and `f = h(start, goal)`.
2. Repeatedly pop the best frontier node.
3. If it is stale, skip it.
4. If it is the goal, reconstruct the path and stop.
5. Otherwise, relax each valid neighbor with `tentative_g = current.g + 1`.

### Why it is useful

The sequential version serves two purposes:

- it is the simplest correct reference implementation
- it provides the baseline used to verify the MPI result

The code uses the sequential result on rank 0 to compare:

- path existence
- optimal path cost
- path reconstruction correctness

## 5. How the Parallel MPI Version Works

The distributed solver is `parallel_aStar(...)`.

### Basic idea

Each process owns only the nodes assigned to it by `owner(node, nproc)`.

That process is responsible for:

- storing the best known `g` for those nodes
- storing the parent pointer for those nodes
- pushing those nodes into its local priority queue
- deciding whether an incoming improvement should be accepted

### Parallel flow

1. Only the owner of the start node inserts the start state into its local queue.
2. Each rank repeatedly:
   - cleans up completed non-blocking sends
   - drains all incoming `NodeUpdate` messages
   - discards stale frontier entries
   - expands its local best node if that node is still promising
3. For every neighbor:
   - if the current rank owns it, relax locally
   - otherwise send a `NodeUpdate` to the owning rank
4. If a rank reaches the goal, it updates its local best goal cost.
5. `MPI_Allreduce(..., MPI_MIN, ...)` shares the best known goal cost globally.
6. Any rank can prune work whose best frontier `f` is already not better than the best known complete path.
7. The search terminates only when:
   - all ranks are locally inactive
   - all sent node updates have been received
8. After termination, parent maps are gathered to rank 0 and the final path is reconstructed there.

## 6. MPI Communication Strategy

The current implementation uses one explicit message type during search:

- `TAG_NODE_UPDATE`

Each message carries:

- which node improved
- the new `g` cost
- the parent that produced that cost

### Why this design is safer than the earlier version

The earlier implementation had several hazards:

- fixed-size global request arrays
- request reuse across unrelated sends and receives
- stack buffers reused before `MPI_Isend` completed
- parent data racing search progress

The current version avoids those problems by:

- using a real `NodeUpdate` struct
- creating an MPI datatype for it
- keeping each outgoing send buffer alive until `MPI_Test` or `MPI_Wait` confirms completion
- tracking pending sends independently in a dynamic container

This makes the communication easier to reason about and far less likely to deadlock or corrupt messages.

## 7. How Termination Works

Termination is one of the hardest parts of distributed search.

It is not enough to say "my queue is empty" because:

- another rank may still send a useful node update later
- a non-blocking send may still be in flight

The code therefore uses a two-part termination check:

1. Local activity:
   - a rank is active if its best frontier node still has `f < global_best_goal`
2. Global message accounting:
   - each rank tracks how many node updates it has sent
   - each rank tracks how many node updates it has received
   - a global sum is computed with `MPI_Allreduce`

Termination happens only when:

- no rank is active
- total sends == total receives

This is a practical and much safer termination condition for this project.

## 8. How Final Path Reconstruction Works

During the search, each owner rank stores parent pointers only for the nodes it owns.

After termination:

1. each rank flattens its local parent map into integer pairs
2. all pairs are gathered to rank 0
3. rank 0 merges them into one parent map
4. rank 0 reconstructs the final path from goal back to start
5. the reconstructed path is validated by checking that:
   - the path is complete
   - the path cost matches the best goal cost

This is simpler and more reliable than trying to reconstruct the path while parent messages are still racing between processes.

## 9. Why the Parallel Version Can Be Better Than the Sequential Version

The parallel version is better in the following ways.

### 1. It can use multiple processes at the same time

The sequential version expands nodes on one process only.

The MPI version lets multiple ranks explore different owned frontier nodes concurrently. On large grids or harder mazes, this can reduce wall-clock time because the search effort is distributed.

### 2. It can prune globally once any rank finds a good goal path

As soon as one rank finds the goal, the best known goal cost is shared with all processes using `MPI_Allreduce`.

That means other ranks can stop expanding nodes whose `f` value is already worse than the best complete path found so far.

This global pruning is one of the main benefits of distributed A*.

### 3. It is more scalable for larger search spaces

The sequential version is limited to one process's execution.

The MPI version is designed so that:

- each rank keeps local frontier state
- node ownership is deterministic
- work is communicated only when needed

As the problem size grows, this structure gives the program a path toward better performance than a single-process run.

### 4. It preserves the A* logic instead of switching to a weaker approximation

This implementation still behaves like A*:

- it uses `g + h`
- it routes node improvements to authoritative owners
- it keeps the best known cost per node
- it prunes with a best-goal bound

So the parallel version remains aligned with the algorithm described in the project documentation, rather than replacing it with a simpler but less faithful partition-and-stitch approach.

## 10. Important Caveat: Parallel Is Not Always Faster

The parallel version is not automatically faster on every input.

It has overhead that the sequential version does not have:

- MPI communication
- synchronization through collectives
- extra bookkeeping for sends and termination detection
- final path gathering

Because of that:

- very small grids may run faster sequentially
- larger grids or more difficult obstacle layouts are more likely to benefit from parallel execution

So the correct claim is:

> The MPI version is designed to outperform the sequential version on sufficiently large or difficult search problems, while still matching its optimal path cost.

That is a stronger and more honest statement than saying it is always faster.

## 11. Why the Current Code Is Better Than the Earlier Version

Compared with the earlier implementation, the current code is improved in several important ways:

### Correctness improvements

- the start node now goes to its real owner, not always rank 0
- goal cost uses true edge cost `g`, not path node count
- stale priority queue entries are handled correctly
- path reconstruction happens after the distributed search fully finishes

### Safety improvements

- no hard-coded request arrays
- no stack buffer reuse with active `MPI_Isend`
- no mixing unrelated MPI operations into the same request slots
- better input validation and safer memory management

### Readability improvements

- explicit data structures: `Grid`, `QueueItem`, `NodeUpdate`, `ParallelResult`
- helper functions isolate responsibilities
- comments explain ownership, stale entries, communication, and termination

## 12. How to Read the Code

If you are reading `A_Star.cpp`, a good order is:

1. `Grid`
2. `heuristic`
3. `owner`
4. `relax_node`
5. `discard_stale_frontier`
6. `seq_aStar`
7. `send_node_update_safely`
8. `drain_incoming_messages`
9. `parallel_aStar`
10. `main`

That order moves from local search logic to distributed MPI orchestration.

## 13. Build and Run

### Build

```bash
make
```

or directly:

```bash
mpic++ -I. -std=c++17 -O2 -Wall -Wextra -pedantic A_Star.cpp -o A_Star
```

### Run

```bash
mpirun -np 4 ./A_Star -f ./Input/Input_simple.txt 0 0 4 4
```

Another example:

```bash
mpirun -np 4 ./A_Star -f ./Input/Input_maze_100.txt 0 0 99 99
```

## 14. Final Summary

This project now has:

- a sequential A* baseline for correctness
- a distributed MPI A* implementation based on hash-owned nodes
- safe non-blocking node-update communication
- global best-goal pruning
- a practical global termination check
- centralized final path reconstruction and validation

The parallel version is better than the sequential version because it can distribute search effort across ranks and prune globally once a good goal path is found. At the same time, the code remains faithful to the original project goal: a true distributed A* search, not a simplified split-and-stitch approximation.
