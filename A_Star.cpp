#include "mpi.h"
#include <unistd.h>
#include <ctype.h>
#include <queue>
#include <unordered_map>
#include <climits>
#include <unordered_set>
#include <assert.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <vector>
#include <iostream>
#include <sstream>
#include <memory>
#include <cstddef>
#include <functional>

using namespace std;

#ifndef _grid_H
#define _grid_H

MPI_Request node_reqs[64];
MPI_Request cost_reqs[64];
MPI_Request parent_reqs[64];
MPI_Status node_stats[64];
MPI_Status cost_stats[64];
MPI_Status parent_stats[64];

typedef struct {
  int cost;   // the current path score f(n) 
  int node;   // the index corresponding to the node
} info_node;

struct compare_nodes {
    bool operator()(info_node const & x, info_node const & y) {
        return x.cost > y.cost;
    }
};

struct grid_t {
  int dim;     
  int *grid;   // Represented as a 1D array where [i][j] => [i*dim + j] 
  grid_t(const int d, int *g) : dim(d), grid(g) {};
};

#endif 

grid_t* readgrid(int x, int y, int a, int b, char *filename) {
  int dim; 

	FILE *input = fopen(filename, "r");
	if (!input) { 
		cout<<"Cant open file: "<<filename<<endl;
		return {};
	}

  // Read the dimension of the grid
  fscanf(input, "%d\n", &dim);

  if (x < 0 || x >= dim || y < 0 || y >= dim) {
    cout<<"out of bounds error"<<endl;
    return {};
  }
  if (a < 0 || a >= dim || b < 0 || b >= dim) {
    cout<<"out of bounds error"<<endl;
    return {};
  }

  int *grid = (int *)malloc(dim*dim*sizeof(int));


  int line_len = 2*dim+1;
  char* line = (char *)malloc(line_len);
  int line_cnt = 0;
  int node_cnt = 0;

  while (fgets(line, line_len, input))  {
    node_cnt = 0;
    for (int i = 0; i < line_len - 2; i++) {
      if (!isspace(line[i])) {
        grid[dim*line_cnt + node_cnt] = (int)(line[i] - '0');
        node_cnt++;
      }
    }
    line_cnt++;
  }
  fclose(input);
  free(line);
  return new grid_t(dim, grid);
}

grid_t* grid;


// _rqst status variables 


// heuristic function 
int heuristic(int start_point, int destination) {
  int start_point_R = start_point / grid->dim;
  int start_point_C = start_point % grid->dim;
  int destination_R = destination / grid->dim;
  int destination_C = destination % grid->dim;
  // manhatten distance 
  return abs(start_point_R - destination_R) + abs(start_point_C - destination_C);
}


vector<int> find_neigbours(int curr, grid_t* grid) {
  int currR = curr / grid->dim;
  int currC = curr % grid->dim;
  int lim = grid->dim - 1;
  vector<int> nbrs;
  if (currR != 0 && grid->grid[curr - grid->dim] == 1)    nbrs.push_back(curr - grid->dim); // UP
  if (currR < lim && grid->grid[curr + grid->dim] == 1)   nbrs.push_back(curr + grid->dim);   // DOWN
  if (currC != 0 && grid->grid[curr - 1] == 1)    nbrs.push_back(curr - 1);  // LEFT 
  if (currC < lim && grid->grid[curr + 1] == 1)   nbrs.push_back(curr + 1); // RIGHT
  return nbrs;
}



void make_path(unordered_map<int, int> parent, int curr, vector<int> *path) {
  path->clear();
  path->push_back(curr);
  while (parent.find(curr) != parent.end()) {
    curr = parent.at(curr);
    path->push_back(curr);
  }
}


vector<int> seq_aStar(int source, int target, grid_t* grid) {
  priority_queue<info_node, vector<info_node>, compare_nodes> pq;
  unordered_set<int> open_list;
  unordered_map<int, int> parent;
  unordered_map<int, int> cost_score;
  vector<int> path;

  pq.push({heuristic(source, target), source});
  open_list.insert(source);

  cost_score.insert({source, 0});

  for (int i = 0; i < grid->dim; i++) {
    for (int j = 0; j < grid->dim; j++) {
      if (i != source / grid->dim || j != source % grid->dim) {
        cost_score.insert({i*grid->dim + j, INT_MAX});
      }
    }
  }

  vector<int> neighbors;
  while (!pq.empty()) {
    int current = pq.top().node;
    if (current == target) {
      make_path(parent, current, &path);
      break;
    }

    pq.pop();
    open_list.erase(current);
    neighbors = find_neigbours(current, grid);
    for (int neighbor: neighbors) { 
      int neighborScore = cost_score.at(neighbor);
      int currentScore = cost_score.at(current) + 1;
      if (currentScore < neighborScore) {
        if (parent.find(current) != parent.end()) {
          if (parent.at(current) != neighbor) {
            parent.emplace(neighbor, current);
          }
        } else {
          parent.emplace(neighbor, current);
        }
        cost_score.erase(neighbor);
        cost_score.emplace(neighbor, currentScore);
        int neighborfScore = currentScore + heuristic(neighbor, target);
        if (open_list.find(neighbor) == open_list.end()) {
          open_list.emplace(neighbor);
          pq.push({neighborfScore, neighbor});
        }
      }
    }
    neighbors.clear();
    assert(pq.size() == open_list.size());
  }

  return path;
}

void parallel_aStar(int start_point, int destination, grid_t* grid, vector<int> *path, int nproc, int rank) {

  priority_queue<info_node, vector<info_node>, compare_nodes> pq;
  unordered_set<int> open_list;
  unordered_map<int, int> parent;
  unordered_map<int, int> g_cost;  
  unordered_set<int> closed_list;

  int pathCost = INT_MAX;
  int valid_path_flag = 0;

  double startTime = MPI_Wtime();
  double A = (sqrt(5) - 1) / 2; // for hashing

  int recv_proc = floor(nproc*(0*A - floor(0*A)));

  if (rank == recv_proc) {
    pq.push({heuristic(start_point, destination), start_point});
    open_list.insert(start_point);


  }
  g_cost.insert({start_point, 0});
  

  
  for (int i = 0; i < grid->dim; i++) {
    for (int j = 0; j < grid->dim; j++) {
      if (i != start_point / grid->dim || j != start_point % grid->dim) {
        g_cost.insert({i*grid->dim + j, INT_MAX});
      }
    }
  }

  int pathCost_buffer;
  int Recv_buffer[3];
  int Send_buffer[3];
  int parent_buffer[2];

  vector<int> nbrs;
  int node_left_rqst = 0;
  int path_left = 0;
  int parents_left = 0;

  while (true) { 
    int ready;
    if (!node_left_rqst) {
      // check whether or not a node is ready to be processed
      MPI_Irecv(&Recv_buffer, 3, MPI_INT, MPI_ANY_SOURCE, rank, MPI_COMM_WORLD, &node_reqs[rank]);
      node_left_rqst = 1;
    }
    MPI_Test(&node_reqs[rank], &ready, &node_stats[rank]);
    
    if (ready) {
      node_left_rqst = 0; 
      // _bufferfer can be processed 
      int nbr = Recv_buffer[0];
      int curr_cost = Recv_buffer[1];
      int curr = Recv_buffer[2]; 
      int nbr_fscore = curr_cost + heuristic(nbr, destination);
      int nbr_score = g_cost.at(nbr);
      if (closed_list.find(nbr) != closed_list.end()) {
        if (curr_cost < nbr_score) {
          closed_list.erase(nbr);
          open_list.insert(nbr);
          pq.push({nbr_fscore, nbr});
        } else {
          continue;
        }
      } else {
        if (open_list.find(nbr) == open_list.end()) {
          open_list.insert(nbr);
          pq.push({nbr_fscore, nbr});
        } else if (curr_cost >= nbr_score) {
          continue;
        }
      }

      parent_buffer[0] = nbr;
      
      parent_buffer[1] = curr;
      
      for (int i = 0; i < nproc; i++) {
        MPI_Isend(&parent_buffer, 2, MPI_INT, i, nproc, MPI_COMM_WORLD, &parent_reqs[i]);
      }

      g_cost.erase(nbr);
      g_cost.emplace(nbr, curr_cost);    
    }
    
    
    int parentUpdate; 
    if (!parents_left) {
      MPI_Irecv(&parent_buffer, 2, MPI_INT, MPI_ANY_SOURCE, nproc, MPI_COMM_WORLD, &parent_reqs[rank]);
      parents_left = 1;
    }

    MPI_Test(&parent_reqs[rank], &parentUpdate, &parent_stats[rank]);


    if(parentUpdate) {
      parents_left = 0;
      int nbr = parent_buffer[0];
      int curr = parent_buffer[1];
      if (parent.find(curr) != parent.end()) {
          if (parent.at(curr) != nbr) {
            parent.erase(nbr);
            parent.emplace(nbr, curr);
          }
        } else {
          parent.erase(nbr);
          parent.emplace(nbr, curr);
      }
    }
    
    int new_cost_flag;
    if (!path_left) {
      MPI_Irecv(&pathCost_buffer, 1, MPI_INT, MPI_ANY_SOURCE, nproc+1, MPI_COMM_WORLD, &cost_reqs[rank]);
      path_left = 1;
    }
    
    MPI_Test(&cost_reqs[rank], &new_cost_flag, &cost_stats[rank]);
    if (new_cost_flag) {
      path_left = 0;
      if (!valid_path_flag || pathCost_buffer < pathCost) {
        valid_path_flag = 1;
        pathCost = pathCost_buffer;
      }
    }

    if (pq.empty() || pq.top().cost >= pathCost) {
      if (valid_path_flag) {
        MPI_Status doneStatus;
        int done;
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &done, &doneStatus);
        if (done == 0) {
          break;
        }
      }
      continue;
    }
    
    int curr = pq.top().node;
    pq.pop();
    open_list.erase(curr);
    closed_list.insert(curr);

    // path found
    if (curr == destination) {
      double endSearchTime = MPI_Wtime();
      make_path(parent, curr, path);
      int new_cost = path->size();

      while (new_cost < 2 || path->back() != start_point) {
        
        int parentUpdate; 
        if (!parents_left) {
          MPI_Irecv(&parent_buffer, 2, MPI_INT, MPI_ANY_SOURCE, nproc, MPI_COMM_WORLD, &parent_reqs[rank]);
          parents_left = 1;
        }

        MPI_Test(&parent_reqs[rank], &parentUpdate, &parent_stats[rank]);
        if(parentUpdate) {
          parents_left = 0;
          int nbr = parent_buffer[0];
          int curr = parent_buffer[1];
          if (parent.find(curr) != parent.end()) {
              if (parent.at(curr) != nbr) {
                parent.erase(nbr);
                parent.emplace(nbr, curr);
              }
            } else {
              parent.erase(nbr);
              parent.emplace(nbr, curr);
          }
        }
        make_path(parent, curr, path);
        new_cost = path->size();
      }
      
      if (new_cost < pathCost) {
        
        double endPathTime = MPI_Wtime();
        for (int i = 0; i < nproc; i++) {
          MPI_Isend(&new_cost, 1, MPI_INT, i, nproc+1, MPI_COMM_WORLD, &cost_reqs[i]);
        }
      }
      continue;
    }

    nbrs = find_neigbours(curr, grid);
    for (int nbr: nbrs) {
      int curr_cost = g_cost.at(curr) + 1;
      Send_buffer[0] = nbr;
      Send_buffer[1] = curr_cost;
      Send_buffer[2] = curr;
      int recipient = floor(nproc*(nbr*A - floor(nbr*A)));;
      MPI_Isend(&Send_buffer, 3, MPI_INT, recipient, recipient, MPI_COMM_WORLD, &node_reqs[recipient]);
    }
    nbrs.clear();
  }
}

int main(int argc, char *argv[]) {
    int rank;
    int nproc;
    char* filename = NULL;

    int x1 = -1;
    int y1 = -1;
    int x2 = -1;
    int y2 = -1;

    MPI_Init(&argc, &argv);

   
    vector<int> coords;    

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "-f") {
            if (i + 1 < argc) {
                filename = argv[++i];
            } else {
                cerr << "Error: '-f' requires a filename\n";
                MPI_Finalize();
                return -1;
            }
        }
        else if (coords.size() < 4) {
          coords.push_back(stoi(arg));
        }
        
    }

    if (coords.size() != 4) {
        cerr << "Usage: " << argv[0]<< " -f <filename> <x1> <y1> <x2> <y2>\n";
        MPI_Finalize();
        return -1;
    }

    x1 = coords[0], y1 = coords[1], x2 = coords[2], y2 = coords[3];

    grid = readgrid(x1, y1, x2, y2, filename);
    int start_point = x1 * grid->dim + y1;
    int destination = x2 * grid->dim + y2;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    vector<int>* spath;
    vector<int> path_s;
    double start_s = MPI_Wtime();
    path_s = (seq_aStar(start_point, destination, grid));
    spath = &path_s;
    double end_s = MPI_Wtime();
    if(!rank){
      
      if (spath->size() != 0) {
          cout <<"Path length: "<< spath->size() << endl;
          for (auto n = spath->rbegin(); n != spath->rend(); n++) {
            cout << "(" <<  *n / grid->dim << "," << *n % grid->dim << ") "; 
          }
          cout<<endl;
      }
    }
    vector<int>* path = new vector<int>;
    double start = MPI_Wtime();
    parallel_aStar(start_point, destination, grid, path, nproc, rank);
    double end = MPI_Wtime();
    if(!rank){
      cout<<"Ending time :: Serial: "<<end - start<<endl;
      
      if (spath->size() != 0) {
          cout <<"Path length: "<<spath->size() << endl;
          for (auto n = spath->rbegin(); n != spath->rend(); n++) {
            cout << "(" <<  *n / grid->dim << "," << *n % grid->dim << ") "; 
          }
          cout<<endl;
      }
      cout<<"Execution Time  :: Parallel:  "<<end_s - start_s<<endl;
    }
    
    MPI_Finalize();
    
    delete path;
    free(grid->grid);
    return 0;
}

