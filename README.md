# Parallel A* Search Algorithm using MPI

This project implements both sequential and parallel versions of the A* (A-star) pathfinding algorithm on grid-based maps. The parallel implementation leverages the Message Passing Interface (MPI) to distribute computation across multiple processes in a distributed-memory environment.

###  Features

- Sequential A* search using priority queue and Manhattan distance heuristic.
- Parallel A* using MPI with distributed cost and parent maps.
- Asynchronous message passing with `MPI_Isend` / `MPI_Irecv` for efficient node expansion.

### 📁 Input Format

Text files representing 2D grid maps:
- First line: grid dimension (e.g., `100`)
- Following lines: binary grid data (`1` = traversable, `0` = wall)

### Instruction for compiling and running the code
- To run this project, you need to ensure that you have MPI installed on your system. 
- To compile the file, open terminal then go to the directory with the makefile and run command "make".
- To run the file, use the following command.
            "mpirun -np 4 ./Astar -f "filename" <x_start> <y_start> <x_dest> <y_dest> "

- The output for the file would be displayed.

- Example of Command to run file:


            '''

                make

                mpirun -np 4 ./A_Star -f ./Input/Input_simple.txt 0 0 4 4

                mpirun -np 4 ./A_Star -f ./Input/Input_maze_100.txt 0 0 99 99

            '''