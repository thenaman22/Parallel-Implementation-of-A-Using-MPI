To run this project, you need to ensure that you have MPI installed on your system. 
To compile the file, open terminal then go to the directory with the makefile and run command "make".

To run the file, use the following command.
            "mpirun -np 4 ./Astar -f "filename" <x_start> <y_start> <x_dest> <y_dest> "

The output for the file would be displayed.

Example of Command to run file:


            '''

                make

                mpirun -np 4 ./A_Star -f ./Input/Input_simple.txt 0 0 4 4

                mpirun -np 4 ./A_Star -f ./Input/Input_maze_100.txt 0 0 99 99

            '''