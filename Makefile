APP_NAME=A_Star
OBJS += A_Star.o

CXX = mpic++
CXXFLAGS = -I. -std=c++17 -O2 -Wall -Wextra -pedantic

default: $(APP_NAME)

$(APP_NAME): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

%.o: %.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

clean:
	/bin/rm -rf *~ *.o $(APP_NAME) *.class
