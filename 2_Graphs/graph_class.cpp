#include <cmath>
#include <iostream>
#include <cstring>
#include <string>
#include <algorithm>
#include <vector>
#include <stdlib.h>
#include <limits>

/* Description of the Graph class. */
class Graph {
  /* Graph class describes a simple graph that consists of n nodes,
   that may be connected by edges with weights. The direction along 
  which the edge is travelled is irrelevant. Method create_neighbor_list 
  returns a vector of neighbors of ith node stored as a vector at ith
  position. Method shortest_path returns a vector containing nodes that 
  belong to the shortest path from the start node to the finish node 
  (returned list stores them in the reverse order). */
private:
  int n;
  std::vector<int> nodes;
  std::vector<std::vector<int> > edges;
public:
  Graph(int);
  Graph(int, std::vector<int>, std::vector<std::vector<int> >);
  std::vector<std::vector<int> > create_neighbor_list(void);
  std::vector<int> shortest_path(int start,
				 int finish, 
				 std::vector<std::vector<int> > neighbor_list);
};

Graph::Graph(int N) : n(N), nodes(n,0), edges(n,std::vector<int>(n,0)) {};

Graph::Graph(int N, std::vector<int> Nodes, std::vector<std::vector<int> > Edges) {
  n = N;
  nodes = Nodes;
  edges = Edges;
};

std::vector<std::vector<int> > Graph::create_neighbor_list(void){
  std::vector<std::vector<int> > nl(n,std::vector<int> (n,0));
  for(int i=0; i<n-1; i++) {
    for(int j=i+1; j<n; j++) {
      if(edges[i][j]>0) {
	nl[i].push_back(j);
	nl[j].push_back(i);
      }
    }
  }
  return nl;
}

std::vector<int> Graph::shortest_path(int start,
				      int finish,
				      std::vector<std::vector<int> > neighbor_list) {
  /* This routine searches a shortest path for an undirected graph
   given the starting and the destination nodes. The Dijkstra's algorithm
  is used. The routine is not able to distinguish between two or more 
  shortest paths that are of the same length. The path is returned as 
  a vector of nodes in the order from the destination to start. */
  const int inf = std::numeric_limits<int>::max();
  int counter = 1, min_dist = 0, current_node = start;
  std::vector<int> visited;
  std::vector<int> not_visited;
  std::vector<int> dist(n);
  std::vector<int> pred(n);
  std::vector<int> path;
  for(int i=0; i<n; i++) 
    not_visited.push_back(i);
  for(int i=0; i<n; i++) {
    if(i==start)
      dist[i]=0;
    else
      dist[i]=inf;
  }

  while(current_node!=finish) {
    min_dist=inf;
    for(unsigned int i=0; i<not_visited.size(); i++) {
      int j = not_visited[i];
      if(dist[j]<=min_dist) {//only looks for one path, does not considers more than one alternatives
	min_dist = dist[j];
	current_node = j;
      }
    }
    visited.push_back(current_node);
    not_visited.erase(std::remove(not_visited.begin(), not_visited.end(), current_node), not_visited.end());
    for(unsigned int i=0; i<neighbor_list[current_node].size(); i++) {
      int j=neighbor_list[current_node][i];
      int new_dist=dist[current_node]+edges[current_node][j];
      if(dist[j]>=new_dist) {
	dist[j]=new_dist;
	pred[j] = current_node;
      }
    }
    counter++;
  }

  path.push_back(current_node);
  do{
    current_node = pred[current_node];
    path.push_back(current_node);
  }while(current_node!=start);

  return path;
}

int main() {
  
  return 1;
}
