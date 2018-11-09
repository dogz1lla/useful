#include <cmath>
#include <string>
#include <cstdio>
#include <limits>
#include <vector>
#include <iostream>
#include <algorithm>

//using namespace std;
int main() {
  /* Enter your code here. Read input from STDIN. Print output to STDOUT */
  const int n = 10; //number of nodes in the graph
  std::vector<int> nodes;
  for(int i=0; i<n; i++)
    nodes.push_back(i);
  int edges[n][n]; //matrix that shows whether a pair of nodes is connected
  srand(time(NULL));
  for(int i=0;i<n;i++){
    for(int j=0;j<n;j++){
      edges[i][j]=rand()%20;//initialized at random in this example
    }
  }

  /* find neighbors for each node */
  std::vector<std::vector<int> > neighbor_list;
  for(int i=0;i<n;i++) {
    neighbor_list.push_back(std::vector<int>(1));
    int s=neighbor_list.size();
    for(int j=0;j<n;j++) {
      if(i==j) continue;
      else {
	if(edges[i][j]>0) neighbor_list[s-1].push_back(j);
      }
    }
  }

  std::vector<int> visited;
  std::vector<int> not_visited;
  std::vector<int> dist(n);
  int inf = std::numeric_limits<int>::max();
  int current_node=nodes[0];//change it to the one specified by the problem
  int target_node=nodes[n-1];//change it to the one specified by the problem
  for(int i=0; i<n; i++)
    not_visited.push_back(i);
  for(int i=0;i<n;i++) {
    if(i==current_node)
      dist[i]=0;
    else
      dist[i]=inf;
  }

  int counter=1;
  int min_dist=0;
  while(current_node!=target_node) {
    min_dist=inf;
    for(int i=0;i<not_visited.size();i++) {
      int j=not_visited[i];
      if(dist[j]<=min_dist) {//only looks for one path, does not considers more than one alternatives
	min_dist=dist[j];
	current_node=j;
      }
    }
    visited.push_back(current_node);
    not_visited.erase(std::remove(not_visited.begin(), not_visited.end(), current_node), not_visited.end());
    for(int i=0;i<neighbor_list[current_node].size();i++) {
      int j=neighbor_list[current_node][i];
      int new_dist=dist[current_node]+edges[current_node][j];
      if(dist[j]>=new_dist) dist[j]=new_dist;
    }
    counter++;
  }

  for(int i=0;i<visited.size();i++)
    std::cout<<visited[i]<<std::endl;
  return 0;
}
