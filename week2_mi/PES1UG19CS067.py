"""
You can create any other helper funtions.
Do not modify the given functions
"""
import heapq as hq

#Helper functions
def def_parents(cost):
    parents_arr=[]
    for i in range(len(cost)):
        parents_arr.append(-1)
    return parents_arr

def debug_path(path):
    for i in path:
        print(i)
        print()

def final_path(parents,parent_node,start_point):
    if start_point==parent_node:
        return [start_point,]
    path=[parent_node,]
    while parent_node!=start_point:
        parent_node=parents[parent_node]
        path.append(parent_node)
    path.reverse()
    return path

def check_node(child,node,visited,cost):
    return child in visited or cost[node][child]<=0

def check_node_astar(child, node,cost,min_cost):
    return child not in min_cost or min_cost[node]+cost[node][child]<=min_cost[child]

#Traversal functions

def A_star_Traversal(cost, heuristic, start_point, goals):
    """
    Perform A* Traversal and find the optimal path 
    Args:
        cost: cost matrix (list of floats/int)
        heuristic: heuristics for A* (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from A*(list of ints)
    """
    path=[]
    visited_067 = set()
    parents_067=def_parents(cost)

    min_func_067={start_point:0}
    priority_Q_067=[(0,start_point)]
    parents_067[start_point]=start_point

    while(len(priority_Q_067)>0):
        node_067=hq.heappop(priority_Q_067)[1]
        #print(node_067)

        if(node_067 in visited_067):
            continue
        
        visited_067.add(node_067)

        if node_067 in goals:
            #debug_path(final_path(parents_067,node_067,start_point))
            return final_path(parents_067,node_067,start_point)
        
        else:
            for child_067 in range(1,len(cost[node_067])):
                if check_node(child_067,node_067,visited_067,cost):
                    continue

                if check_node_astar(child_067,node_067,cost,min_func_067):

                    parents_067[child_067]=node_067
                    min_func_067[child_067]=min_func_067[node_067]+cost[node_067][child_067]

                hq.heappush(priority_Q_067,(min_func_067[child_067]+heuristic[child_067],child_067))

    return path


def DFS_Traversal(cost, start_point, goals):
    """
    Perform DFS Traversal and find the optimal path 
        cost: cost matrix (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from DFS(list of ints)
    """
    path=[]
    visited_067 = set()
    parents_067=def_parents(cost)
    stack_067=[]

    stack_067.append(start_point)
    parents_067[start_point]=start_point

    while (len(stack_067)>0):
        node_067=stack_067.pop()

        if(node_067 in visited_067):
            continue

        visited_067.add(node_067)
        
        if node_067 in goals:
            #debug_path(final_path(parents_067,node_067,start_point))
            #debug_path(stack)
            return final_path(parents_067,node_067,start_point)

        else:
            for child_067 in range(len(cost[node_067])-1,0,-1):
                if check_node(child_067,node_067,visited_067,cost):
                    continue

                parents_067[child_067]=node_067
                stack_067.append(child_067)

    return path


