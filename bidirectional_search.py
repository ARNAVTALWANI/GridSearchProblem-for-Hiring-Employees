from collections import deque
import numpy as np
from search_problems import Node, GraphSearchProblem

def bidirectional_search(problem):
    """
        Implement a bidirectional search algorithm that takes instances of SimpleSearchProblem (or its derived
        classes) and provides a valid and optimal path from the initial state to the goal state.

        :param problem: instance of SimpleSearchProblem
        :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
                 num_nodes_expanded: number of nodes expanded by the search
                 max_frontier_size: maximum frontier size during search
        """
    
    max_frontier_size = 0
    num_nodes_expanded = 0
    path = []

    # Initialize the frontiers and explored sets for both the forward and backward searches
    forward_frontier = deque([Node(None, problem.init_state, None, 0)])
    backward_frontier = deque([Node(None, problem.goal_states[0], None, 0)])

    forward_explored = set()
    backward_explored = set()
    
    forward_parents = {problem.init_state: None}
    backward_parents = {goal: None for goal in problem.goal_states}

    meeting_point = None

    while forward_frontier and backward_frontier:
        # Forward search
        if forward_frontier:
            curr_node = forward_frontier.popleft()
            forward_explored.add(curr_node.state)
            num_nodes_expanded += 1

            if curr_node.state in backward_explored:
                meeting_point = curr_node.state
                break

            for action, neighbor in problem.get_actions(curr_node.state):
                if neighbor not in forward_explored and neighbor not in forward_parents:
                    forward_parents[neighbor] = curr_node.state
                    forward_frontier.append(Node(curr_node, neighbor, None, curr_node.path_cost + 1))
                    max_frontier_size = max(max_frontier_size, len(forward_frontier))

                    if neighbor in backward_explored:
                        meeting_point = neighbor
                        break

        # Backward search
        if backward_frontier:
            curr_node = backward_frontier.popleft()
            backward_explored.add(curr_node.state)
            num_nodes_expanded += 1

            if curr_node.state in forward_explored:
                meeting_point = curr_node.state
                break

            for action, neighbor in problem.get_actions(curr_node.state):
                if neighbor not in backward_explored and neighbor not in backward_parents:
                    backward_parents[neighbor] = curr_node.state
                    backward_frontier.append(Node(curr_node, neighbor, None, curr_node.path_cost + 1))
                    max_frontier_size = max(max_frontier_size, len(backward_frontier))

                    if neighbor in forward_explored:
                        meeting_point = neighbor
                        break
        
    if meeting_point:
        fwd_path = []
        curr_state = meeting_point
        while curr_state is not None:
            fwd_path.insert(0, curr_state)
            curr_state = forward_parents[curr_state]

        bwd_path = []
        curr_state = meeting_point
        while curr_state is not None:
            curr_state = backward_parents[curr_state]
            if curr_state is not None:
                bwd_path.append(curr_state)
            
        path = fwd_path + bwd_path

    return path, num_nodes_expanded, max_frontier_size


if __name__ == '__main__':
    # Simple example
    goal_states = [0]
    init_state = 9
    V = np.arange(0, 10)
    E = np.array([[0, 1],
                  [1, 2],
                  [2, 3],
                  [3, 4],
                  [4, 5],
                  [5, 6],
                  [6, 7],
                  [7, 8],
                  [8, 9],
                  [0, 6],
                  [1, 7],
                  [2, 5],
                  [9, 4]])
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)

    # Use stanford_large_network_facebook_combined.txt to make your own test instances
    E = np.loadtxt('../datasets/stanford_large_network_facebook_combined.txt', dtype=int)
    V = np.unique(E)
    goal_states = [349]
    init_state = 0
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = bidirectional_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)

    # Be sure to compare with breadth_first_search!