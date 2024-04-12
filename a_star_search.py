import queue
import numpy as np
from search_problems import Node, GridSearchProblem, get_random_grid_problem


def a_star_search(problem):
    """
    Uses the A* algorithm to solve an instance of GridSearchProblem. Use the methods of GridSearchProblem along with
    structures and functions from the allowed imports (see above) to implement A*.

    :param problem: an instance of GridSearchProblem to solve
    :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
             num_nodes_expanded: number of nodes expanded by your search
             max_frontier_size: maximum frontier size during search
    """

    num_nodes_expanded = 0
    max_frontier_size = 0
    path = []

    # Initialize the frontier and explored set
    frontier = queue.PriorityQueue()
    frontier.put((0, problem.init_state))
    explored = set()

    # Initialize the cost and parent dictionaries
    cost = {problem.init_state: 0}
    parent = {problem.init_state: None}

    while not frontier.empty():
        # Pop the node with the lowest f-value
        curr_cost, curr_state = frontier.get()
        num_nodes_expanded += 1

        # Check if the current state is the goal state
        if problem.goal_test(curr_state):
            while curr_state is not None:
                path.append(curr_state)
                curr_state = parent[curr_state]
            path.reverse()
            return path, num_nodes_expanded, max_frontier_size

        # Add the current state to the explored set
        explored.add(curr_state)

        # Expand the current state
        for action in problem.get_actions(curr_state):
            child_state = problem.transition(curr_state, action)
            new_cost = cost[curr_state] + problem.action_cost(curr_state, action, child_state)

            # Check if the child state is in the frontier
            if child_state not in explored and (child_state not in cost or new_cost < cost[child_state]):
                cost[child_state] = new_cost
                priority = new_cost + problem.heuristic(child_state)
                frontier.put((priority, child_state))
                max_frontier_size = max(max_frontier_size, frontier.qsize())
                parent[child_state] = curr_state

    return path, num_nodes_expanded, max_frontier_size

def search_phase_transition():
    """
    Simply fill in the prob. of occupancy values for the 'phase transition' and peak nodes expanded within 0.05. You do
    NOT need to submit your code that determines the values here: that should be computed on your own machine. Simply
    fill in the values!

    :return: tuple containing (transition_start_probability, transition_end_probability, peak_probability)
    """
    
    transition_start_probability = 0.35
    transition_end_probability = 0.45
    peak_nodes_expanded_probability = 0.40
    return transition_start_probability, transition_end_probability, peak_nodes_expanded_probability

'''
def run_experiment(p_occ, M, N, n_runs):
    solvable_count = 0
    total_nodes_generated = 0

    for _ in range(n_runs):
        problem = get_random_grid_problem(p_occ, M, N)
        path, num_nodes_expanded, _ = a_star_search(problem)
        correct = problem.check_solution(path)

        if correct:
            solvable_count += 1
        total_nodes_generated += num_nodes_expanded

    return solvable_count / n_runs, total_nodes_generated / n_runs
'''

if __name__ == '__main__':
    
    # Test your code here!
    # Create a random instance of GridSearchProblem
    p_occ = 0.25
    M = 500
    N = 500
    problem = get_random_grid_problem(p_occ, M, N)
    # Solve it
    path, num_nodes_expanded, max_frontier_size = a_star_search(problem)
    # Check the result
    correct = problem.check_solution(path)
    print("Solution is correct: {:}".format(correct))
    # Plot the result
    problem.plot_solution(path)
    '''

    # Experiment and compare with BFS
    n_runs = 100
    p_occ_values = np.arange(0.1, 0.91, 0.05)
    solvable_ratios = []
    nodes_generated_avg = []

    # Run the experiment for N = 500
    for p_occ in p_occ_values:
        solvable, nodes_generated = run_experiment(p_occ, 500, 500, n_runs)
        solvable_ratios.append(solvable)
        nodes_generated_avg.append(nodes_generated)

    # Plot the results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 7))

    # Plot solvability ratio
    plt.subplot(1, 2, 1)
    plt.plot(p_occ_values, solvable_ratios, marker='o')
    plt.title('Solvability Ratio for N = 500')
    plt.xlabel('Occupancy Probability')
    plt.ylabel('Solvability Ratio')

    # Plot average nodes generated
    plt.subplot(1, 2, 2)
    plt.plot(p_occ_values, nodes_generated_avg, marker='o')
    plt.title('Average Nodes Generated for N = 500')
    plt.xlabel('Occupancy Probability')
    plt.ylabel('Average Nodes Generated')

    plt.tight_layout()
    plt.show()
    '''