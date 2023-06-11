from utils.utils import simulate_random_dag
import networkx as nx
import numpy as np
import math
from collections import defaultdict
import random

# thanks to 
class _Graph():
    r"""Helper class to handle graph properties.

    Parameters
    ----------
    vertices : list
        List of nodes.
    """

    def __init__(self, vertices):
        self.graph = defaultdict(list)
        self.V = vertices

    def addEdge(self, u, v):
        """Adding edge to graph."""
        self.graph[u].append(v)

    def isCyclicUtil(self, v, visited, recStack):
        """Utility function to return whether graph is cyclic."""
        # Mark current node as visited and
        # adds to recursion stack
        visited[v] = True
        recStack[v] = True

        # Recur for all neighbours
        # if any neighbour is visited and in
        # recStack then graph is cyclic
        for neighbour in self.graph[v]:
            if visited[neighbour] == False:
                if self.isCyclicUtil(neighbour, visited, recStack) == True:
                    return True
            elif recStack[neighbour] == True:
                return True

        # The node needs to be poped from
        # recursion stack before function ends
        recStack[v] = False
        return False

    def isCyclic(self):
        """Returns whether graph is cyclic."""
        visited = [False] * self.V
        recStack = [False] * self.V
        for node in range(self.V):
            if visited[node] == False:
                if self.isCyclicUtil(node, visited, recStack) == True:
                    return True
        return False

    def topologicalSortUtil(self, v, visited, stack):
        """A recursive function used by topologicalSort ."""
        # Mark the current node as visited.
        visited[v] = True

        # Recur for all the vertices adjacent to this vertex
        for i in self.graph[v]:
            if visited[i] == False:
                self.topologicalSortUtil(i, visited, stack)

        # Push current vertex to stack which stores result
        stack.insert(0, v)

    def topologicalSort(self):
        """A sorting function. """
        # Mark all the vertices as not visited
        visited = [False]*self.V
        stack = []

        # Call the recursive helper function to store Topological
        # Sort starting from all vertices one by one
        for i in range(self.V):
            if visited[i] == False:
                self.topologicalSortUtil(i, visited, stack)

        return stack

# thanks to Tigramite https://github.com/jakobrunge/tigramite
def structural_causal_process(links: dict,
                              T: int,
                              transient_fraction=0.2,
                              seed=None):
    random_state = np.random.RandomState(seed)

    N = len(links.keys())
    noises = [random_state.randn for j in range(N)]
    if N != max(links.keys())+1:
        raise ValueError("links keys must match N.")

    if isinstance(noises, np.ndarray):
        if noises.shape != (T + int(math.floor(transient_fraction*T)), N):
            raise ValueError(
                "noises.shape must match ((transient_fraction + 1)*T, N).")
    else:
        if N != len(noises):
            raise ValueError("noises keys must match N.")

    # Check parameters
    max_lag = 0
    contemp_dag = _Graph(N)
    for j in range(N):
        for link_props in links[j]:
            var, lag = link_props[0]
            coeff = link_props[1]
            func = link_props[2]
            if lag == 0:
                contemp = True
            if var not in range(N):
                raise ValueError("var must be in 0..{}.".format(N-1))
            if 'float' not in str(type(coeff)):
                raise ValueError("coeff must be float.")
            if lag > 0 or type(lag) != int:
                raise ValueError("lag must be non-positive int.")
            max_lag = max(max_lag, abs(lag))

            # Create contemp DAG
            if var != j and lag == 0:
                contemp_dag.addEdge(var, j)

    if contemp_dag.isCyclic() == 1:
        raise ValueError("Contemporaneous links must not contain cycle.")

    causal_order = contemp_dag.topologicalSort()
    transient = int(math.floor(transient_fraction*T))

    data = np.zeros((T+transient, N), dtype='float32')
    for j in range(N):
        if isinstance(noises, np.ndarray):
            data[:, j] = noises[:, j]
        else:
            data[:, j] = noises[j](T+transient)

    for t in range(max_lag, T+transient):
        for j in causal_order:
            # This loop is only entered if intervention_type != 'hard'
            for link_props in links[j]:
                var, lag = link_props[0]
                coeff = link_props[1]
                func = link_props[2]
                data[t, j] += coeff * func(data[t + lag, var])

    data = data[transient:]

    nonstationary = (np.any(np.isnan(data)) or np.any(np.isinf(data)))

    return data, nonstationary


def generate_ode(d: int,
                 degree: int,
                 max_lag: int,
                 lag_edge_prob: float,
                 w_range: tuple = (0.5, 1.0),
                 func=lambda x: x):
    if d <= 0 or degree <= 0:
        raise ValueError("d and degree must larger than 0")
    if max_lag < 0 or lag_edge_prob < 0:
        raise ValueError("max_lag and lag_edge_prob must no less than 0")

    links = {i: [] for i in range(d)}
    links_with_out_func = {i: [] for i in range(d)}

    ode_graphs = np.zeros((d, d, max_lag+1))
    G = simulate_random_dag(d, degree, w_range)
    W = nx.to_numpy_array(G)

    for i in range(d):
        for j in range(d):
            if W[i,j]!=0:
                k = random.randint(0, max_lag)
                ode_graphs[i,j,k] = W[i,j]
                links[j].append(((i, -k), ode_graphs[i, j, k], func))
                links_with_out_func[j].append(
                            ((i, -k), ode_graphs[i, j, k]))
    return links, links_with_out_func


if __name__ == "__main__":
    import os
    import json
    d_list = [5, 10, 15]
    degree_list = [5, 8, 10]
    max_lag_list = [2, 2, 4]
    lag_edge_prob = 0.5

    for i in range(len(d_list)):
        for j in range(10):
            links, links_with_out_func = generate_ode(
                "sparse", d_list[i], degree_list[i], max_lag_list[i], lag_edge_prob)
            T = 1000
            data, nonstat = structural_causal_process(links, T)
            # Data must be array of shape (time, variables)
            path = "data/{}_{}_{}/{}".format(d_list[i],
                                             degree_list[i], max_lag_list[i], j)
            if not os.path.exists(path):
                os.makedirs(path)
            np.save("{}/{}".format(path, "data.npy"), data)
            a = json.dumps(links_with_out_func, indent=4,
                           separators=(",", ": "))
            with open("{}/{}".format(path, "links.json"), "w") as f:
                f.write(a)
