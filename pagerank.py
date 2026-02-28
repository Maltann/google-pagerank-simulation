import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

def draw_graph(M):
    n = len(M)
    G = nx.DiGraph()

    # creating the links between the nodes
    for i in range(n):
        for j in range(n):
            if M[j][i] != 0:
                G.add_edge(i+1, j+1, weight=round(M[j][i], 3))

    # if the graph is planar, outputing a planar graph representation, else doing regular
    is_planar, embedding = nx.check_planarity(G)
    print(is_planar)

    if(is_planar):
        pos = nx.planar_layout(G)
    else:
        pos = nx.spring_layout(G)

    nx.draw(G, pos, with_labels=True, node_size=(10000/n), node_color="lightblue", arrows=True) #draw nodes

    # adding the weights to the graph
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # exporting the graph as a .png file
    plt.savefig("graph.png", dpi=300, bbox_inches="tight")

    return True


def generate_random_network(n):
    """
    Generate a random graph of a web network

    Parameters
    ----------
    n : int
        the number of the pages in the network

    Returns
    -------
    numpy array
        adjacency matrix where M_i,j represents the link from 'j' to 'i', such that for all 'j'
        sum(i, M_i,j) = 1
    """
    M = np.zeros((n,n)) # generate a nxn matrix full of zeros
    
    for i in range(n):
        already_linked = [] # setup a list of the already linked webpages on the current webpage
        number_of_links = min(random.randint(0, n-1), 100) # take a random amount of links to have on the page, 100 maximum
        for j in range(number_of_links):
            # taking a page where there is no current link + not creating self links
            page_number = random.randint(0, n-1)
            while (page_number in already_linked) or (page_number == i):
                page_number = random.randint(0, n-1)

            M[page_number][i] = 1 / (number_of_links) # putting the weight of the link in the matrix

            already_linked.append(page_number) # add the page to the already linked pages

    return M # return the matrix of the graph


def pagerank(M, d: float = 0.85):
    """PageRank algorithm with explicit number of iterations. Returns ranking of nodes (pages) in the adjacency matrix.

    Parameters
    ----------
    M : numpy array
        adjacency matrix where M_i,j represents the link from 'j' to 'i', such that for all 'j'
        sum(i, M_i,j) = 1
    d : float, optional
        damping factor, by default 0.85

    Returns
    -------
    numpy array
        a vector of ranks such that v_i is the i-th rank from [0, 1],

    """
    occur = 0

    N = M.shape[1]
    w = np.ones(N) / N
    M_hat = d * M
    v = M_hat @ w + (1 - d) / N
    while np.linalg.norm(w - v) >= 1e-10:
        w = v
        v = M_hat @ w + (1 - d) / N
        occur+=1

    print(f"found perron vector in {occur} operations")
    return v

""" manual example of a graph (basic one)
M = np.array([[0, 0, 0, 0, 0],
              [0.33, 0, 0, 0, 0],
              [0.33, 0, 0, 0, 1],
              [0.33, 0.5, 1, 0, 0],
              [0, 0.5, 0, 1, 0]])

v = pagerank(M, 0.85)
print(v)"""

n = int(input("how many websites you want on the internet : "))

A = generate_random_network(n)
print(A) # displaying the generated matrix
print("\n")

w = pagerank(A, 0.85)
print(w)

if len(A) < 30:
    draw_graph(A) # exporting the graph of the randomly generated network if less than 30 websites