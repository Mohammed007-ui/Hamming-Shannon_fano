# Hamming-Shannon_fano
Consider a discrete memoryless source with symbols and statistics {0.125, 0.0625, 0.25, 0.0625, 0.125, 0.125, 0.25} for its output. 
Apply the Huffman and Shannon-Fano to this source. 
Show that draw the tree diagram, the average code word length, Entropy, Variance, Redundancy, Efficiency.
Aim
To implement Huffman and Shannon-Fano encoding for a discrete memoryless source, analyze their performance, and compare metrics such as entropy, average code length, variance, redundancy, and efficiency. Additionally, visualize the Huffman tree.

Tools Required
-> Python/Matlab for implementation
-> Pen & Paper for tree diagrams

Program
import numpy as np
import heapq
import networkx as nx
import matplotlib.pyplot as plt

# Given probabilities
probabilities = np.array([0.125, 0.0625, 0.25, 0.0625, 0.125, 0.125, 0.25])

# Calculate Entropy (H)
entropy = -np.sum(probabilities * np.log2(probabilities))
print(f"Entropy (H): {entropy:.4f} bits")

# Huffman Coding
class Node:
    def __init__(self, prob, symbol, left=None, right=None):
        self.prob = prob
        self.symbol = symbol
        self.left = left
        self.right = right
    def __lt__(self, other):
        return self.prob < other.prob

def huffman_coding(probabilities):
    heap = [Node(prob, str(i)) for i, prob in enumerate(probabilities)]
    heapq.heapify(heap)
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(left.prob + right.prob, left.symbol + right.symbol, left, right)
        heapq.heappush(heap, merged)
    huffman_codes = {}
    def generate_codes(node, code=""):
        if node:
            if not node.left and not node.right:
                huffman_codes[node.symbol] = code
            generate_codes(node.left, code + "0")
            generate_codes(node.right, code + "1")
    generate_codes(heap[0])
    return huffman_codes, heap[0]

huffman_codes, huffman_tree = huffman_coding(probabilities)
avg_code_length_huffman = np.sum(probabilities * np.array([len(huffman_codes[str(i)]) for i in range(len(probabilities))]))
variance_huffman = np.sum(probabilities * (np.array([len(huffman_codes[str(i)]) for i in range(len(probabilities))]) - avg_code_length_huffman) ** 2)
redundancy_huffman = avg_code_length_huffman - entropy
efficiency_huffman = (entropy / avg_code_length_huffman) * 100

print("\nHuffman Codes:")
for symbol, code in sorted(huffman_codes.items()):
    print(f"Symbol {symbol}: {code}")

# Shannon-Fano Coding
def shannon_fano_coding(symbols, probabilities):
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_symbols = [symbols[i] for i in sorted_indices]
    sorted_probs = probabilities[sorted_indices]
    codes = {symbol: "" for symbol in sorted_symbols}
    def recursive_partition(start, end):
        if start >= end:
            return
        total = sum(sorted_probs[start:end+1])
        accum = 0
        partition = start
        while partition < end:
            accum += sorted_probs[partition]
            if accum >= total / 2:
                break
            partition += 1
        for i in range(start, partition+1):
            codes[sorted_symbols[i]] += "0"
        for i in range(partition+1, end+1):
            codes[sorted_symbols[i]] += "1"
        recursive_partition(start, partition)
        recursive_partition(partition+1, end)
    recursive_partition(0, len(sorted_probs) - 1)
    return {str(i): codes[s] for i, s in enumerate(sorted_symbols)}

shannon_fano_codes = shannon_fano_coding(list(range(len(probabilities))), probabilities)
avg_code_length_shannon_fano = np.sum(probabilities * np.array([len(shannon_fano_codes[str(i)]) for i in range(len(probabilities))]))
variance_shannon_fano = np.sum(probabilities * (np.array([len(shannon_fano_codes[str(i)]) for i in range(len(probabilities))]) - avg_code_length_shannon_fano) ** 2)
redundancy_shannon_fano = avg_code_length_shannon_fano - entropy
efficiency_shannon_fano = (entropy / avg_code_length_shannon_fano) * 100

print("\nShannon-Fano Codes:")
for symbol, code in sorted(shannon_fano_codes.items()):
    print(f"Symbol {symbol}: {code}")

# Summary Table
print("\nComparison:")
print(f"{'Metric':<20}{'Huffman':<15}{'Shannon-Fano'}")
print(f"{'-'*50}")
print(f"{'Entropy (H)':<20}{entropy:.4f}{'-':<15}")
print(f"{'Avg. Code Length (L)':<20}{avg_code_length_huffman:.4f}{avg_code_length_shannon_fano:.4f}")
print(f"{'Variance':<20}{variance_huffman:.4f}{variance_shannon_fano:.4f}")
print(f"{'Redundancy (R)':<20}{redundancy_huffman:.4f}{redundancy_shannon_fano:.4f}")
print(f"{'Efficiency (%)':<20}{efficiency_huffman:.2f}{efficiency_shannon_fano:.2f}")

# Function to draw tree
def draw_tree(node, pos=None, x=0, y=0, layer=1, graph=None):
    if graph is None:
        graph = nx.DiGraph()
    if pos is None:
        pos = {}
    pos[node.symbol] = (x, y)
    if node.left:
        graph.add_edge(node.symbol, node.left.symbol, label='0')
        pos = draw_tree(node.left, pos, x - 1 / 2 ** layer, y - 1, layer + 1, graph)
    if node.right:
        graph.add_edge(node.symbol, node.right.symbol, label='1')
        pos = draw_tree(node.right, pos, x + 1 / 2 ** layer, y - 1, layer + 1, graph)
    return pos

def plot_tree(tree, title):
    graph = nx.DiGraph()
    pos = draw_tree(tree, graph=graph)
    plt.figure(figsize=(8, 5))
    nx.draw(graph, pos, with_labels=True, node_size=2000, node_color='lightblue')
    edge_labels = {(u, v): d['label'] for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    plt.title(title)
    plt.show()

plot_tree(huffman_tree, "Huffman Coding Tree")

Calculation
Results.
