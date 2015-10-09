import numpy as np
import sys

def links_to_edges(s):
    src, dests = s.split(': ')
    src = int(src)
    dests = [int(to) for to in dests.split(' ')]
    if src % 100000 == 0:
        print(src)
    for d in dests:
        if d != src:
            yield (src, d)

if __name__ == '__main__':
    num_nodes = 5716808

    links = np.array([edge for line in open(sys.argv[1]) for edge in links_to_edges(line)], dtype=np.uint32)

    np.savez("links.npz", links=links)
