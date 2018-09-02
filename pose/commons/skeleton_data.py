import numpy as np
from scipy.io import loadmat

limb_parents = [1, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, 14, 14, 1, 4, 7, 10, 13]
limb_parents = np.array(limb_parents).astype(int)

limb_ratios = [2.0, 2.5,
               1.37, 2.8, 2.4,
               1.37, 2.8, 2.4,
               1.05, 4.2, 3.6,
               1.05, 4.2, 3.6,
               0, 2.25, 0.8,
               1.2, 1.2,
               2.0, 2.0]

joint_names = ['head_top', 'neck',
               'right_shoulder', 'right_elbow', 'right_wrist',
               'left_shoulder', 'left_elbow', 'left_wrist',
               'right_hip', 'right_knee', 'right_ankle',
               'left_hip', 'left_knee', 'left_ankle',
               'pelvis', 'spine', 'head',
               'right_hand', 'left_hand',
               'right_foot', 'left_foot',
               ]

limb_root = 14
head_limb = 0

limb_ratios = np.array(limb_ratios)
limb_ratios /= limb_ratios[head_limb]

joint_names = np.array(joint_names)

n_limbs = len(limb_parents)


def gen_limb_graph(limb_parents):
    n = len(limb_parents)
    G = [[] for i in xrange(n)]
    for i in range(n):
        j = limb_parents[i]
        if i != j:
            G[j].append(i)
    return G


def bfs_order(G, root):
    from collections import deque
    q = deque([root])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in G[u]:
            q.append(v)
    return order


limb_graph = gen_limb_graph(limb_parents)
limb_order = bfs_order(limb_graph, limb_root)