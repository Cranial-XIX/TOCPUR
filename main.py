import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import time

from copy import deepcopy
from mip import Model, xsum, maximize, BINARY, INTEGER

################################################################################
#
# The Single-day Method for PCP Problem:
# 1) greedy
# 2) TOP
# 3) TOCP
#
################################################################################
class SingleDayMethod:
    def __init__(self, nodes=None, edges=None):
        self.nodes = nodes
        self.edges = edges
        self.init_graph()

    def init_graph(self):
        self.n = self.nodes.shape[0]
        n = self.n

        self.shortest_path = [[[] for i in range(n)] for j in range(n)]
        self.min_edge_length = 100
        for i in range(n):
            for j in range(n):
                if self.edges[i,j] < 100:
                    self.shortest_path[i][j] = [i,j]
                    if self.edges[i,j] < self.min_edge_length and i != j:
                        self.min_edge_length = self.edges[i,j]

        self.dist = deepcopy(self.edges)
        self.floyd_warshall()
        
    def floyd_warshall(self): 
        n = self.n
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    ikj = self.dist[i,k] + self.dist[k,j]
                    if ikj < self.dist[i,j]:
                        self.dist[i,j] = ikj
                        path_ik = deepcopy(self.shortest_path[i][k])
                        path_kj = deepcopy(self.shortest_path[k][j])
                        path_ij = path_ik + path_kj[1:]
                        self.shortest_path[i][j] = path_ij

    def greedy(self, v,
               mustvisits,
               weights,
               max_length,
               n_agents):

        paths = [[v] for _ in range(n_agents)]

        m_leftover = set(mustvisits)
        if v in m_leftover:
            m_leftover.remove(v)
        travelled_lengths = [0 for _ in range(n_agents)]
        travelled_weights = [0 for _ in range(n_agents)]
        finished = [False for _ in range(n_agents)]

        n_leftover = [x for x in range(self.n) if x not in m_leftover and x != v]

        while sum(finished) < n_agents:
            for a in range(n_agents):
                if not finished[a]:
                    curr_node = paths[a][-1]
                    length = travelled_lengths[a]
                    found = False
                    if len(m_leftover) > 0: # prioritize must-visit node
                        min_dist = np.inf
                        next_visit = None
                        for next_node in m_leftover:
                            if self.dist[curr_node, next_node] < min_dist and \
                                    length + self.dist[curr_node, next_node] + \
                                    self.dist[next_node, v] < max_length:
                                next_visit = next_node
                                min_dist = self.dist[curr_node, next_node]
                        if next_visit is not None: # can find a must-visit node
                            found = True
                            path = deepcopy(
                                self.shortest_path[curr_node][next_visit][1:])
                            paths[a] = paths[a] + path
                            for p in path:
                                if p in m_leftover:
                                    m_leftover.remove(p)
                                    travelled_weights[a] += weights[p]
                                if p in n_leftover:
                                    n_leftover.remove(p)
                                    travelled_weights[a] += weights[p]
                            travelled_lengths[a] += \
                                    self.dist[curr_node, next_visit]
                    if not found: # search for a non must-visit node then
                        max_rate = 0
                        next_visit = None
                        for next_node in n_leftover:
                            if weights[next_node] / \
                                self.dist[curr_node, next_node] > max_rate and \
                                length + self.dist[curr_node, next_node] + \
                                self.dist[next_node, v] < max_length:
                                next_visit = next_node
                                max_rate = weights[next_node] / (
                                        self.dist[curr_node, next_node] + 1e-10)
                        if next_visit is not None: # find some node to visit
                            path = deepcopy(
                                self.shortest_path[curr_node][next_visit][1:])
                            paths[a] = paths[a] + path
                            for p in path:
                                if p in m_leftover:
                                    m_leftover.remove(p)
                                    travelled_weights[a] += weights[p]
                                if p in n_leftover:
                                    n_leftover.remove(p)
                                    travelled_weights[a] += weights[p]
                            travelled_lengths[a] += \
                                    self.dist[curr_node, next_visit]
                        else: # cannot visit any node, just return
                            path = deepcopy(self.shortest_path[curr_node][v][1:])
                            paths[a] = paths[a] + path
                            for p in path:
                                if p in m_leftover:
                                    m_leftover.remove(p)
                                    travelled_weights[a] += weights[p]
                                if p in n_leftover:
                                    n_leftover.remove(p)
                                    travelled_weights[a] += weights[p]
                            travelled_lengths[a] += \
                                    self.dist[curr_node, next_visit]
                            finished[a] = True
        # build the connectivity matrix
        X = np.zeros((self.n, self.n, n_agents))
        for m, path in enumerate(paths):
            prev = path[0]
            for v in path[1:]:
                X[prev, v, m] = 1
                prev = v
        return X

    def top(self, v,
            mustvisits,
            weights,
            max_length,
            n_agents):
        """
        weights: [n,]
        """
        N = self.n
        M = n_agents
        L = max_length
        w = list(weights)
        MAX_VISIT = int(L / self.min_edge_length)
        l = list(np.array(self.edges).reshape(-1))

        MD = Model("plan")
        MD.verbose = 0
        x = [MD.add_var(var_type=BINARY) for i in range(N*N*M)]
        y = [MD.add_var(var_type=BINARY) for i in range(N*M)]
        u = [MD.add_var(var_type=INTEGER) for i in range(N*M)]
        z = [MD.add_var(var_type=BINARY) for i in range(N)]

        MD.objective = maximize(xsum(z[i] * w[i] for i in range(N)))
        MD.emphasis = 1

        for i in mustvisits:
            MD += (z[i] == 1)

        for m in range(M):
            MD += xsum(x[j*M+m] for j in range(1, N)) == 1

        for m in range(M):
            MD += xsum(x[i*M*N+m] for i in range(1, N)) == 1

        for i in range(1, N):
            MD += sum(y[i*M+m] for m in range(M)) >= z[i]
            MD += sum(y[i*M+m] for m in range(M)) <= z[i] * M

        for m in range(M):
            for k in range(N):
                MD += xsum(x[i*M*N+k*M+m] for i in range(N)) == y[k*M+m]
                MD += xsum(x[k*M*N+i*M+m] for i in range(N)) == y[k*M+m]

        for m in range(M):
            c5 = []
            for i in range(N):
                for j in range(N):
                    c5.append(l[i*N+j]*x[i*M*N+j*M+m])
            MD += xsum(iter(c5)) <= L

        for m in range(M):
            for i in range(1, N):
                for j in range(1, N):
                    MD += (u[i*M+m] - u[j*M+m] + 1) <= (N-1)*(1-x[i*M*N+j*M+m])
                    MD += 2 <= u[i*M+m]
                    MD += u[i*M+m] <= N
        MD.optimize(max_seconds=1000)

        X = np.zeros((N, N, M))
        for m in range(M):
            for i in range(N):
                for j in range(N):
                    X[i,j,m] = x[i*N*M+j*M+m].x
        return X

    def tocp(self, v,
             mustvisits,
             weights,
             max_length,
             n_agents):
        """
        weights: [n,]
        """
        N = self.n
        M = n_agents
        L = max_length
        w = list(weights)
        MAX_VISIT = int(L / self.min_edge_length)
        l = list(np.array(self.edges).reshape(-1))

        MD = Model("plan")
        MD.verbose = 0
        x = [MD.add_var(var_type=BINARY)  for i in range(N*N*M)]
        y = [MD.add_var(var_type=BINARY)  for i in range(N*M)]
        u = [MD.add_var(var_type=INTEGER) for i in range(N*N*M)]
        z = [MD.add_var(var_type=BINARY)  for i in range(N)]

        # objective: maximize the collected rewards
        MD.objective = maximize(xsum(w[i] * z[i] for i in range(N)))

        # condition 1: must visit all must-visit nodes
        for i in mustvisits:
            MD += (z[i] == 1)

        # condition 2: each vehicle comes out of and goes back in node 1 once
        for m in range(M):
            MD += xsum(x[j*M+m] for j in range(1, N)) == 1
            MD += xsum(x[j*M*N+m] for j in range(1, N)) == 1
            for i in range(N):
                MD += x[i*N*M+i*M+m] == 0
                #for j in range(N):
                #    if self.edges[i,j] > 100:
                #        MD += x[i*N*M+j*M+m] == 0

        # condition 3: z_i should be 1 if we ever visit v_i
        #              y_im should be 1 if m ever visit v_i
        for m in range(M):
            for i in range(N):
                MD += xsum(x[i*N*M+j*M+m] for j in range(N)) >= y[i*M+m]
                MD += xsum(
                        x[i*N*M+j*M+m] for j in range(N)) <= y[i*M+m] * MAX_VISIT
        for i in range(N):
            MD += z[i] * M >= xsum(y[i*M+m] for m in range(M))
            MD += z[i] <= xsum(y[i*M+m] for m in range(M))

        # condition 4: flow conservation
        for m in range(M):
            for i in range(N):
                MD += xsum(x[i*M*N+j*M+m] for j in range(N)) == xsum(
                        x[j*M*N+i*M+m] for j in range(N))

        # condition 5: travel within budget
        for m in range(M):
            c5 = []
            for i in range(N):
                for j in range(N):
                    c5.append(l[i*N+j]*x[i*M*N+j*M+m])
            MD += xsum(iter(c5)) <= L

        # condition 6: preserve strong connectivity
        for m in range(M):
            for i in range(N):
                MD += u[i*N*M+i*M+m] == 0

            # the source should be (out - in = sum_i=1^N y_im)
            MD += (xsum(u[j*M+m] for j in range(N)) \
                    - xsum(u[i*N*M+m] for i in range(N))) == xsum(
                            y[i*M+m] for i in range(1, N))

            # flow conservation (out - in = y_im)
            for i in range(1, N):
                MD += (xsum(u[j*N*M+i*M+m] for j in range(N)) \
                        - xsum(u[i*N*M+j*M+m] for j in range(N))) == y[i*M+m]

            # capacity constraints, allow flow only if x[i,j,m] == 1
            for i in range(N):
                for j in range(N):
                    MD += u[i*N*M+j*M+m] <= N * x[i*N*M+j*M+m]
                    MD += u[i*N*M+j*M+m] >= 0

        MD.optimize(max_seconds=1000)

        X = np.zeros((N, N, M))
        for m in range(M):
            for i in range(N):
                for j in range(N):
                    X[i,j,m] = x[i*N*M+j*M+m].x
        return X

################################################################################
#
# The PCP Solver with Greeday Single-day Optimal Solution
#
################################################################################
class PCP:
    def __init__(self, method, seed, H=10, read_file=None, n_agents=None):
        self.method = method
        np.random.seed(seed)
        random.seed(seed)

        if read_file is not None:
            self.init_from_file(read_file)
        else:
            n = self.n = np.random.randint(6)*2 + 10 # number of vertices

            # nodes are uniformly sampled within [-10, 10]^2
            self.nodes = (np.random.rand(n, 2) - 0.5) * 2 * 10
            self.nodes[0] = self.nodes[0] * 0.

            # every node has edges to the closest 3~6 nodes
            self.edges = np.zeros((n, n)) + 1000
            for i in range(n):
                vi = self.nodes[i]
                dists = np.sqrt(
                        np.sum(np.square(vi.reshape(1,2) - self.nodes),-1))
                idx = np.argsort(dists)
                topk = np.random.randint(3) + 3
                valid_dists = dists[idx[:topk]]
                self.edges[i, idx[:topk]] = valid_dists
                self.edges[idx[:topk], i] = valid_dists

            # max travel budget
            self.max_length = np.random.rand() * n * 2 + 20

            self.singleday = SingleDayMethod(self.nodes, self.edges)

            if n_agents is None:
                self.n_agents = np.random.randint(4) + 2
            else:
                self.n_agents = n_agents

            # make sure must visit vertices are reachable
            self.n_mustvisits = max(
                    min(np.random.randint(3)+1, self.n_agents), 1)

            self.mask = np.ones((n,))
            self.mask[0] = 0
            avails = []
            for i in range(1, n):
                min_dist = self.singleday.dist[0,i] + self.singleday.dist[i,0]
                if min_dist < self.max_length:
                    avails.append(i)
                else:
                    self.mask[i] = 0.

            self.n_mustvisits = min(self.n_mustvisits, len(avails))
            self.mustvisits = list(
                    sorted(random.sample(avails, self.n_mustvisits)))

            # true parameter mu, hidden from agents
            self.mu = np.random.rand(self.n) * 0.9 + 0.1
            self.mu = self.mu * self.mask

            self.last_visit = np.zeros((self.n))

            self.H = H

            init_estimate = np.clip(self.mu.copy() + \
                    np.random.randn(*self.mu.shape)*0.1, 0, 1) * self.mask

            self.true_cumulative = np.zeros((self.n))
            self.sum_mu = init_estimate

    def init_from_file(self, read_file):
        with open(read_file, 'r') as f:
            lines = f.readlines()

            line1 = lines[0]
            self.n, self.n_agents, self.H, self.max_length = line1.split(" ")
            self.n = int(self.n)
            self.n_agents = int(self.n_agents)
            self.H = int(self.H)
            self.max_length = float(self.max_length)

            line2 = lines[1]
            self.mustvisits = [int(x) for x in line2.split(" ")]
            self.n_mustvisits = len(self.mustvisits)
            self.mask = np.ones((self.n))
            self.mask[0] = 0
            for v in self.mustvisits:
                self.mask[v] = 0

            line3 = lines[2]
            self.mu = line3.split(" ")
            self.mu = np.array([float(x) for x in self.mu])

            node_pos = []
            connect = []
            for i,line in enumerate(lines[3:]):
                x, y = line.split(" ")
                if i < self.n:
                    node_pos.append([float(x), float(y)])
                else:
                    connect.append([int(x), int(y)])

            self.nodes = np.array(node_pos)
            self.edges = np.zeros((self.n, self.n))+1000
            for a,b in connect:
                dist = np.sqrt(
                        np.sum(np.square(self.nodes[a] - self.nodes[b])))
                self.edges[a,b] = dist
                self.edges[b,a] = dist

            self.last_visit = np.zeros((self.n))
            init_estimate = self.mu.copy()*10
                    #np.random.randn(*self.mu.shape)*0.1, 0, 1) * self.mask

            self.true_cumulative = np.zeros((self.n))
            self.sum_mu = init_estimate

            self.singleday = SingleDayMethod(self.nodes, self.edges)
            #import pdb; pdb.set_trace()

    def stats(self):
        save = {
            "N": self.n,
            "V": self.nodes,
            "E": self.edges,
            "L": self.max_length,
            "M": self.n_agents,
            "C": self.mustvisits,
            "mu": self.mu,
            "H": self.H,
        }
        return save

    def simulate(self):
        costs = []
        failures = []
        matrices = []

        for t in range(2, self.H+2):
            # calculate estimated rewards
            self.last_visit += 1
            daily_improve = np.clip(
                (self.mu+np.random.randn(*self.mu.shape)*0.1)*self.mask,
                0, 1)
            self.true_cumulative += daily_improve

            Ti = (t - self.last_visit)
            bar_mu = self.sum_mu / Ti
            cumulative = bar_mu * self.last_visit

            # plan the path
            if "tocp" in self.method:
                X = self.singleday.tocp(
                    0, self.mustvisits,
                    cumulative, self.max_length, self.n_agents)
            elif "top" in self.method:
                X = self.singleday.top(
                    0, self.mustvisits,
                    cumulative, self.max_length, self.n_agents)
            elif "greedy" in self.method:
                X = self.singleday.greedy(
                    0, self.mustvisits,
                    cumulative, self.max_length, self.n_agents)
            else:
                raise NotImplementedError

            # evaluate the collected reward
            fail, visited, reward = self.evaluate_matrix(X, self.true_cumulative)
            cost = self.true_cumulative.sum() - reward

            # udpate estimation
            self.last_visit = self.last_visit * (1 - visited)
            self.sum_mu = self.sum_mu + self.true_cumulative
            self.true_cumulative = self.true_cumulative * (1 - visited)

            # update eval statistics
            failures.append(fail)
            costs.append(cost)
            matrices.append(X.copy())

        return np.array(costs), np.array(failures), matrices

    def evaluate_matrix(self, X, reward):
        edges = self.edges # [n, n]
        length = (X * edges.reshape(self.n, self.n, 1)).sum(0).sum(0)
        under_budget = (length > self.max_length).sum() == 0
        visited = np.sign(X.sum(-1).sum(-1))
        reward = (visited * reward).sum()
        visit_must = True
        for v in self.mustvisits:
            if visited[v] == 0:
                visit_must = False
        fail = (not visit_must) or (not under_budget)
        return fail, visited, reward

    def visualize(self):
        plt.figure()
        n = self.n
        plt.scatter(self.nodes[:,0], self.nodes[:,1], s=5, c='k')
        plt.scatter(self.nodes[0,0], self.nodes[0,1], s=50, c='r')
        mv = np.array(self.mustvisits)
        plt.scatter(self.nodes[mv,0], self.nodes[mv,1], s=30, c='b')
        for i in range(n):
            for j in range(n):
                if self.edges[i,j] < 100:
                    plt.plot([self.nodes[i][0], self.nodes[j][0]],
                             [self.nodes[j][0], self.nodes[j][1]], linestyle='--',
                             color='k', alpha=0.5)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("imgs/tmp.png")


def plot(nodes, edges, X, max_length, mustvisits, method, seed, cost):
    n_agents = X.shape[-1]
    fig, axs = plt.subplots(1, n_agents+1, figsize=(10*(n_agents+1), 10), sharey=True)
    n = nodes.shape[0]
    lw = 3
    for ai, ax in enumerate(axs):
        ax.scatter(nodes[:,0], nodes[:,1], s=200, c='y')
        ax.scatter(nodes[0,0], nodes[0,1], s=200, c='k')
        for x in mustvisits:
            ax.scatter(nodes[x,0], nodes[x,1], s=200, c='r')
        for i in range(n):
            for j in range(n):
                if edges[i,j] < max_length:
                    ax.plot([nodes[i][0], nodes[j][0]],
                            [nodes[i][1], nodes[j][1]],
                            linestyle='--', color='k', alpha=0.1,
                            zorder=0, linewidth=lw)

    colors = {
        'greedy': 'b',
        'top': 'r',
        'tocp': 'g'}
    for m in range(n_agents):
        for i in range(n):
            for j in range(n):
                if i != j:
                    if X[i,j,m] > 0:
                        x = [nodes[i][0], nodes[j][0]]
                        y = [nodes[i][1], nodes[j][1]]
                        axs[m+1].plot(x, y, linestyle='-',
                                color=colors[method], linewidth=lw, zorder=0)

    for ai, ax in enumerate(axs):
        ax.axis('off')
        if ai == 0:
            ax.set_title("Original Map", fontsize=50)
        else:
            ax.set_title(f"Tour of Agent {ai}", fontsize=50) 

    plt.tight_layout()
    plt.text(-30, -10, f"Cost = {cost[0]:.2f}", fontsize=50)
    plt.savefig(f"imgs/{method}-{seed}.png")


def sample(method, seed=-1):
    pcp = PCP(method, seed, H=1, read_file=None, n_agents=3)
    cost, vm, matrices = pcp.simulate()
    plot(pcp.nodes,
         pcp.edges,
         matrices[0],
         pcp.max_length,
         pcp.mustvisits,
         method,
         seed,
         cost)

def main(method, seed=-1, H=10, read_file=None):
    folder = "results"
    t1 = time.time()
    pcp = PCP(method, seed, H, read_file)
    cost, fail, _ = pcp.simulate()
    stat = pcp.stats()
    stat['seed'] = seed
    stat['cost'] = cost 
    stat['fail'] = fail
    t2 = time.time()
    stat['time'] = t2 - t1
    print(f"{seed}: {cost.mean()}")
    if read_file is None:
        with open(f"{folder}/{method}-H{H}-s{seed}.npy", "wb") as f:
            np.save(f,stat)
        f.close()
    else:
        with open(f"{folder}/{read_file}.npy", "wb") as f:
            np.save(f,stat)
        f.close()

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
