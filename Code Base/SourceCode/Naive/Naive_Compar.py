import os
import math
import csv
import time
import numpy as np
import networkx as nx
from collections import defaultdict
import community as community_louvain 
import networkx as nx
from collections import defaultdict
import random
import networkx as nx
from sklearn.cluster import KMeans
import numpy as np

random.seed(42)  # Set the seed for deterministic results

EDGE_FILE = "tensor_edges.csv"

class Tensor:
    def __init__(self, tensor_id):
        self.id = tensor_id
        self.rank = 0
        self.node_ids = []
        self.node_weights = []
        self.node_matrices = []
        self.tensor_matrix = None
        self.community_id = None

    def compute_node_matrix(self, weight):
        size = 2
        matrix = np.zeros((size, size))
        np.fill_diagonal(matrix, weight * math.sqrt(self.rank))
        return matrix

    def build_tensor_matrix(self):
        # Contract all node matrices (sum in this case)
        if not self.node_matrices:
            return None
        self.tensor_matrix = sum(self.node_matrices)
        return self.tensor_matrix

class Community:
    def __init__(self):
        self.tensors = []
        self.contracted_matrix = None

class TensorNetwork:
    def __init__(self):
        self.tensors = []
        self.tensor_map = {}
        self.communities = []
        self.G = nx.Graph()

    def load_tensor(self, folder_path, i):
        tensor = Tensor(i)
        tensor_path = os.path.join(folder_path, f"tensor_{i}.csv")
        node_path = os.path.join(folder_path, f"node_weightages_{i}.csv")

        if not os.path.exists(tensor_path) or not os.path.exists(node_path):
            return None

        # Estimate rank from tensor file
        max_index = 0
        with open(tensor_path, newline='') as f:
            next(f)
            for row in f:
                r, c, *_ = map(float, row.strip().split(','))
                max_index = max(max_index, int(r), int(c))
        tensor.rank = int(math.ceil(math.log2(max_index + 1)))


        with open(node_path, newline='') as f:
            next(f)
            for row in f:
                _, node_id, weight = row.strip().split(',')
                node_id = int(node_id)
                weight = float(weight)
                tensor.node_ids.append(node_id)
                tensor.node_weights.append(weight)
                tensor.node_matrices.append(tensor.compute_node_matrix(weight))

        tensor.build_tensor_matrix()
        self.tensors.append(tensor)
        self.tensor_map[tensor.id] = tensor
        return tensor

    def load_all_tensors(self, folder_path):
        i = 1
        while True:
            tensor = self.load_tensor(folder_path, i)
            if tensor is None:
                break
            print(f" Loaded tensor_{i}.csv")
            i += 1

    def load_tensor_edges(self, path):
        with open(path, newline='') as file:
            reader = csv.reader(file)
            next(reader, None)
            for row in reader:
                if len(row) < 4:
                    continue
                try:
                    a, b = int(row[0]), int(row[1])
                    shared = int(row[2])
                    weight = float(row[3])
                except ValueError:
                    continue
            
                self.G.add_edge(a, b, shared=shared, weight=weight, combined=shared * weight)

    def detect_communities(self, resolution=1.0, n_clusters=None):

            laplacian_matrix = nx.laplacian_matrix(self.G).toarray()

           
            kmeans = KMeans(n_clusters=n_clusters, random_state=42) 
            kmeans.fit(laplacian_matrix)

           
            partition_dict = {node: label for node, label in zip(self.G.nodes(), kmeans.labels_)}

 
            partition = defaultdict(list)
            for node, cid in partition_dict.items():
                partition[cid].append(node)

           
            self.communities = [Community() for _ in partition]
            for cid, nodes in partition.items():
                for tid in nodes:
                    if tid in self.tensor_map:
                        tensor = self.tensor_map[tid]
                        tensor.community_id = cid
                        self.communities[cid].tensors.append(tensor)

            print(f" Formed {len(self.communities)} deterministic communities with resolution={resolution}.")



    # def compute_tensor_matrix(self, tensor):
    #     """
    #     Create the full tensor matrix for a given tensor using its node matrices.
    #     This version assumes a simplified structure where individual node matrices
    #     are combined via outer product or diagonal sum.
    #     """
    #     size = 2  # Each node matrix is 2x2
    #     tensor_size = size  # Start with one node, grow by Kronecker product

    #     if not tensor.node_matrices:
    #         return np.eye(2)

    #     matrix = tensor.node_matrices[0]
    #     for i in range(1, len(tensor.node_matrices)):
    #         matrix = np.kron(matrix, tensor.node_matrices[i])

    #     return matrix



    def compute_tensor_matrix(self, tensor, out_dim=16):
        """
        Compute a fixed-size matrix approximation (out_dim x out_dim) for a tensor.
        Prevents shape mismatch in downstream matrix contractions.
        """
        matrix = np.zeros((out_dim, out_dim))

        for mat in tensor.node_matrices:
            vec = mat.flatten()
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            vec_small = vec[:out_dim] if len(vec) >= out_dim else np.pad(vec, (0, out_dim - len(vec)))
            outer = np.outer(vec_small, vec_small)
            energy = np.linalg.norm(mat)
            matrix += energy * outer

        return matrix



    def contract_community(self, comm):
        if len(comm.tensors) == 1:
            t = comm.tensors[0]
            t.tensor_matrix = self.compute_tensor_matrix(t)
            comm.contracted_matrix = t.tensor_matrix
            return

       
        for t in comm.tensors:
            t.tensor_matrix = self.compute_tensor_matrix(t)

  
        tensor_ids = [t.id for t in comm.tensors]
        subgraph = self.G.subgraph(tensor_ids).copy()

    
        for u, v, data in subgraph.edges(data=True):
            shared_qubits = data.get("SharedQubits", 1)
            weight = data.get("weight", 1.0)
            combined = shared_qubits * weight
            subgraph[u][v]["combined"] = combined

   
        centrality = nx.betweenness_centrality(subgraph, weight='combined')
        ranked = sorted(comm.tensors, key=lambda t: -centrality[t.id])

        contracted = None
        used = set()

        while ranked:
            A = ranked.pop(0)
            if A.id in used:
                continue

            best_B = None
            best_weight = -1
            for B in ranked:
                if B.id in used:
                    continue
                if subgraph.has_edge(A.id, B.id):
                    w = subgraph[A.id][B.id]["combined"]
                    if w > best_weight:
                        best_weight = w
                        best_B = B

            if best_B:
                mat_A = A.tensor_matrix
                mat_B = best_B.tensor_matrix
                result = np.dot(mat_A, mat_B) 

                contracted = result if contracted is None else np.dot(contracted, result)
                used.update([A.id, best_B.id])
            else:
                # No suitable pair, just add this matrix in
                mat_A = A.tensor_matrix
                contracted = mat_A if contracted is None else np.dot(contracted, mat_A)
                used.add(A.id)

        comm.contracted_matrix = contracted
        comm.effective_rank = sum(t.rank for t in comm.tensors)
        comm.unique_nodes = len({nid for t in comm.tensors for nid in t.node_ids})




    def contract_final(self):


        final = None
        total_weight = 0.0

        for comm in self.communities:
            mat = comm.contracted_matrix
            if mat is None:
                print(" Skipping community with None matrix")
                continue

          
            if final is None:
                MATRIX_DIM = mat.shape[0]
                final = np.zeros((MATRIX_DIM, MATRIX_DIM))

            if mat.shape != final.shape:
                print(f" Skipping community with invalid matrix shape: {mat.shape}")
                continue

   
            norm = np.linalg.norm(mat, ord='fro')
            if norm == 0:
                print(f" Skipping zero matrix for community")
                continue
            mat_normalized = mat / norm

   
            weight = len(comm.tensors) + 1e-6
            total_weight += weight


            final += mat_normalized * weight

        if final is not None and total_weight > 0:
            final /= total_weight
            print(" Final contracted matrix:")
            print(final)
        else:
            print(" No valid community matrices to contract.")








def main():
    folder = "G:\\PDC-Proj\\Dataset-5"
    edge_path = os.path.join(folder, EDGE_FILE)

    net = TensorNetwork()
    community_times = []


    total_start = time.time()

    print("Loading tensors...")
    t0 = time.time()
    net.load_all_tensors(folder)
    t1 = time.time()
    tensor_count = len(net.tensors)
    print(f" Loaded {tensor_count} tensors in {t1 - t0:.4f} seconds")

    print("🔗 Loading tensor edges...")
    t2 = time.time()
    net.load_tensor_edges(edge_path)
    t3 = time.time()
    edge_count = net.G.number_of_edges()
    print(f" Loaded {edge_count} edges in {t3 - t2:.4f} seconds")

    print(" Detecting communities...")
    t4 = time.time()
    net.detect_communities(n_clusters=8)
    t5 = time.time()
    community_count = len(net.communities)
    print(f" Detected {community_count} communities in {t5 - t4:.4f} seconds")

    print(" Contracting communities...")
    t6 = time.time()
    total_contract_time = 0.0
    matrix_shapes = []


    for i, comm in enumerate(net.communities):
        print(f"\n Contracting Community {i} with {len(comm.tensors)} tensors...")
        c_start = time.time()
        net.contract_community(comm)
        c_end = time.time()
        contract_duration = c_end - c_start
        total_contract_time += contract_duration
        community_times.append((i, len(comm.tensors), contract_duration))
        if comm.contracted_matrix is not None:
            matrix_shapes.append(comm.contracted_matrix.shape)
        print(f"    Time: {contract_duration:.4f} seconds")
    t7 = time.time()



    print("\n Contracting final matrix...")
    t8 = time.time()
    final_matrix = net.contract_final()
    t9 = time.time()

    total_end = time.time()

    #  Summary Report Card
    print("\n ========= Report Card =========")
    print(f" Tensors loaded: {tensor_count}")
    print(f" Edges loaded: {edge_count}")
    print(f" Communities detected: {community_count}")
    avg_tensors_per_community = tensor_count / community_count if community_count else 0
    print(f" Avg tensors per community: {avg_tensors_per_community:.2f}")
    if matrix_shapes:
        print(f" Community matrix sizes: {set(matrix_shapes)}")
    else:
        print(" No valid community matrices produced.")

    print(f"\n Timing Breakdown:")
    print(f"   - Tensor loading time: {t1 - t0:.4f}s")
    print(f"   - Edge loading time: {t3 - t2:.4f}s")
    print(f"   - Community detection time: {t5 - t4:.4f}s")
    print(f"   - Community contraction time: {t7 - t6:.4f}s")
    print(f"   - Final contraction time: {t9 - t8:.4f}s")
    print(f"   - Total pipeline time: {total_end - total_start:.4f} seconds")

    print("\n Per-Community Contraction Times:")
    for i, size, dur in community_times:
        print(f"   - Community {i:2d} | Tensors: {size:2d} | Time: {dur:.4f}s")

    if final_matrix is not None:
        final_norm = np.linalg.norm(final_matrix, ord='fro')
        print(f"\n Final Matrix Frobenius Norm: {final_norm:.4f}")
    print(" ================================")


if __name__ == "__main__":
    main()
