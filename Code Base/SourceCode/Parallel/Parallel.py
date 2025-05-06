import os
import math
import csv
import time
import numpy as np
import networkx as nx
import pyopencl as cl
from mpi4py import MPI
from collections import defaultdict
from sklearn.cluster import KMeans
import numpy as np
import random
random.seed(42) 

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

           
            platforms = cl.get_platforms()
            platform = platforms[0]  
            device = platform.get_devices()[0]  

            self.context = cl.Context([device])  
            self.queue = cl.CommandQueue(self.context)  


            program_source = """
            __kernel void matrix_mult(__global const float* A, __global const float* B, __global float* C, const unsigned int N)
            {
                int i = get_global_id(0);
                int j = get_global_id(1);

                if (i < N && j < N) {
                    float sum = 0.0f;
                    for (int k = 0; k < N; ++k) {
                        sum += A[i * N + k] * B[k * N + j];
                    }
                    C[i * N + j] = sum;
                }
            }
            """

            self.program = cl.Program(self.context, program_source).build()  

    def load_tensor(self, folder_path, i):
        tensor = Tensor(i)
        tensor_path = os.path.join(folder_path, f"tensor_{i}.csv")
        node_path = os.path.join(folder_path, f"node_weightages_{i}.csv")

        if not os.path.exists(tensor_path) or not os.path.exists(node_path):
            return None


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
            print(f"=>> Loaded tensor_{i}.csv")
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

        if n_clusters is None:
            n_clusters = max(2, len(self.G.nodes()) // 2) 

     
        laplacian_matrix = nx.laplacian_matrix(self.G).toarray()


        kmeans = KMeans(n_clusters=n_clusters, random_state=42)  
        kmeans.fit(laplacian_matrix)


        partition_dict = {node: label for node, label in zip(self.G.nodes(), kmeans.labels_)}


        partition = defaultdict(list)
        for node, cid in partition_dict.items():
            partition[cid].append(node)


        self.communities = [Community() for _ in range(len(partition))]

       
        for cid, nodes in partition.items():
            community = self.communities[cid]
            community.id = cid 
            
            for tid in nodes:
                if tid in self.tensor_map:
                    tensor = self.tensor_map[tid]
                    tensor.community_id = cid
                    community.tensors.append(tensor)
                    print(f"Assigned tensor {tid} to community {cid}") 

        for idx, comm in enumerate(self.communities):
            if not comm.tensors:
                print(f"[Warning] Community {idx} has no tensors assigned!")
            else:
                print(f"Community {idx} has {len(comm.tensors)} tensors.")  

        print(f"Formed {len(self.communities)} deterministic communities with resolution={resolution}.")


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



    def compute_tensor_matrix(self, tensor, out_dim=512):

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
        ranked = sorted(comm.tensors, key=lambda t: -centrality.get(t.id, 0))
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


                contracted = self.opencl_matrix_multiply(mat_A, mat_B)
                used.update([A.id, best_B.id])
            else:
                mat_A = A.tensor_matrix
                contracted = mat_A if contracted is None else self.opencl_matrix_multiply(contracted, mat_A)
                used.add(A.id)

        comm.contracted_matrix = contracted
        comm.effective_rank = sum(t.rank for t in comm.tensors)
        comm.unique_nodes = len({nid for t in comm.tensors for nid in t.node_ids})

    def opencl_matrix_multiply(self, mat_A, mat_B):
        N = mat_A.shape[0]
        buffer_A = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=mat_A)
        buffer_B = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=mat_B)
        buffer_C = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, mat_A.nbytes)

        self.program.matrix_mult(self.queue, (N, N), None, buffer_A, buffer_B, buffer_C, np.int32(N))
        result = np.zeros_like(mat_A)
        cl.enqueue_copy(self.queue, result, buffer_C).wait()
        return result


    def contract_final(self):
        final = None
        total_weight = 0.0

        for comm in self.communities:
            mat = comm.contracted_matrix
            if mat is None:
                print("!!  Skipping community with None matrix")
                continue


            if final is None:
                MATRIX_DIM = mat.shape[0]
                final = np.zeros((MATRIX_DIM, MATRIX_DIM))

            if mat.shape != final.shape:
                print(f"!!  Skipping community with invalid matrix shape: {mat.shape}")
                continue

            norm = np.linalg.norm(mat, ord='fro')
            if norm == 0:
                print(f"!!  Skipping zero matrix for community")
                continue
            mat_normalized = mat / norm

            weight = len(comm.tensors) + 1e-6
            total_weight += weight


            final += mat_normalized * weight

        if final is not None and total_weight > 0:
            final /= total_weight
            print("=> Final contracted matrix:")
            print(final)
        else:
            print("X No valid community matrices to contract.")






def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    folder = "G:\\PDC-Proj\\Dataset-5"
    edge_path = os.path.join(folder, EDGE_FILE)
    net = TensorNetwork()

    global_start_time = time.time()

    load_start = time.time()
    if rank == 0:
        net.load_all_tensors(folder)
        load_tensors_duration = time.time() - load_start
        print(f"Rank {rank}: Tensor Loading Duration: {load_tensors_duration:.4f} sec")

        net.load_tensor_edges(edge_path)
        edge_loading_duration = time.time() - load_start - load_tensors_duration
        print(f"Rank {rank}: Edge Loading Duration: {edge_loading_duration:.4f} sec")

        detect_start = time.time()
        net.detect_communities(n_clusters=8)
        community_detection_duration = time.time() - detect_start
        print(f"Rank {rank}: Community Detection Duration: {community_detection_duration:.4f} sec")

        communities = net.communities
    else:
        communities = None
    load_duration = time.time() - load_start

    comm.Barrier()

    bcast_start = time.time()
    communities = comm.bcast(communities, root=0)
    bcast_duration = time.time() - bcast_start
    print(f"Rank {rank}: Broadcast Duration: {bcast_duration:.4f} sec")

    comm.Barrier()

    local_communities = np.array_split(communities, size)[rank]

    local_times = []
    local_matrix_shapes = []

    community_contraction_times = {}


    contraction_start = time.time()
    for community in local_communities:
        start = time.time()
        if hasattr(community, 'id') and hasattr(community, 'tensors'):
            net.contract_community(community)
            duration = time.time() - start
            local_times.append((community.id, len(community.tensors), duration))
            if hasattr(community, 'contracted_matrix') and community.contracted_matrix is not None:
                local_matrix_shapes.append(community.contracted_matrix.shape)
            community_contraction_times[community.id] = duration
        else:
            print(f"[Rank {rank}] Warning: Community {community} lacks 'id' or 'tensors' attributes")

    contraction_duration = time.time() - contraction_start
    print(f"Rank {rank}: Community Contraction Duration: {contraction_duration:.4f} sec")

    comm.Barrier()


    gather_start = time.time()
    all_times = comm.gather(local_times, root=0)
    all_shapes = comm.gather(local_matrix_shapes, root=0)
    contracted_comms = comm.gather(local_communities, root=0)
    gather_duration = time.time() - gather_start
    print(f"Rank {rank}: Gather Duration: {gather_duration:.4f} sec")

    # Final contraction
    final_contraction_start = time.time()
    if rank == 0:
        all_times = [item for sublist in all_times for item in sublist]
        all_shapes = [s for sublist in all_shapes for s in sublist]
        net.communities = [comm for sublist in contracted_comms for comm in sublist]

        final_matrix = net.contract_final()
        final_contraction_duration = time.time() - final_contraction_start
        total_time = time.time() - global_start_time
        print(f"Rank {rank}: Final Contraction Duration: {final_contraction_duration:.4f} sec")

        # Report with detailed timing
        print("\n===== COMMUNITY CONTRACTION REPORT =====\n")
        print(f"Total processes used       : {size}")
        print(f"Total time taken           : {total_time:.4f} sec")
        print(f"Initial load time (Rank 0) : {load_duration:.4f} sec")
        print(f"Tensor loading time        : {load_tensors_duration:.4f} sec")
        print(f"Edge loading time          : {edge_loading_duration:.4f} sec")
        print(f"Community detection time   : {community_detection_duration:.4f} sec")
        print(f"Broadcast time             : {bcast_duration:.4f} sec")
        print(f"Data gather time           : {gather_duration:.4f} sec")
        print(f"Final contraction time     : {final_contraction_duration:.4f} sec")
        print(f"Total communities          : {len(all_times)}\n")

        # Per-community contraction times
        print(f"{'CommID':<8} {'#Tensors':<10} {'Time(s)':<10}")
        print("-" * 32)
        # Report per-community contraction times
        print(f"\n Per-Community Contraction Times:")  # Added for clarity
        for comm_id, num_tensors, duration in sorted(all_times, key=lambda x: x[0]):
            print(f"   - Community: {comm_id:<8} |  Tensors: {num_tensors:<10} | Time: {duration:<10.4f}")

        if all_shapes:
            avg_shape = np.mean([np.prod(shape) for shape in all_shapes])
            print(f"\nAverage contracted matrix size: {avg_shape:.2f} elements")
            print(f"Total contracted matrices     : {len(all_shapes)}")

        print("\n===== END OF REPORT =====\n")

if __name__ == "__main__":
    main()
