#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <time.h>
#include <math.h>
#include <metis.h>
#include <unistd.h>  // for sysconf

#define LINE_SIZE 256
#define MAX_PATH 512  // or any size you find appropriate

#define EDGE_FILE "tensor_edges.csv"  // Path to the edge CSV

int tensorcount=0;

typedef struct {
    int id;
    int rank;
    int node_count;
    int* node_ids;
    float* node_weights;
    int community_id;  // Set during METIS partitioning

    float** node_matrices;  // Dynamically allocated array for matrices at each node
} Tensor;

typedef struct {
    Tensor** tensors;                      // Dynamically allocated array of tensor pointers
    int tensor_count;
    int effective_rank;                    // Store computed effective rank
    int unique_nodes;                      // Store number of unique nodes
    float* contracted_matrix;

} Community;

typedef struct {
    Tensor* all_tensors;                    // Dynamically allocated array of all tensors
    int total_tensors;
    Community* communities;                 // Dynamically allocated array of communities
    int num_communities;
    int** edges;                           // Dynamically allocated connectivity matrix
    float** weights;                       // Dynamically allocated edge weights
} TensorNetwork;

int load_all_tensors_from_folder(TensorNetwork* net, const char* folder_path);

int count_files_starting_with_tensor(const char* folder_path) {
    DIR* dir = opendir(folder_path);
    if (!dir) {
        perror("Could not open directory");
        return -1;
    }

    struct dirent* entry;
    int count = 0;

    // Count files starting with "tensor_"
    while ((entry = readdir(dir)) != NULL) {
        if (strncmp(entry->d_name, "tensor_", 7) == 0 && strstr(entry->d_name, ".csv")) {
            count++;
        }
    }

    closedir(dir);
    return count;
}


// --- Load tensor edges from a CSV file
int load_tensor_edges(const char* path, TensorNetwork* net) {
    FILE* file = fopen(path, "r");
    if (!file) {
        perror("Failed to open tensor edges CSV");
        return -1;
    }


    char line[LINE_SIZE];
    fgets(line, LINE_SIZE, file); // skip header

    int tensorA, tensorB, sharedQubits;
    float weight;

    while (fgets(line, LINE_SIZE, file)) {
        if (sscanf(line, "%d,%d,%d,%f", &tensorA, &tensorB, &sharedQubits, &weight) == 4) {
            net->edges[tensorA][tensorB] = 1;  // Mark the connection
            net->edges[tensorB][tensorA] = 1;  // Symmetric connection
            net->weights[tensorA][tensorB] = weight;
            net->weights[tensorB][tensorA] = weight;
        }
    }

    fclose(file);
    return 0;
}

// --- Load node weightages for a specific tensor
int load_node_weightages(const char* filepath, Tensor* tensor) {
    FILE* file = fopen(filepath, "r");
    if (!file) {
        fprintf(stderr, "Warning: Could not open node weightage file: %s\n", filepath);
        return -1;
    }

    char line[LINE_SIZE];
    fgets(line, LINE_SIZE, file); // Skip header

    int tensorId, nodeId;
    float weightage;
    tensor->node_count = 0;

    // Initial allocation with some buffer for nodes
    tensor->node_ids = malloc(10 * sizeof(int));
    tensor->node_weights = malloc(10 * sizeof(float));
    tensor->node_matrices = malloc(10 * sizeof(float*)); // For node matrices

    if (!tensor->node_ids || !tensor->node_weights || !tensor->node_matrices) {
        fprintf(stderr, "Memory allocation failed for node arrays\n");
        fclose(file);
        return -1;
    }

    while (fgets(line, LINE_SIZE, file)) {
        if (sscanf(line, "%d,%d,%f", &tensorId, &nodeId, &weightage) == 3) {
            // Resize arrays when node_count reaches a multiple of 10
            if (tensor->node_count % 10 == 0) {
                tensor->node_ids = realloc(tensor->node_ids, (tensor->node_count + 10) * sizeof(int));
                tensor->node_weights = realloc(tensor->node_weights, (tensor->node_count + 10) * sizeof(float));
                tensor->node_matrices = realloc(tensor->node_matrices, (tensor->node_count + 10) * sizeof(float*));

                if (!tensor->node_ids || !tensor->node_weights || !tensor->node_matrices) {
                    fprintf(stderr, "Memory reallocation failed\n");
                    fclose(file);
                    return -1;
                }
            }

            tensor->node_ids[tensor->node_count] = nodeId;
            tensor->node_weights[tensor->node_count] = weightage;
            tensor->node_count++;

            // Allocate matrix for each node
            tensor->node_matrices[tensor->node_count - 1] = malloc(2 * 2 * sizeof(float)); // 2x2 matrix for simplicity
        }
    }

    fclose(file);
    return 0;
}



// Helper function to determine if two tensors share nodes
int share_nodes(Tensor* t1, Tensor* t2) {
    for (int i = 0; i < t1->node_count; i++) {
        for (int j = 0; j < t2->node_count; j++) {
            if (t1->node_ids[i] == t2->node_ids[j]) {
                return 1; // Found a shared node
            }
        }
    }
    return 0; // No shared nodes
}








int load_tensor_from_csv(const char* filepath, Tensor* t, int id) {
    FILE* file = fopen(filepath, "r");
    if (!file) {
        perror("Failed to open tensor CSV");
        return -1;
    }

    char line[LINE_SIZE];
    fgets(line, LINE_SIZE, file); // skip header

    int max_row = 0, max_col = 0;
    while (fgets(line, LINE_SIZE, file)) {
        int row, col;
        float real, imag;
        if (sscanf(line, "%d,%d,%f,%f", &row, &col, &real, &imag) == 4) {
            if (row > max_row) max_row = row;
            if (col > max_col) max_col = col;
        }
    }
    fclose(file);

    int dimension = (max_row > max_col ? max_row : max_col);
    int rank = (int)log2(dimension);
    if (rank <= 0 || rank > 30) {
        fprintf(stderr, "❌ Invalid tensor dimension in %s\n", filepath);
        return -1;
    }

    t->id = id;
    t->rank = rank;
    t->node_count = 0;  // Will be filled by load_node_weightages
    

    return 0;
}




int load_all_tensors_from_folder(TensorNetwork* net, const char* folder_path) {
    DIR* dir = opendir(folder_path);
    if (!dir) {
        perror("Could not open directory");
        return -1;
    }

    // Count files starting with "tensor_" in the directory to determine how many tensors are there
    tensorcount = count_files_starting_with_tensor(folder_path);
    net->all_tensors = malloc(tensorcount * sizeof(Tensor));  // Allocate space for tensors
    net->edges = (int**)malloc(tensorcount * sizeof(int*));    // Allocate an array of pointers to int for edges
    net->weights = (float**)malloc(tensorcount * sizeof(float*));  // Allocate an array of pointers to float for weights

    // Initialize edges and weights
    for (int i = 0; i < tensorcount; i++) {
        net->edges[i] = (int*)malloc(tensorcount * sizeof(int));   // Initialize edges for each tensor
        net->weights[i] = (float*)malloc(tensorcount * sizeof(float));  // Initialize weights for each tensor

        for (int j = 0; j < tensorcount; j++) {
            net->edges[i][j] = 0;           // Initialize the edges to 0
            net->weights[i][j] = 0.0f;      // Initialize the weights to 0.0f
        }
    }

    struct dirent* entry;
    int id = 0;
    char filepath[MAX_PATH];

    // Count total nodes in the system
    int total_nodes = 0;

    // Load tensors and their corresponding node weightages in sequence
    for (int i = 1; i <= tensorcount; i++) {
        // Generate file paths for tensor and node weightages
        snprintf(filepath, MAX_PATH, "%s/tensor_%d.csv", folder_path, i);  // Tensor file
        char weightage_filepath[MAX_PATH];
        snprintf(weightage_filepath, MAX_PATH, "%s/node_weightages_%d.csv", folder_path, i);  // Node weightages file

        // Check if both the tensor file and node weightages file exist
        FILE* tensor_file = fopen(filepath, "r");
        FILE* weightage_file = fopen(weightage_filepath, "r");

        if (tensor_file && weightage_file) {
            // Successfully opened the files, now load the tensor and its corresponding node weightages
            if (load_tensor_from_csv(filepath, &net->all_tensors[id], id) == 0) {
                // Load node weightages for the tensor
                load_node_weightages(weightage_filepath, &net->all_tensors[id]);

                // Count nodes
                total_nodes += net->all_tensors[id].node_count;

                printf("📄 Tensor %d from %s processed completely\n", net->all_tensors[id].id, filepath);
                id++;
            }
        } else {
            if (!tensor_file) {
                fprintf(stderr, "❌ Could not open tensor file: %s\n", filepath);
            }
            if (!weightage_file) {
                fprintf(stderr, "❌ Could not open weightage file: %s\n", weightage_filepath);
            }
        }

        // Close the files after processing
        if (tensor_file) fclose(tensor_file);
        if (weightage_file) fclose(weightage_file);
    }

    closedir(dir);
    net->total_tensors = id;

    // Print the total number of nodes after loading data
    printf("📊 Total nodes in the system: %d\n", total_nodes);

    return id > 0 ? 0 : -1;  // Return error if no tensors loaded
}


void compute_node_matrix(float* matrix, int size, float weight, int rank) {
    // Example: scale identity matrix by weight * sqrt(rank)
    for (int i = 0; i < size * size; i++) {
        matrix[i] = 0.0f;
    }
    for (int i = 0; i < size; i++) {
        matrix[i * size + i] = weight * sqrtf(rank);
    }
}


float* contract_tensor_node_graph(Tensor* t, int size) {
    float* result = calloc(size * size, sizeof(float));
    for (int i = 0; i < t->node_count; i++) {
        for (int j = 0; j < size * size; j++) {
            result[j] += t->node_matrices[i][j];  // Simple sum or use another contraction rule
        }
    }
    return result;
}

float compute_tensor_centrality(TensorNetwork* net, Tensor* t) {
    float sum = 0.0f;
    for (int i = 0; i < net->total_tensors; i++) {
        sum += net->weights[t->id][i];
    }
    return sum;
}


void free_tensor(Tensor* t) {
    if (!t) return;
    for (int i = 0; i < t->node_count; i++) {
        free(t->node_matrices[i]);
    }
    free(t->node_matrices);
    free(t->node_ids);
    free(t->node_weights);
}





// --- STEP 1: Community Detection using METIS
void detect_communities(TensorNetwork* net) {
    idx_t n = net->total_tensors;
    if (n < 2) {
        fprintf(stderr, "❌ Not enough tensors to partition.\n");
        return;
    }

    // --- Compute average tensor rank
    int total_rank = 0;
    for (int i = 0; i < n; ++i) {
        total_rank += net->all_tensors[i].rank;
    }
    float avg_rank = total_rank / (float)n;

    // --- Determine num_parts based on tensor count and average rank
    idx_t max_parts = sysconf(_SC_NPROCESSORS_ONLN); // number of cores
    idx_t num_parts = fmin(n / (int)(avg_rank + 1), max_parts);
    if (num_parts < 2) num_parts = 2;

    net->num_communities = num_parts;

    printf("🔧 Auto-selected %d communities (avg tensor rank: %.2f, cores: %d)\n",
        num_parts, avg_rank, max_parts);
 

    // --- Build graph with weights
    idx_t* xadj = malloc((n + 1) * sizeof(idx_t));
    idx_t* adjncy = malloc(n * (n - 1) * sizeof(idx_t)); // worst case
    idx_t* adjwgt = malloc(n * (n - 1) * sizeof(idx_t)); // edge weights

    idx_t edge_index = 0;
    for (idx_t i = 0; i < n; ++i) {
        xadj[i] = edge_index;
        for (idx_t j = 0; j < n; ++j) {
            if (i != j && net->edges[i][j] == 1) {
                adjncy[edge_index] = j;
                adjwgt[edge_index] = (idx_t)(net->weights[i][j] * 1000);  // Scaling the weight for METIS
                edge_index++;
            }
        }
    }
    xadj[n] = edge_index;

    idx_t* part = malloc(n * sizeof(idx_t));
    idx_t ncon = 1, objval;

    int result = METIS_PartGraphKway(&n, &ncon, xadj, adjncy,
                                     NULL, NULL, NULL,
                                     &num_parts,
                                     NULL, NULL, NULL,
                                     &objval, part);

    if (result != METIS_OK) {
        fprintf(stderr, "❌ METIS partitioning failed.\n");
        free(xadj); free(adjncy); free(adjwgt); free(part);
        return;
    }

    // --- Assign communities
    net->communities = malloc(num_parts * sizeof(Community));
    for (int i = 0; i < num_parts; ++i) {
        net->communities[i].tensors = malloc(10 * sizeof(Tensor*));
        net->communities[i].tensor_count = 0;


    }

    // Assign tensors to communities based on METIS partitioning result
    for (int i = 0; i < n; ++i) {
        int cid = part[i];
        Community* comm = &net->communities[cid];
        net->all_tensors[i].community_id = cid;  // Store community ID
        comm->tensors[comm->tensor_count++] = &net->all_tensors[i];

        if (comm->tensor_count % 10 == 0) {
            comm->tensors = realloc(comm->tensors, (comm->tensor_count + 10) * sizeof(Tensor*));
        }
    }

    printf("✅ METIS community detection complete. %d communities.\n", net->num_communities);

    free(xadj); free(adjncy); free(adjwgt); free(part);
}


// --- STEP 2: Intra-community contraction with sophisticated node-by-node logic
// Define a structure to hold each tensor and its corresponding centrality
typedef struct {
    Tensor* tensor;
    float centrality;
} TensorCentrality;

// Helper function for sorting based on centrality (descending order)
int compare_centralities(const void* a, const void* b) {
    float centrality_a = ((TensorCentrality*)a)->centrality;
    float centrality_b = ((TensorCentrality*)b)->centrality;
    return (centrality_b > centrality_a) - (centrality_a > centrality_b); // Descending order
}




// Function to perform contraction between two tensors A and B into a larger matrix C
float* contract_two_tensors(Tensor* A, Tensor* B, int size) {
    // Assume we are contracting over the shared indices (e.g., node_ids)
    int shared_idx = -1;
    for (int i = 0; i < A->node_count; i++) {
        for (int j = 0; j < B->node_count; j++) {
            if (A->node_ids[i] == B->node_ids[j]) {
                shared_idx = A->node_ids[i];
                break;
            }
        }
        if (shared_idx != -1) break;  // Found shared index
    }

    if (shared_idx == -1) {
        printf("No shared node found for contraction between tensors A and B.\n");
        return NULL;
    }

    // Assuming a simple matrix multiplication contraction rule for now
    // Allocate memory for the resultant matrix (contracted tensor)
    float* result_matrix = calloc(size * size, sizeof(float));

    // Iterate over each node and perform contraction (matrix multiplication)
    for (int i = 0; i < A->node_count; i++) {
        if (A->node_ids[i] == shared_idx) {
            for (int j = 0; j < B->node_count; j++) {
                if (B->node_ids[j] == shared_idx) {
                    // Multiply the corresponding node weights and add to the result matrix
                    float weight_A = A->node_weights[i];
                    float weight_B = B->node_weights[j];
                    for (int m = 0; m < size; m++) {
                        for (int n = 0; n < size; n++) {
                            result_matrix[m * size + n] += weight_A * weight_B * A->node_matrices[i][m * size + n] * B->node_matrices[j][n];
                        }
                    }
                }
            }
        }
    }

    return result_matrix;  // Return the contracted matrix
}

// Intra-community contraction function, now using two-tensor contraction logic
void contract_community(TensorNetwork* net, Community* comm, int cid) {
    printf("Contracting Community %d with %d tensors...\n", cid, comm->tensor_count);
    fflush(stdout);

    // Track which nodes have been processed (to avoid double counting)
    int* processed_nodes = calloc(comm->tensor_count * 100, sizeof(int));
    int num_processed = 0;

    // Track the effective rank contribution of each node
    float total_effective_rank = 0.0f;

    // Create an array of TensorCentrality to hold tensor pointers and their centralities
    TensorCentrality* tensor_centralities = malloc(comm->tensor_count * sizeof(TensorCentrality));

    // Populate the tensor_centralities array with tensors and their centrality values
    for (int i = 0; i < comm->tensor_count; i++) {
        tensor_centralities[i].tensor = comm->tensors[i];
        tensor_centralities[i].centrality = compute_tensor_centrality(net, comm->tensors[i]);
    }

    // Sort the tensor_centralities array based on centrality
    qsort(tensor_centralities, comm->tensor_count, sizeof(TensorCentrality), compare_centralities);

    // Contract tensors pair by pair within the community
    float* contracted_matrix = NULL;
    for (int i = 0; i < comm->tensor_count - 1; i++) {
        for (int j = i + 1; j < comm->tensor_count; j++) {
            Tensor* A = tensor_centralities[i].tensor;
            Tensor* B = tensor_centralities[j].tensor;

            // Contract the tensors into a larger matrix
            float* result_matrix = contract_two_tensors(A, B, 2);  // Assuming 2x2 matrices

            if (result_matrix != NULL) {
                if (contracted_matrix == NULL) {
                    contracted_matrix = result_matrix;  // First contraction result
                } else {
                    // Add the result of the current contraction to the previous contracted matrix
                    for (int k = 0; k < 4; k++) {  // Assuming 2x2 matrices
                        contracted_matrix[k] += result_matrix[k];
                    }
                    free(result_matrix);  // Free the current matrix after adding
                }
            }
        }
    }

    // Calculate effective rank for the contracted community tensor
    int effective_rank = (int)ceil(total_effective_rank);

    printf("➡️ Resultant Tensor from Community %d has effective rank %d (%.2f) from %d unique nodes\n",
           cid, effective_rank, total_effective_rank, num_processed);
    fflush(stdout);

    comm->effective_rank = effective_rank;
    comm->unique_nodes = num_processed;

    comm->contracted_matrix = contracted_matrix;  // Store the contracted matrix

    free(processed_nodes);
    free(tensor_centralities);
}








// --- STEP 3: Inter-community contraction

void contract_final_stage(TensorNetwork* net) {
    printf("Final contraction across %d community tensors...\n", net->num_communities);
    fflush(stdout);

    int size = 2;  // Assuming 2x2 matrices for contraction
    float** community_couplings = malloc(net->num_communities * sizeof(float*));
    for (int i = 0; i < net->num_communities; i++) {
        community_couplings[i] = calloc(net->num_communities, sizeof(float));
    }

    int total_steps = net->num_communities * (net->num_communities - 1) / 2;
    int current_step = 0;

    for (int i = 0; i < net->num_communities; ++i) {
        for (int j = i + 1; j < net->num_communities; ++j) {
            for (int t1 = 0; t1 < net->communities[i].tensor_count; ++t1) {
                Tensor* tensor1 = net->communities[i].tensors[t1];
                for (int t2 = 0; t2 < net->communities[j].tensor_count; ++t2) {
                    Tensor* tensor2 = net->communities[j].tensors[t2];
                    if (net->edges[tensor1->id][tensor2->id]) {
                        float edge_weight = net->weights[tensor1->id][tensor2->id];
                        for (int n1 = 0; n1 < tensor1->node_count; ++n1) {
                            int node1 = tensor1->node_ids[n1];
                            float weight1 = tensor1->node_weights[n1];
                            for (int n2 = 0; n2 < tensor2->node_count; ++n2) {
                                if (tensor2->node_ids[n2] == node1) {
                                    float weight2 = tensor2->node_weights[n2];
                                    if (edge_weight >= 0 && weight1 >= 0 && weight2 >= 0 &&
                                        tensor1->rank > 0 && tensor2->rank > 0) {
                                        float coupling = edge_weight * weight1 * weight2 *
                                                         sqrtf(tensor1->rank * tensor2->rank);
                                        community_couplings[i][j] += coupling;
                                        community_couplings[j][i] += coupling;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            current_step++;
            if (total_steps != 0 && current_step % (total_steps / 10 + 1) == 0) {
                int percentage = (current_step * 100) / total_steps;
                printf("\rProgress: [%d%%]", percentage);
                fflush(stdout);
            }
        }
    }

    printf("\rProgress: [100%%] - Contraction complete!\n");
    fflush(stdout);

    float total_effective_rank = 0.0f;
    int total_unique_nodes = 0;

    for (int i = 0; i < net->num_communities; ++i) {
        Community* comm = &net->communities[i];
        total_effective_rank += comm->effective_rank;
        total_unique_nodes += comm->unique_nodes;

        float coupling_sum = 0.0f;
        for (int j = 0; j < net->num_communities; ++j) {
            if (i != j) coupling_sum += community_couplings[i][j];
        }
        total_effective_rank += coupling_sum * 0.5f;
    }

    float scaling_factor = 1.0f / (1.0f + log10f(1 + net->num_communities));
    int final_rank = (int)ceilf(total_effective_rank * scaling_factor);

    printf("✅ Final contracted tensor has effective rank %d (%.2f) with scaling factor %.2f\n",
           final_rank, total_effective_rank, scaling_factor);
    printf("   Total unique nodes across all communities: %d\n", total_unique_nodes);

    float* final_matrix = calloc(size * size, sizeof(float));
    for (int i = 0; i < net->num_communities; i++) {
        for (int j = 0; j < size * size; j++) {
            final_matrix[j] += net->communities[i].contracted_matrix[j];
        }
    }

    printf("\n🧮 Final contracted matrix:\n");
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%8.3f ", final_matrix[i * size + j]);
        }
        printf("\n");
    }

    for (int i = 0; i < net->num_communities; i++) {
        free(community_couplings[i]);
    }
    free(community_couplings);
    free(final_matrix);
}


// --- MAIN
int main() {
    TensorNetwork net;
    memset(&net, 0, sizeof(TensorNetwork));
    
    // Get folder path from arguments or use default
    const char* folder_path = "/home/hash/PDC-Project/Dataset";
    
    // Construct full path to edge file
    char edge_path[MAX_PATH];
    snprintf(edge_path, MAX_PATH, "%s/%s", folder_path, EDGE_FILE);
    
    printf("🔍 Loading tensors from: %s\n", folder_path);
    printf("🔍 Edge file: %s\n", edge_path);

    if (load_all_tensors_from_folder(&net, folder_path) != 0) {
        fprintf(stderr, "❌ Failed to load tensors from folder\n");
        return 1;
    }

    // Load tensor edges (connections and weights)
    if (load_tensor_edges(edge_path, &net) != 0) {
        fprintf(stderr, "❌ Failed to load tensor edges\n");
        return 1;
    }

    printf("📄 Loaded %d tensors from folder\n\n", net.total_tensors);

    clock_t netstart = clock();
    clock_t start, end;
    double time_spent1;
    double time_spent2;
    double time_spent3;
    double time_spent4;

    start = clock();
    // Step 1: Detect communities
    detect_communities(&net);
    end = clock();
    time_spent1 = (double)(end - start) / CLOCKS_PER_SEC;
    printf("⏱️ Time for community detection: %.6f seconds\n\n", time_spent1);

    start = clock();
    // Step 2: Intra-community contraction
    for (int i = 0; i < net.num_communities; ++i) {
        contract_community(&net, &net.communities[i], i);
    }

    end = clock();
    time_spent2 = (double)(end - start) / CLOCKS_PER_SEC;
    printf("⏱️ Time for intra-community contraction: %.6f seconds\n\n", time_spent2);

    start = clock();
    // Step 3: Final contraction
    contract_final_stage(&net);
    end = clock();
    time_spent3 = (double)(end - start) / CLOCKS_PER_SEC;
    printf("⏱️ Time for final contraction: %.6f seconds\n", time_spent3);
    
    end = clock();
    time_spent4 = (double)(end - netstart) / CLOCKS_PER_SEC;
    printf("========================================================================\n");
    printf("⏱️ Time for community detection: %.6f seconds\n", time_spent1);
    printf("⏱️ Time for intra-community contraction: %.6f seconds\n\n", time_spent2);
    printf("⏱️ Time for final contraction: %.6f seconds\n", time_spent3);
    printf("⏱️ Total Time for ComPar: %.6f seconds\n", time_spent4);

    return 0;
}
