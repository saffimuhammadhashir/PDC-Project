using Yao
using CSV
using DataFrames

# Generate a random quantum circuit
function generate_random_circuit(n_qubits::Int, depth::Int)
    circuit = chain(n_qubits)
    for _ in 1:depth
        gate = rand(["H", "X", "Y", "Z", "CNOT", "RX", "RZ"])
        q1 = rand(1:n_qubits)
        q2 = q1
        while q2 == q1 && gate == "CNOT"
            q2 = rand(1:n_qubits)
        end

        if gate == "H"
            push!(circuit, put(n_qubits, q1 => Yao.H))
        elseif gate == "X"
            push!(circuit, put(n_qubits, q1 => Yao.X))
        elseif gate == "Y"
            push!(circuit, put(n_qubits, q1 => Yao.Y))
        elseif gate == "Z"
            push!(circuit, put(n_qubits, q1 => Yao.Z))
        elseif gate == "CNOT" && q1 != q2
            push!(circuit, control(n_qubits, q1, q2 => Yao.X))
        elseif gate == "RX"
            push!(circuit, put(n_qubits, q1 => Yao.rot(Yao.X, rand())))
        elseif gate == "RZ"
            push!(circuit, put(n_qubits, q1 => Yao.rot(Yao.Z, rand())))
        end
    end
    return circuit
end

# Extract gate info
function extract_gate_info(block)
    if block isa Yao.PutBlock
        gate_type = string(typeof(block.content)) |> x -> replace(x, "Yao." => "")
        qubit_info = string(block.locs)
    elseif block isa Yao.ControlBlock
        gate_type = "Control(" * string(typeof(block.content)) * ")" |> x -> replace(x, "Yao." => "")
        qubit_info = "ctrl=" * string(block.ctrl_locs) * ", target=" * string(block.locs)
    else
        gate_type = string(typeof(block))
        qubit_info = "unknown"
    end
    return gate_type, qubit_info
end

# Generate CSV dataset and tensor matrices
function create_circuit_dataset(num_circuits::Int, n_qubits::Int, depth::Int; save_path::String = "dataset/")
    isdir(save_path) || mkpath(save_path)
    println("Generating $num_circuits circuits with $n_qubits qubits and depth $depth")

    for i in 1:num_circuits
        println("\n🚧 Generating circuit $i of $num_circuits...")
        circuit = generate_random_circuit(n_qubits, depth)

        # --- Save Gate DataFrame ---
        df = DataFrame(Gate=String[], Qubits=String[])
        for (j, block) in enumerate(circuit.blocks)
            gate_type, qubit_info = extract_gate_info(block)
            push!(df, (gate_type, qubit_info))

            # Progress for gate CSV
            percent = round(j / length(circuit.blocks) * 1000, digits=1)
            print("\r📄 [Circuit CSV] Saving gates... $percent%")
            flush(stdout)
        end
        CSV.write(joinpath(save_path, "circuit_$(i).csv"), df)
        println(" ✅ Done.")

        # --- Save Unitary Matrix ---
        println("🧮 Computing unitary matrix for tensor $i")
        mat_data = mat(circuit)
        mat_real = real.(mat_data)
        mat_imag = imag.(mat_data)

        mat_df = DataFrame(
            Row=Int[],
            Col=Int[],
            Real=Float64[],
            Imag=Float64[],
        )

        total_elements = prod(size(mat_data))
        counter = 0

        for row in 1:size(mat_data, 1), col in 1:size(mat_data, 2)
            push!(mat_df, (row, col, mat_real[row, col], mat_imag[row, col]))
            counter += 1

            # Progress for tensor CSV
            if counter % 10 == 0 || counter == total_elements
                percent = round(counter / total_elements * 100, digits=1)
                print("\r🧾 [Tensor CSV] Writing matrix... $percent%")
                flush(stdout)
            end
        end
        CSV.write(joinpath(save_path, "tensor_$(i).csv"), mat_df)
        println(" ✅ Done.")

        println("✅ Circuit #$i saved successfully.\n")
    end

    println("🎉 Dataset generation complete. Saved to: $save_path")
end

function generate_tensor_graph(num_circuits::Int, n_qubits::Int; save_path::String = "dataset/")
    edges_df = DataFrame(TensorA=Int[], TensorB=Int[], SharedQubits=Int[], Weight=Float64[])

    for i in 1:num_circuits, j in i+1:num_circuits
        shared = rand(0:2)  # Random number of shared qubits (simulate)
        if shared > 0
            weight = rand(0.0:2:10)  # Random edge weight between 0.5 and 1.0
            push!(edges_df, (i-1, j-1, shared, weight))
        end
    end

    CSV.write(joinpath(save_path, "tensor_edges.csv"), edges_df)
    println("✅ Tensor network graph saved to tensor_edges.csv")
end

# Function to calculate the weightage for each node in the tensor network
function calculate_node_weightages(num_circuits::Int; save_path::String = "dataset/")
    node_weightages = DataFrame(TensorID=Int[], NodeID=Int[], Weightage=Float64[])

    for i in 1:num_circuits
        println("\n🚧 Generating node weightages for circuit $i of $num_circuits...")

        # Load the circuit
        circuit = generate_random_circuit(3, 20)  # Example: 11 qubits, depth 50
        tensor_ids = 1:length(circuit.blocks)  # Assign a unique ID for each tensor/block
        
        for (tensor_id, block) in zip(tensor_ids, circuit.blocks)
            # Simple heuristic for weightage: based on number of qubits (size of tensor)
            weightage = length(block.locs)  # This is a simple example; can be made more complex
            
            # Create a row in the DataFrame for each node in the tensor
            for node_id in 1:length(block.locs)
                push!(node_weightages, (tensor_id, node_id, weightage))
            end
        end

        # Save node weightages to a CSV file
        CSV.write(joinpath(save_path, "node_weightages_$(i).csv"), node_weightages)
        println(" ✅ Node weightages for circuit #$i saved.")
    end

    println("🎉 Node weightage generation complete.")
end

# Example usage: generate weightages for each node in the tensor network for 10000 circuits
# # Example usage
create_circuit_dataset(20, 3, 20; save_path="G:/PDC-Proj/Dataset-5")
calculate_node_weightages(20, save_path="G:/PDC-Proj/Dataset-5")
generate_tensor_graph(20, 3; save_path="G:/PDC-Proj/Dataset-5")