using Yao
using CSV
using DataFrames

# Generate a random quantum circuit
function generate_random_circuit(n_qubits::Int, depth::Int)
    circuit = chain(n_qubits)

    for _ in 1:depth
        gate = rand(["H", "X", "Y", "Z", "CNOT", "RX", "RZ"])
        q1 = rand(1:n_qubits)
        
        # Make sure q2 is different from q1 for CNOT gates
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

# Function to extract gate details from circuit blocks
function extract_gate_info(block)
    if block isa Yao.PutBlock
        gate_type = string(typeof(block.content))
        gate_type = replace(gate_type, "Yao." => "")  # Remove Yao. prefix for cleaner output
        qubit_info = string(block.locs)
    elseif block isa Yao.ControlBlock
        gate_type = "Control(" * string(typeof(block.content)) * ")"
        gate_type = replace(gate_type, "Yao." => "")  # Remove Yao. prefix
        qubit_info = "ctrl=" * string(block.ctrl_locs) * ", target=" * string(block.locs)
    else
        gate_type = string(typeof(block))
        qubit_info = "unknown"
    end
    
    return gate_type, qubit_info
end

# Save all circuits to individual CSV files
function create_circuit_dataset(num_circuits::Int, n_qubits::Int, depth::Int; save_path::String = "dataset/")
    isdir(save_path) || mkpath(save_path)
    
    println("Generating $num_circuits circuits with $n_qubits qubits and depth $depth")
    
    for i in 1:num_circuits
        circuit = generate_random_circuit(n_qubits, depth)
        df = DataFrame(Gate=String[], Qubits=String[])
        
        for block in circuit.blocks
            gate_type, qubit_info = extract_gate_info(block)
            push!(df, (gate_type, qubit_info))
        end
        
        output_file = joinpath(save_path, "circuit_$(i).csv")
        CSV.write(output_file, df)
        
        if i % 10 == 0
            println("Generated $i circuits")
        end
    end
    
    println("Dataset generation complete. Files saved to $save_path")
end

# Run the dataset generation
create_circuit_dataset(100, 5, 20; save_path="/home/hash/PDC-Project/Dataset/")