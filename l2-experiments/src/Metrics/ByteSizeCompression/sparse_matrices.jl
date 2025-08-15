"""
    CSC(matrices::Vector{Matrix{Float32}})

Convert a vector of dense matrices to Compressed Sparse Column (CSC) format.

# Arguments
- `matrices::Vector{Matrix{Float32}}`: Vector of dense matrices to convert

# Returns
- `Vector{SparseMatrixCSC{Float32, Int64}}`: Vector of matrices in CSC format
"""
function CSC(matrices::Vector{Matrix{Float32}})
    return map(matrix -> sparse(matrix), matrices)
end

"""
    COO(matrices::Vector{Matrix{Float32}})

Convert a vector of dense matrices to Coordinate (COO) format.

# Arguments
- `matrices::Vector{Matrix{Float32}}`: Vector of dense matrices to convert

# Returns
- `Vector{Tuple{Vector{Int64}, Vector{Int64}, Vector{Float32}}}`: Vector of (rows, cols, vals) tuples
"""
function COO(matrices::Vector{Matrix{Float32}})
    return map(matrix -> COO(matrix), matrices)
end

"""
    COO(matrix::Matrix{Float32})

Convert a dense matrix to Coordinate (COO) format.

# Arguments
- `matrix::Matrix{Float32}`: Dense matrix to convert

# Returns
- `Tuple{Vector{Int64}, Vector{Int64}, Vector{Float32}}`: (rows, cols, vals) tuple
"""
function COO(matrix::Matrix{Float32})
   return COO(sparse(matrix)) 
end

"""
    COO(CSC_matrix::SparseMatrixCSC{Float32, Int64})

Convert a Compressed Sparse Column (CSC) matrix to Coordinate (COO) format.

# Arguments
- `CSC_matrix::SparseMatrixCSC{Float32, Int64}`: Sparse matrix in CSC format

# Returns
- `Tuple{Vector{Int64}, Vector{Int64}, Vector{Float32}}`: (rows, cols, vals) tuple representing the matrix in COO format
"""
function COO(CSC_matrix::SparseMatrixCSC{Float32, Int64})
    #CSC representation
    rowval = CSC_matrix.rowval
    colptr = CSC_matrix.colptr
    nzval = CSC_matrix.nzval

    #COO representation
    rows = rowval
    val = nzval
    cols = Int64[]

    cols = map(i -> last_matching_colptr_index(i, colptr), 1:length(nzval))

    return rows, cols, val
end