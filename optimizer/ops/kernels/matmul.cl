/*
 * matmul.cl — Tiled Matrix Multiplication Kernel
 * ================================================
 * Computes: C = A * B  where A is (M x K), B is (K x N), C is (M x N)
 *
 * Uses a tile-based approach to exploit local memory for cache efficiency.
 * TILE_SIZE is set to 16 — a good balance for most GPU local memory sizes.
 */

#define TILE_SIZE 16

__kernel void matmul(
    __global const float* A,    /* Input matrix A [M x K] */
    __global const float* B,    /* Input matrix B [K x N] */
    __global float*       C,    /* Output matrix C [M x N] */
    const int M,
    const int K,
    const int N
) {
    /* Allocate local (shared) memory tiles */
    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    /* Global row and column for this work-item */
    int row = get_global_id(0);
    int col = get_global_id(1);

    /* Local row and column within the tile */
    int localRow = get_local_id(0);
    int localCol = get_local_id(1);

    float acc = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        /* Load tile from A into local memory */
        int aCol = t * TILE_SIZE + localCol;
        tileA[localRow][localCol] = (row < M && aCol < K)
            ? A[row * K + aCol]
            : 0.0f;

        /* Load tile from B into local memory */
        int bRow = t * TILE_SIZE + localRow;
        tileB[localRow][localCol] = (bRow < K && col < N)
            ? B[bRow * N + col]
            : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        /* Accumulate partial dot product */
        for (int k = 0; k < TILE_SIZE; k++) {
            acc += tileA[localRow][k] * tileB[k][localCol];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /* Write result */
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

/*
 * dense_forward — Optimized Dense (fully connected) layer forward pass
 * ==========================================================
 * out = inputs @ weights + bias
 * Uses tiling to improve cache locality.
 */
__kernel void dense_forward(
    __global const float* inputs,
    __global const float* weights,
    __global const float* bias,
    __global float*       output,
    const int batch_size,
    const int in_features,
    const int out_features
) {
    __local float tile_in[TILE_SIZE][TILE_SIZE];
    __local float tile_w[TILE_SIZE][TILE_SIZE];

    int row = get_global_id(0); // batch index
    int col = get_global_id(1); // output index

    int localRow = get_local_id(0);
    int localCol = get_local_id(1);

    float sum = 0.0f;
    int numTiles = (in_features + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Load inputs into local memory
        int inCol = t * TILE_SIZE + localCol;
        tile_in[localRow][localCol] = (row < batch_size && inCol < in_features)
            ? inputs[row * in_features + inCol] : 0.0f;

        // Load weights into local memory
        int wRow = t * TILE_SIZE + localRow;
        tile_w[localRow][localCol] = (wRow < in_features && col < out_features)
            ? weights[wRow * out_features + col] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_in[localRow][k] * tile_w[k][localCol];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < batch_size && col < out_features) {
        output[row * out_features + col] = sum + bias[col];
    }
}
