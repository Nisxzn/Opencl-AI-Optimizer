/*
 * activations.cl — Element-wise Activation Kernels
 * ==================================================
 * Fully vectorised over all elements of the input tensor.
 * Each work-item handles one float value.
 */

/* ------------------------------------------------------------------ */
/* ReLU: f(x) = max(0, x)                                             */
/* ------------------------------------------------------------------ */
__kernel void relu(
    __global const float* input,
    __global float*       output,
    const int n_elements
) {
    int idx = get_global_id(0);
    if (idx >= n_elements) return;
    output[idx] = fmax(0.0f, input[idx]);
}

/* ------------------------------------------------------------------ */
/* Leaky ReLU: f(x) = x if x > 0 else alpha * x                      */
/* ------------------------------------------------------------------ */
__kernel void leaky_relu(
    __global const float* input,
    __global float*       output,
    const float           alpha,
    const int             n_elements
) {
    int idx = get_global_id(0);
    if (idx >= n_elements) return;
    float x = input[idx];
    output[idx] = (x > 0.0f) ? x : alpha * x;
}

/* ------------------------------------------------------------------ */
/* Sigmoid: f(x) = 1 / (1 + exp(-x))                                 */
/* ------------------------------------------------------------------ */
__kernel void sigmoid(
    __global const float* input,
    __global float*       output,
    const int n_elements
) {
    int idx = get_global_id(0);
    if (idx >= n_elements) return;
    output[idx] = 1.0f / (1.0f + exp(-input[idx]));
}

/* ------------------------------------------------------------------ */
/* Tanh: f(x) = tanh(x)                                               */
/* ------------------------------------------------------------------ */
__kernel void tanh_act(
    __global const float* input,
    __global float*       output,
    const int n_elements
) {
    int idx = get_global_id(0);
    if (idx >= n_elements) return;
    output[idx] = tanh(input[idx]);
}

/* ------------------------------------------------------------------ */
/* Softmax — numerically stable, single-pass (small vectors only)     */
/* For large softmax, use a two-pass reduction approach.               */
/* ------------------------------------------------------------------ */
__kernel void softmax_row(
    __global const float* input,
    __global float*       output,
    const int n_classes,
    const int batch_size
) {
    int b = get_global_id(0);
    if (b >= batch_size) return;

    const __global float* row = input + b * n_classes;
    __global float*       out = output + b * n_classes;

    /* 1) Find max for numerical stability */
    float max_val = row[0];
    for (int i = 1; i < n_classes; i++) {
        max_val = fmax(max_val, row[i]);
    }

    /* 2) Compute exp and sum */
    float sum = 0.0f;
    for (int i = 0; i < n_classes; i++) {
        out[i] = exp(row[i] - max_val);
        sum += out[i];
    }

    /* 3) Normalise */
    for (int i = 0; i < n_classes; i++) {
        out[i] /= sum;
    }
}

/* ------------------------------------------------------------------ */
/* ELU: f(x) = x if x >= 0 else alpha * (exp(x) - 1)                 */
/* ------------------------------------------------------------------ */
__kernel void elu(
    __global const float* input,
    __global float*       output,
    const float           alpha,
    const int             n_elements
) {
    int idx = get_global_id(0);
    if (idx >= n_elements) return;
    float x = input[idx];
    output[idx] = (x >= 0.0f) ? x : alpha * (exp(x) - 1.0f);
}
