/*
 * pooling.cl — 2D Pooling Kernels (NCHW format)
 * ===============================================
 * Supports Max Pooling and Average Pooling.
 * Each work-item computes one output spatial location for one feature map.
 */

/* ------------------------------------------------------------------ */
/* Max Pooling 2D                                                      */
/* output[n][c][h_out][w_out] = max over pooling window               */
/* ------------------------------------------------------------------ */
__kernel void max_pool2d(
    __global const float* input,     /* (N, C, H, W)         */
    __global float*       output,    /* (N, C, H_out, W_out) */
    const int N,
    const int C,
    const int H,
    const int W,
    const int H_out,
    const int W_out,
    const int pool_h,
    const int pool_w,
    const int stride_h,
    const int stride_w
) {
    int n     = get_global_id(0);
    int c     = get_global_id(1);
    int hw    = get_global_id(2);

    if (n >= N || c >= C || hw >= H_out * W_out) return;

    int h_out = hw / W_out;
    int w_out = hw % W_out;

    int h_start = h_out * stride_h;
    int w_start = w_out * stride_w;

    float max_val = -FLT_MAX;

    for (int ph = 0; ph < pool_h; ph++) {
        for (int pw = 0; pw < pool_w; pw++) {
            int h_in = h_start + ph;
            int w_in = w_start + pw;
            if (h_in < H && w_in < W) {
                float val = input[
                    n * (C * H * W) + c * (H * W) + h_in * W + w_in
                ];
                max_val = fmax(max_val, val);
            }
        }
    }

    output[n * (C * H_out * W_out) + c * (H_out * W_out) + h_out * W_out + w_out] = max_val;
}

/* ------------------------------------------------------------------ */
/* Average Pooling 2D                                                  */
/* output[n][c][h_out][w_out] = mean over pooling window              */
/* ------------------------------------------------------------------ */
__kernel void avg_pool2d(
    __global const float* input,
    __global float*       output,
    const int N,
    const int C,
    const int H,
    const int W,
    const int H_out,
    const int W_out,
    const int pool_h,
    const int pool_w,
    const int stride_h,
    const int stride_w
) {
    int n     = get_global_id(0);
    int c     = get_global_id(1);
    int hw    = get_global_id(2);

    if (n >= N || c >= C || hw >= H_out * W_out) return;

    int h_out = hw / W_out;
    int w_out = hw % W_out;

    int h_start = h_out * stride_h;
    int w_start = w_out * stride_w;

    float sum   = 0.0f;
    int   count = 0;

    for (int ph = 0; ph < pool_h; ph++) {
        for (int pw = 0; pw < pool_w; pw++) {
            int h_in = h_start + ph;
            int w_in = w_start + pw;
            if (h_in < H && w_in < W) {
                sum += input[
                    n * (C * H * W) + c * (H * W) + h_in * W + w_in
                ];
                count++;
            }
        }
    }

    output[n * (C * H_out * W_out) + c * (H_out * W_out) + h_out * W_out + w_out] =
        (count > 0) ? sum / (float)count : 0.0f;
}

/* ------------------------------------------------------------------ */
/* Global Average Pooling 2D                                           */
/* output[n][c] = mean of input[n][c][*][*]  → (N, C)                */
/* ------------------------------------------------------------------ */
__kernel void global_avg_pool2d(
    __global const float* input,
    __global float*       output,
    const int N,
    const int C,
    const int H,
    const int W
) {
    int n = get_global_id(0);
    int c = get_global_id(1);

    if (n >= N || c >= C) return;

    float sum = 0.0f;
    int hw = H * W;
    int base = n * (C * hw) + c * hw;

    for (int i = 0; i < hw; i++) {
        sum += input[base + i];
    }

    output[n * C + c] = sum / (float)hw;
}
