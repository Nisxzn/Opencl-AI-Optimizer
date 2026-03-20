/*
 * conv2d.cl — 2D Convolution Kernel (NCHW format)
 * ================================================
 * Computes: out[n][c_out][h][w] = sum over (c_in, kh, kw) of
 *               input[n][c_in][h*stride+kh][w*stride+kw] * kernel[c_out][c_in][kh][kw]
 *           + bias[c_out]
 *
 * Data layout: NCHW (batch, channels, height, width)
 * Each work-item computes one output element.
 */
__kernel void conv2d(
    __global const float* input,    /* (N, C_in,  H,     W)     */
    __global const float* weights,  /* (C_out, C_in, KH, KW)    */
    __global const float* bias,     /* (C_out,)                  */
    __global float*       output,   /* (N, C_out, H_out, W_out) */
    const int N,
    const int C_in,
    const int H,
    const int W,
    const int C_out,
    const int KH,
    const int KW,
    const int H_out,
    const int W_out,
    const int stride,
    const int padding
) {
    /* Decode the flattened global ID into (n, c_out, h_out, w_out) */
    int n     = get_global_id(0);
    int c_out = get_global_id(1);
    int hw    = get_global_id(2);

    if (n >= N || c_out >= C_out || hw >= H_out * W_out) return;

    int h_out_idx = hw / W_out;
    int w_out_idx = hw % W_out;

    float acc = bias[c_out];

    for (int c = 0; c < C_in; c++) {
        for (int kh = 0; kh < KH; kh++) {
            for (int kw = 0; kw < KW; kw++) {
                int h_in = h_out_idx * stride + kh - padding;
                int w_in = w_out_idx * stride + kw - padding;

                /* Boundary check (zero-padding handled implicitly) */
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    int in_idx = n * (C_in * H * W)
                               + c * (H * W)
                               + h_in * W
                               + w_in;
                    int wt_idx = c_out * (C_in * KH * KW)
                               + c * (KH * KW)
                               + kh * KW
                               + kw;
                    acc += input[in_idx] * weights[wt_idx];
                }
            }
        }
    }

    int out_idx = n * (C_out * H_out * W_out)
                + c_out * (H_out * W_out)
                + h_out_idx * W_out
                + w_out_idx;
    output[out_idx] = acc;
}


/*
 * im2col — flatten patches for efficient batched convolution (optional path)
 * ==========================================================================
 * Used by the higher-level Python code to convert input to a column matrix,
 * then matmul with reshaped weights — faster for larger kernels.
 */
__kernel void im2col(
    __global const float* input,
    __global float*       col_output,
    const int C_in,
    const int H,
    const int W,
    const int KH,
    const int KW,
    const int H_out,
    const int W_out,
    const int stride,
    const int padding
) {
    int h_out = get_global_id(0);
    int w_out = get_global_id(1);

    if (h_out >= H_out || w_out >= W_out) return;

    int col_offset = (h_out * W_out + w_out) * (C_in * KH * KW);

    for (int c = 0; c < C_in; c++) {
        for (int kh = 0; kh < KH; kh++) {
            for (int kw = 0; kw < KW; kw++) {
                int h_in = h_out * stride + kh - padding;
                int w_in = w_out * stride + kw - padding;
                int patch_idx = c * (KH * KW) + kh * KW + kw;

                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    col_output[col_offset + patch_idx] =
                        input[c * H * W + h_in * W + w_in];
                } else {
                    col_output[col_offset + patch_idx] = 0.0f;
                }
            }
        }
    }
}
