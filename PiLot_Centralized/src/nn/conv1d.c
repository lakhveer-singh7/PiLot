#include "nn_types.h"
#include <math.h>
#include <string.h>

/* ================================================================== */
/*  Conv1D Forward                                                     */
/* ================================================================== */
void conv1d_forward(const tensor_t* in, const conv1d_config_t* cfg,
                    tensor_t* out) {
    if (!in || !cfg || !out || !in->data || !cfg->weights || !out->data) {
        log_error("conv1d_forward: NULL arg"); return;
    }

    int B  = in->batch_size;
    int Ci = cfg->in_channels;
    int Co = cfg->out_channels;
    int K  = cfg->kernel_size;
    int S  = cfg->stride;
    int P  = cfg->padding;
    int Li = in->length;
    int Lo = out->length;

    memset(out->data, 0, (size_t)B * Co * Lo * sizeof(float));

    for (int b = 0; b < B; b++) {
        for (int co = 0; co < Co; co++) {
            for (int lo = 0; lo < Lo; lo++) {
                float sum = 0.0f;
                int start = lo * S - P;
                for (int ci = 0; ci < Ci; ci++) {
                    for (int k = 0; k < K; k++) {
                        int li = start + k;
                        if (li >= 0 && li < Li) {
                            float w = cfg->weights[co * Ci * K + ci * K + k];
                            float x = in->data[b * Ci * Li + ci * Li + li];
                            sum += w * x;
                        }
                    }
                }
                if (cfg->bias)
                    sum += cfg->bias[co];
                out->data[b * Co * Lo + co * Lo + lo] = sum;
            }
        }
    }
}

/* ================================================================== */
/*  Conv1D Backward                                                    */
/* ================================================================== */
void conv1d_backward(const tensor_t* grad_out, const tensor_t* in,
                     const conv1d_config_t* cfg, tensor_t* grad_in,
                     float* grad_w, float* grad_b) {
    if (!grad_out || !in || !cfg || !grad_in) {
        log_error("conv1d_backward: NULL arg"); return;
    }

    int B  = in->batch_size;
    int Ci = cfg->in_channels;
    int Co = cfg->out_channels;
    int K  = cfg->kernel_size;
    int S  = cfg->stride;
    int P  = cfg->padding;
    int Li = in->length;
    int Lo = grad_out->length;

    if (grad_w) memset(grad_w, 0, (size_t)cfg->weights_size);
    if (grad_b) memset(grad_b, 0, (size_t)cfg->bias_size);
    if (grad_in && grad_in->data)
        memset(grad_in->data, 0,
               (size_t)B * Ci * Li * sizeof(float));

    for (int b = 0; b < B; b++) {
        for (int co = 0; co < Co; co++) {
            for (int lo = 0; lo < Lo; lo++) {
                float g = grad_out->data[b * Co * Lo + co * Lo + lo];
                int start = lo * S - P;

                if (grad_b)
                    grad_b[co] += g / B;

                for (int ci = 0; ci < Ci; ci++) {
                    for (int k = 0; k < K; k++) {
                        int li = start + k;
                        if (li >= 0 && li < Li) {
                            float x = in->data[b * Ci * Li + ci * Li + li];
                            if (grad_w)
                                grad_w[co * Ci * K + ci * K + k] += g * x / B;
                            if (grad_in && grad_in->data)
                                grad_in->data[b * Ci * Li + ci * Li + li] +=
                                    g * cfg->weights[co * Ci * K + ci * K + k];
                        }
                    }
                }
            }
        }
    }
}

/* ================================================================== */
/*  create / free Conv1D config (Xavier init)                          */
/* ================================================================== */
conv1d_config_t* create_conv1d_config(int in_ch, int out_ch, int kernel,
                                      int stride, int padding) {
    conv1d_config_t* c = (conv1d_config_t*)calloc(1, sizeof(conv1d_config_t));
    if (!c) return NULL;

    c->in_channels  = in_ch;
    c->out_channels = out_ch;
    c->kernel_size  = kernel;
    c->stride       = stride;
    c->padding      = padding;

    int nw = out_ch * in_ch * kernel;
    c->weights_size = nw * (int)sizeof(float);
    c->bias_size    = out_ch * (int)sizeof(float);

    c->weights = (float*)malloc((size_t)c->weights_size);
    c->bias    = (float*)calloc(out_ch, sizeof(float));
    if (!c->weights || !c->bias) {
        free(c->weights); free(c->bias); free(c);
        return NULL;
    }

    /* Kaiming (He) init */
    float scale = sqrtf(2.0f / (float)(in_ch * kernel));
    for (int i = 0; i < nw; i++)
        c->weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;

    log_info("Conv1D %d→%d  k=%d s=%d p=%d  (%.1fKB)",
             in_ch, out_ch, kernel, stride, padding,
             c->weights_size / 1024.0f);
    return c;
}

void free_conv1d_config(conv1d_config_t* c) {
    if (c) { free(c->weights); free(c->bias); free(c); }
}

/* ================================================================== */
/*  Group Normalisation  (forward + backward)                          */
/* ================================================================== */
void group_norm_forward(tensor_t* in, tensor_t* out, int num_groups) {
    if (!in || !out || !in->data || !out->data) return;

    int B = in->batch_size, C = in->channels, L = in->length;
    int cpg = C / num_groups;        /* channels per group */
    float eps = 1e-5f;

    for (int b = 0; b < B; b++) {
        for (int g = 0; g < num_groups; g++) {
            int c0 = g * cpg;
            /* mean */
            double sum = 0;
            int cnt = cpg * L;
            for (int c = c0; c < c0 + cpg; c++)
                for (int l = 0; l < L; l++)
                    sum += in->data[b * C * L + c * L + l];
            float mean = (float)(sum / cnt);
            /* var */
            double vsum = 0;
            for (int c = c0; c < c0 + cpg; c++)
                for (int l = 0; l < L; l++) {
                    float d = in->data[b * C * L + c * L + l] - mean;
                    vsum += d * d;
                }
            float std = sqrtf((float)(vsum / cnt) + eps);
            /* normalise */
            for (int c = c0; c < c0 + cpg; c++)
                for (int l = 0; l < L; l++) {
                    int idx = b * C * L + c * L + l;
                    out->data[idx] = (in->data[idx] - mean) / std;
                }
        }
    }
}

void group_norm_backward(const tensor_t* in, const tensor_t* grad_out,
                         tensor_t* grad_in, int num_groups) {
    if (!in || !grad_out || !grad_in) return;

    int B = in->batch_size, C = in->channels, L = in->length;
    int cpg = C / num_groups;
    float eps = 1e-5f;

    for (int b = 0; b < B; b++) {
        for (int g = 0; g < num_groups; g++) {
            int c0 = g * cpg;
            int N = cpg * L;

            /* recompute mean, var */
            float mean = 0.0f;
            for (int c = c0; c < c0 + cpg; c++)
                for (int l = 0; l < L; l++)
                    mean += in->data[b * C * L + c * L + l];
            mean /= (float)N;

            float var = 0.0f;
            for (int c = c0; c < c0 + cpg; c++)
                for (int l = 0; l < L; l++) {
                    float d = in->data[b * C * L + c * L + l] - mean;
                    var += d * d;
                }
            var /= (float)N;
            float inv_std = 1.0f / sqrtf(var + eps);

            /* accumulate sums for backward */
            float sum_dy = 0.0f, sum_dy_xmu = 0.0f;
            for (int c = c0; c < c0 + cpg; c++)
                for (int l = 0; l < L; l++) {
                    int idx = b * C * L + c * L + l;
                    float xmu = in->data[idx] - mean;
                    sum_dy     += grad_out->data[idx];
                    sum_dy_xmu += grad_out->data[idx] * xmu;
                }

            /* backward formula */
            for (int c = c0; c < c0 + cpg; c++)
                for (int l = 0; l < L; l++) {
                    int idx = b * C * L + c * L + l;
                    float xmu = in->data[idx] - mean;
                    grad_in->data[idx] = inv_std * (
                        grad_out->data[idx]
                        - sum_dy / N
                        - xmu * sum_dy_xmu / (N * (var + eps))
                    );
                }
        }
    }
}
