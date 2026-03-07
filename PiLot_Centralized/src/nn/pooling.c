#include "nn_types.h"
#include <float.h>
#include <string.h>

/* ================================================================== */
/*  Dual Pooling (GAP + GMP concatenated)                              */
/*  Input:  [B, C, L]   →  Output: [B, 2C, 1]                        */
/* ================================================================== */
void dual_pooling1d(const tensor_t* in, tensor_t* out) {
    if (!in || !out || !in->data || !out->data) {
        log_error("dual_pooling1d: NULL arg"); return;
    }

    int B = in->batch_size, C = in->channels, L = in->length;

    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            float sum = 0.0f;
            float mx  = -FLT_MAX;
            for (int l = 0; l < L; l++) {
                float v = in->data[b * C * L + c * L + l];
                sum += v;
                if (v > mx) mx = v;
            }
            out->data[b * 2 * C + c]     = sum / L;   /* GAP */
            out->data[b * 2 * C + C + c] = mx;        /* GMP */
        }
    }
}

/* ================================================================== */
/*  Dual Pooling Backward                                              */
/* ================================================================== */
void dual_pooling1d_backward(const tensor_t* grad_out, const tensor_t* in,
                             tensor_t* grad_in) {
    if (!grad_out || !in || !grad_in ||
        !grad_out->data || !in->data || !grad_in->data) {
        log_error("dual_pooling1d_backward: NULL arg"); return;
    }

    int B = in->batch_size, C = in->channels, L = in->length;
    tensor_fill_zeros(grad_in);

    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            /* GAP backward: distribute evenly */
            float gap_g = grad_out->data[b * 2 * C + c] / (float)L;
            for (int l = 0; l < L; l++)
                grad_in->data[b * C * L + c * L + l] += gap_g;

            /* GMP backward: pass to argmax only */
            int   mx_pos = 0;
            float mx_val = -FLT_MAX;
            for (int l = 0; l < L; l++) {
                float v = in->data[b * C * L + c * L + l];
                if (v > mx_val) { mx_val = v; mx_pos = l; }
            }
            float gmp_g = grad_out->data[b * 2 * C + C + c];
            grad_in->data[b * C * L + c * L + mx_pos] += gmp_g;
        }
    }
}
