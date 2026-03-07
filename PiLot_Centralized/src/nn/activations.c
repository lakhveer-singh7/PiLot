#include "nn_types.h"
#include <math.h>
#include <stdlib.h>

/* ================================================================== */
/*  Dropout Forward  (inverted dropout)                                */
/* ================================================================== */
void dropout_forward(const tensor_t* in, tensor_t* out, float* mask,
                     float drop_rate, int is_training) {
    if (!in || !out || !in->data || !out->data || !mask) return;

    int n = in->batch_size * in->channels * in->length;
    float scale = 1.0f / (1.0f - drop_rate);

    if (is_training) {
        for (int i = 0; i < n; i++) {
            float r = (float)rand() / (float)RAND_MAX;
            if (r < drop_rate) {
                mask[i] = 0.0f;
                out->data[i] = 0.0f;
            } else {
                mask[i] = scale;
                out->data[i] = in->data[i] * scale;
            }
        }
    } else {
        for (int i = 0; i < n; i++)
            out->data[i] = in->data[i];
    }
}

void dropout_backward(const tensor_t* grad_out, const float* mask,
                      tensor_t* grad_in) {
    if (!grad_out || !mask || !grad_in) return;
    int n = grad_out->batch_size * grad_out->channels * grad_out->length;
    for (int i = 0; i < n; i++)
        grad_in->data[i] = grad_out->data[i] * mask[i];
}

/* ================================================================== */
/*  Leaky ReLU  (alpha = 0.01)                                        */
/* ================================================================== */
void relu_forward(tensor_t* in, tensor_t* out) {
    if (!in || !out || !in->data || !out->data) return;
    int n = in->batch_size * in->channels * in->length;
    for (int i = 0; i < n; i++)
        out->data[i] = in->data[i] > 0.0f ? in->data[i] : 0.01f * in->data[i];
}

void relu_backward(const tensor_t* grad_out, const tensor_t* in,
                   tensor_t* grad_in) {
    if (!grad_out || !in || !grad_in) return;
    int n = in->batch_size * in->channels * in->length;
    for (int i = 0; i < n; i++)
        grad_in->data[i] = in->data[i] > 0.0f
                           ? grad_out->data[i]
                           : 0.01f * grad_out->data[i];
}

/* ================================================================== */
/*  Softmax                                                            */
/* ================================================================== */
void softmax_forward(tensor_t* in, tensor_t* out) {
    if (!in || !out || !in->data || !out->data) return;
    if (in->length != 1) { log_error("softmax expects length==1, got %d", in->length); return; }
    int B = in->batch_size;
    int C = in->channels;

    for (int b = 0; b < B; b++) {
        float* x = in->data  + b * C;
        float* y = out->data + b * C;
        float mx = x[0];
        for (int i = 1; i < C; i++) if (x[i] > mx) mx = x[i];
        float s = 0.0f;
        for (int i = 0; i < C; i++) { y[i] = expf(x[i] - mx); s += y[i]; }
        for (int i = 0; i < C; i++) y[i] /= s;
    }
}

/* ================================================================== */
/*  Cross-Entropy Loss  +  Backward (combined softmax-CE gradient)     */
/* ================================================================== */
float cross_entropy_loss(const tensor_t* preds, const int* labels, int n) {
    if (!preds || !labels || !preds->data) return -1.0f;
    int C = preds->channels;
    float total = 0.0f;
    for (int i = 0; i < n; i++) {
        int lbl = labels[i];
        if (lbl < 0 || lbl >= C) continue;
        float p = preds->data[i * C + lbl];
        if (p < 1e-7f) p = 1e-7f;
        total -= logf(p);
    }
    return total / n;
}

void cross_entropy_backward(const tensor_t* softmax_out, const int* labels,
                            int batch_size, tensor_t* grad_logits) {
    int C = softmax_out->channels;
    for (int b = 0; b < batch_size; b++) {
        float* g = grad_logits->data + b * C;
        float* p = softmax_out->data + b * C;
        int lbl  = labels[b];
        if (lbl < 0 || lbl >= C) continue;
        for (int j = 0; j < C; j++) g[j] = p[j];
        g[lbl] -= 1.0f;
    }
    float inv = 1.0f / batch_size;
    int total = batch_size * C;
    for (int i = 0; i < total; i++)
        grad_logits->data[i] *= inv;
}
