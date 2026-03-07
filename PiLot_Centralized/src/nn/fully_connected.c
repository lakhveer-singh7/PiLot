#include "nn_types.h"
#include <string.h>
#include <math.h>

/* ================================================================== */
/*  Fully-Connected Forward                                            */
/*  Input flattened: in_features = channels * length                   */
/* ================================================================== */
void fully_connected_forward(const tensor_t* in, const fc_config_t* cfg,
                             tensor_t* out) {
    if (!in || !cfg || !out || !in->data || !cfg->weights || !out->data) {
        log_error("fc_forward: NULL arg"); return;
    }

    int B   = in->batch_size;
    int Fi  = cfg->in_features;
    int Fo  = cfg->out_features;

    for (int b = 0; b < B; b++) {
        const float* x = in->data + b * Fi;
        float*       y = out->data + b * Fo;
        for (int o = 0; o < Fo; o++) {
            float s = cfg->bias ? cfg->bias[o] : 0.0f;
            const float* w = cfg->weights + o * Fi;
            for (int i = 0; i < Fi; i++)
                s += w[i] * x[i];
            y[o] = s;
        }
    }
}

/* ================================================================== */
/*  Fully-Connected Backward                                           */
/* ================================================================== */
void fully_connected_backward(const tensor_t* grad_out, const tensor_t* in,
                              const fc_config_t* cfg, tensor_t* grad_in,
                              float* grad_w, float* grad_b) {
    if (!grad_out || !in || !cfg || !grad_in) {
        log_error("fc_backward: NULL arg"); return;
    }

    int B   = in->batch_size;
    int Fi  = cfg->in_features;
    int Fo  = cfg->out_features;

    if (grad_w) memset(grad_w, 0, (size_t)cfg->weights_size);
    if (grad_b) memset(grad_b, 0, (size_t)cfg->bias_size);

    for (int b = 0; b < B; b++) {
        const float* x  = in->data + b * Fi;
        const float* gy = grad_out->data + b * Fo;
        float*       gx = grad_in->data + b * Fi;

        memset(gx, 0, (size_t)Fi * sizeof(float));

        for (int o = 0; o < Fo; o++) {
            float g = gy[o];
            if (grad_b) grad_b[o] += g / B;
            const float* w = cfg->weights + o * Fi;
            float* gw_row  = grad_w ? grad_w + o * Fi : NULL;
            for (int i = 0; i < Fi; i++) {
                gx[i] += g * w[i];
                if (gw_row) gw_row[i] += g * x[i] / B;
            }
        }
    }
}

/* ================================================================== */
/*  create / free FC config (Xavier init)                              */
/* ================================================================== */
fc_config_t* create_fc_config(int in_features, int out_features) {
    fc_config_t* c = (fc_config_t*)calloc(1, sizeof(fc_config_t));
    if (!c) return NULL;

    c->in_features  = in_features;
    c->out_features = out_features;
    c->weights_size = out_features * in_features * (int)sizeof(float);
    c->bias_size    = out_features * (int)sizeof(float);

    c->weights = (float*)malloc((size_t)c->weights_size);
    c->bias    = (float*)calloc(out_features, sizeof(float));
    if (!c->weights || !c->bias) {
        free(c->weights); free(c->bias); free(c);
        return NULL;
    }

    float scale = sqrtf(2.0f / (float)(in_features + out_features));
    int nw = out_features * in_features;
    for (int i = 0; i < nw; i++)
        c->weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;

    log_info("FC %d → %d  (%.1fKB)", in_features, out_features,
             c->weights_size / 1024.0f);
    return c;
}

void free_fc_config(fc_config_t* c) {
    if (c) { free(c->weights); free(c->bias); free(c); }
}
