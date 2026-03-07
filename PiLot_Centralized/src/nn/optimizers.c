#include "nn_types.h"
#include <math.h>

/* ================================================================== */
/*  Cosine Annealing LR with warm restart                              */
/* ================================================================== */
float lr_cosine_annealing(float base_lr, int epoch, int T_max, float eta_min) {
    if (epoch < 3) {
        /* linear warmup */
        return eta_min + (base_lr - eta_min) * ((float)epoch / 3.0f);
    }
    int eff   = epoch - 3;
    int pos   = eff % T_max;
    float cos = cosf((float)pos / (float)T_max * 3.14159265f);
    return eta_min + 0.5f * (base_lr - eta_min) * (1.0f + cos);
}

/* ================================================================== */
/*  Adam / AdamW                                                       */
/* ================================================================== */
void adam_update(float* w, const float* gw, float* m, float* v,
                int size, float lr, float b1, float b2,
                float eps, int t, float wd) {
    if (!w || !gw || !m || !v) return;
    float bc1 = 1.0f - powf(b1, (float)t);
    float bc2 = 1.0f - powf(b2, (float)t);
    if (bc1 < 1e-10f) bc1 = 1e-10f;
    if (bc2 < 1e-10f) bc2 = 1e-10f;

    for (int i = 0; i < size; i++) {
        float g = gw[i] + wd * w[i];          /* decoupled weight decay */
        m[i] = b1 * m[i] + (1.0f - b1) * g;
        v[i] = b2 * v[i] + (1.0f - b2) * g * g;
        float mh = m[i] / bc1;
        float vh = v[i] / bc2;
        w[i] -= lr * mh / (sqrtf(vh) + eps);
    }
}

void adam_update_bias(float* b, const float* gb, float* m, float* v,
                     int size, float lr, float b1, float b2,
                     float eps, int t) {
    if (!b || !gb || !m || !v) return;
    float bc1 = 1.0f - powf(b1, (float)t);
    float bc2 = 1.0f - powf(b2, (float)t);
    if (bc1 < 1e-10f) bc1 = 1e-10f;
    if (bc2 < 1e-10f) bc2 = 1e-10f;

    for (int i = 0; i < size; i++) {
        m[i] = b1 * m[i] + (1.0f - b1) * gb[i];
        v[i] = b2 * v[i] + (1.0f - b2) * gb[i] * gb[i];
        float mh = m[i] / bc1;
        float vh = v[i] / bc2;
        b[i] -= lr * mh / (sqrtf(vh) + eps);
    }
}
