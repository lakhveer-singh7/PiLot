#include "nn_types.h"
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Helper RNG                                                         */
/* ------------------------------------------------------------------ */
static float randf(void) {
    return 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f;
}

static float gauss_noise(void) {
    /* CLT approximation: sum of 6 uniforms */
    float s = 0.0f;
    for (int i = 0; i < 6; i++) s += (float)rand() / (float)RAND_MAX;
    return s - 3.0f;
}

/* ------------------------------------------------------------------ */
/*  Individual transforms                                              */
/* ------------------------------------------------------------------ */
static void jitter(float* d, int n, float std) {
    for (int i = 0; i < n; i++) d[i] += gauss_noise() * std;
}

static void scaling(float* d, int n, float range) {
    float s = 1.0f + randf() * range;
    for (int i = 0; i < n; i++) d[i] *= s;
}

static void magnitude_warp(float* d, int n, float std) {
    float k[4];
    for (int i = 0; i < 4; i++) k[i] = 1.0f + gauss_noise() * std;
    for (int i = 0; i < n; i++) {
        float t = (float)i / (float)(n - 1) * 3.0f;
        int k0 = (int)t; if (k0 >= 3) k0 = 2;
        float f = t - (float)k0;
        float w = k[k0] * (1.0f - f) + k[k0 + 1] * f;
        d[i] *= w;
    }
}

static void time_shift(float* d, int n, int max_shift) {
    int shift = (rand() % (2 * max_shift + 1)) - max_shift;
    if (shift == 0) return;
    float* tmp = (float*)malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) {
        int s = i - shift;
        if (s < 0)  s = 0;
        if (s >= n) s = n - 1;
        tmp[i] = d[s];
    }
    memcpy(d, tmp, (size_t)n * sizeof(float));
    free(tmp);
}

/* ------------------------------------------------------------------ */
/*  Public: apply augmentation pipeline                                */
/* ------------------------------------------------------------------ */
void apply_augmentation(float* data, int length) {
    jitter(data, length, 0.03f);
    if (rand() % 100 < 70) scaling(data, length, 0.15f);
    if (rand() % 100 < 50) magnitude_warp(data, length, 0.1f);
    if (rand() % 100 < 50) time_shift(data, length, 5);
}
