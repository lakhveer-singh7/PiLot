#include "nn_types.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ------------------------------------------------------------------ */
/*  Tensor creation / destruction  (plain malloc, no memory limit)     */
/* ------------------------------------------------------------------ */
tensor_t* tensor_create(int batch_size, int channels, int length) {
    if (batch_size <= 0 || channels <= 0 || length <= 0) {
        log_error("Invalid tensor dims: %dx%dx%d", batch_size, channels, length);
        return NULL;
    }

    tensor_t* t = (tensor_t*)calloc(1, sizeof(tensor_t));
    if (!t) return NULL;

    int total = batch_size * channels * length;
    t->allocated_size = total * (int)sizeof(float);
    t->data = (float*)calloc(total, sizeof(float));
    if (!t->data) { free(t); return NULL; }

    t->batch_size = batch_size;
    t->channels   = channels;
    t->length     = length;
    t->stride     = length;
    return t;
}

void tensor_free(tensor_t* t) {
    if (t) {
        free(t->data);
        free(t);
    }
}

void tensor_fill_zeros(tensor_t* t) {
    if (!t || !t->data) return;
    memset(t->data, 0,
           (size_t)t->batch_size * t->channels * t->length * sizeof(float));
}

void tensor_fill_random(tensor_t* t) {
    if (!t || !t->data) return;
    int n = t->batch_size * t->channels * t->length;
    for (int i = 0; i < n; i++)
        t->data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

void tensor_copy(tensor_t* dst, const tensor_t* src) {
    if (!dst || !src || !dst->data || !src->data) return;
    if (dst->batch_size != src->batch_size ||
        dst->channels   != src->channels   ||
        dst->length     != src->length) {
        log_error("tensor_copy: shape mismatch");
        return;
    }
    memcpy(dst->data, src->data, (size_t)src->allocated_size);
}

void tensor_print(const tensor_t* t, const char* name) {
    if (!t || !t->data) { printf("%s: NULL\n", name ? name : "tensor"); return; }
    printf("%s: (%d,%d,%d) data[0..4]=", name ? name : "tensor",
           t->batch_size, t->channels, t->length);
    int n = t->batch_size * t->channels * t->length;
    int show = n < 5 ? n : 5;
    for (int i = 0; i < show; i++) printf(" %.4f", t->data[i]);
    printf("\n");
}

static inline int tidx(const tensor_t* t, int b, int c, int l) {
    return b * t->channels * t->stride + c * t->stride + l;
}

float tensor_get(const tensor_t* t, int b, int c, int l) {
    if (!t || !t->data) return 0.0f;
    if (b >= t->batch_size || c >= t->channels || l >= t->length) return 0.0f;
    return t->data[tidx(t, b, c, l)];
}

void tensor_set(tensor_t* t, int b, int c, int l, float v) {
    if (!t || !t->data) return;
    if (b >= t->batch_size || c >= t->channels || l >= t->length) return;
    t->data[tidx(t, b, c, l)] = v;
}
