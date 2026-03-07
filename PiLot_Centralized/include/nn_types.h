#ifndef NN_TYPES_H
#define NN_TYPES_H

#include "logging.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/*  Tensor                                                             */
/* ------------------------------------------------------------------ */
typedef struct tensor {
    float* data;
    int    label;
    int    batch_size;
    int    channels;
    int    length;
    int    stride;
    int    allocated_size;   /* bytes */
} tensor_t;

/* Tensor helpers */
tensor_t* tensor_create(int batch_size, int channels, int length);
void      tensor_free(tensor_t* t);
void      tensor_fill_zeros(tensor_t* t);
void      tensor_fill_random(tensor_t* t);
void      tensor_copy(tensor_t* dst, const tensor_t* src);
void      tensor_print(const tensor_t* t, const char* name);
float     tensor_get(const tensor_t* t, int b, int c, int l);
void      tensor_set(tensor_t* t, int b, int c, int l, float v);

/* ------------------------------------------------------------------ */
/*  Conv1D layer                                                       */
/* ------------------------------------------------------------------ */
typedef struct {
    int    in_channels;
    int    out_channels;
    int    kernel_size;
    int    stride;
    int    padding;
    float* weights;          /* [out_ch][in_ch][kernel] */
    float* bias;             /* [out_ch] */
    int    weights_size;     /* bytes */
    int    bias_size;        /* bytes */
} conv1d_config_t;

conv1d_config_t* create_conv1d_config(int in_ch, int out_ch, int kernel,
                                      int stride, int padding);
void free_conv1d_config(conv1d_config_t* c);

void conv1d_forward(const tensor_t* in, const conv1d_config_t* cfg,
                    tensor_t* out);
void conv1d_backward(const tensor_t* grad_out, const tensor_t* in,
                     const conv1d_config_t* cfg, tensor_t* grad_in,
                     float* grad_w, float* grad_b);

void group_norm_forward(tensor_t* in, tensor_t* out, int num_groups);
void group_norm_backward(const tensor_t* in, const tensor_t* grad_out,
                         tensor_t* grad_in, int num_groups);

/* ------------------------------------------------------------------ */
/*  Fully-connected layer                                              */
/* ------------------------------------------------------------------ */
typedef struct {
    int    in_features;
    int    out_features;
    float* weights;          /* [out][in] */
    float* bias;             /* [out] */
    int    weights_size;     /* bytes */
    int    bias_size;        /* bytes */
} fc_config_t;

fc_config_t* create_fc_config(int in_features, int out_features);
void         free_fc_config(fc_config_t* c);

void fully_connected_forward(const tensor_t* in, const fc_config_t* cfg,
                             tensor_t* out);
void fully_connected_backward(const tensor_t* grad_out, const tensor_t* in,
                              const fc_config_t* cfg, tensor_t* grad_in,
                              float* grad_w, float* grad_b);

/* ------------------------------------------------------------------ */
/*  Activations / Loss                                                 */
/* ------------------------------------------------------------------ */
void  relu_forward(tensor_t* in, tensor_t* out);
void  relu_backward(const tensor_t* grad_out, const tensor_t* in,
                    tensor_t* grad_in);
void  softmax_forward(tensor_t* in, tensor_t* out);
float cross_entropy_loss(const tensor_t* preds, const int* labels, int n);
void  cross_entropy_backward(const tensor_t* softmax_out, const int* labels,
                             int batch_size, tensor_t* grad_logits);

void dropout_forward(const tensor_t* in, tensor_t* out, float* mask,
                     float drop_rate, int is_training);
void dropout_backward(const tensor_t* grad_out, const float* mask,
                      tensor_t* grad_in);

/* ------------------------------------------------------------------ */
/*  Pooling                                                            */
/* ------------------------------------------------------------------ */
void dual_pooling1d(const tensor_t* in, tensor_t* out);
void dual_pooling1d_backward(const tensor_t* grad_out, const tensor_t* in,
                             tensor_t* grad_in);

/* ------------------------------------------------------------------ */
/*  Optimizers                                                         */
/* ------------------------------------------------------------------ */
void adam_update(float* w, const float* gw, float* m, float* v,
                int size, float lr, float b1, float b2,
                float eps, int t, float wd);
void adam_update_bias(float* b, const float* gb, float* m, float* v,
                     int size, float lr, float b1, float b2,
                     float eps, int t);
float lr_cosine_annealing(float base_lr, int epoch, int T_max, float eta_min);

/* ------------------------------------------------------------------ */
/*  Data augmentation                                                  */
/* ------------------------------------------------------------------ */
void apply_augmentation(float* data, int length);

#endif /* NN_TYPES_H */
