#ifndef NN_TYPES_H
#define NN_TYPES_H

#include "lw_pilot_sim.h"

// Tensor structure
typedef struct tensor {
    float* data;
    int batch_size;
    int channels;
    int length;
    int stride;
    int allocated_size;
    int sample_id;       // NEW: Sample sequence number for pipeline coordination
} tensor_t;

// Convolution layer configuration
typedef struct {
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    float* weights;      // Shape: [out_channels][in_channels][kernel_size]
    float* bias;         // Shape: [out_channels]
    int weights_size;
    int bias_size;
} conv1d_config_t;

// Fully connected layer configuration
typedef struct {
    int in_features;
    int out_features;
    float* weights;      // Shape: [out_features][in_features]
    float* bias;         // Shape: [out_features]
    int weights_size;
    int bias_size;
} fc_config_t;

// Function prototypes for tensor operations
tensor_t* tensor_create(int batch_size, int channels, int length);
void tensor_free(tensor_t* tensor);
void tensor_fill_random(tensor_t* tensor);
void tensor_fill_zeros(tensor_t* tensor);
void tensor_copy(tensor_t* dst, const tensor_t* src);
void tensor_print(const tensor_t* tensor, const char* name);
float tensor_get(const tensor_t* tensor, int batch, int channel, int pos);
void tensor_set(tensor_t* tensor, int batch, int channel, int pos, float value);

// Neural network layer functions
void conv1d_forward(const tensor_t* input, const conv1d_config_t* config, tensor_t* output);
void conv1d_backward(const tensor_t* grad_output, const tensor_t* input, 
                     const conv1d_config_t* config, tensor_t* grad_input,
                     float* grad_weights, float* grad_bias);
conv1d_config_t* create_conv1d_config(int in_channels, int out_channels, int kernel_size, int stride, int padding);
void free_conv1d_config(conv1d_config_t* config);
void group_norm_forward(tensor_t* input, tensor_t* output, int num_groups);

void relu_forward(tensor_t* input, tensor_t* output);
void relu_backward(const tensor_t* grad_output, const tensor_t* input, tensor_t* grad_input);
void softmax_forward(tensor_t* input, tensor_t* output);
float cross_entropy_loss(const tensor_t* predictions, const int* true_labels, int num_samples);
void cross_entropy_backward(const tensor_t* predictions, const int* true_labels, 
                           int num_samples, tensor_t* grad_output);

void global_average_pooling1d(const tensor_t* input, tensor_t* output);
void global_max_pooling1d(const tensor_t* input, tensor_t* output);
void dual_pooling1d(const tensor_t* input, tensor_t* output);
void global_average_pooling1d_backward(const tensor_t* grad_output, const tensor_t* input, tensor_t* grad_input);
void global_max_pooling1d_backward(const tensor_t* grad_output, const tensor_t* input, tensor_t* grad_input);

void fully_connected_forward(const tensor_t* input, const fc_config_t* config, tensor_t* output);
void fully_connected_backward(const tensor_t* grad_output, const tensor_t* input, 
                             const fc_config_t* config, tensor_t* grad_input,
                             float* grad_weights, float* grad_bias);
fc_config_t* create_fc_config(int in_features, int out_features);
void free_fc_config(fc_config_t* config);

// Memory management
void* sim_malloc(size_t size);
void sim_free(void* ptr);
void sim_free_tracked(void* ptr, size_t size);
void print_memory_usage(void);
void print_memory_usage(void);

#endif // NN_TYPES_H