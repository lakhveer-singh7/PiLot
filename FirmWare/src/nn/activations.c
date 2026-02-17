#include "nn_types.h"
#include <math.h>

void relu_forward(tensor_t* input, tensor_t* output) {
    if (!input || !output || !input->data || !output->data) {
        log_error("Invalid tensors for ReLU forward");
        return;
    }
    
    if (input->batch_size != output->batch_size || 
        input->channels != output->channels || 
        input->length != output->length) {
        log_error("Tensor dimension mismatch for ReLU");
        return;
    }
    
    int total_elements = input->batch_size * input->channels * input->length;
    for (int i = 0; i < total_elements; i++) {
        output->data[i] = input->data[i] > 0.0f ? input->data[i] : 0.0f;
    }
}

void relu_backward(const tensor_t* grad_output, const tensor_t* input, tensor_t* grad_input) {
    if (!grad_output || !input || !grad_input || 
        !grad_output->data || !input->data || !grad_input->data) {
        log_error("Invalid tensors for ReLU backward");
        return;
    }
    
    int total_elements = input->batch_size * input->channels * input->length;
    for (int i = 0; i < total_elements; i++) {
        grad_input->data[i] = input->data[i] > 0.0f ? grad_output->data[i] : 0.0f;
    }
}

void softmax_forward(tensor_t* input, tensor_t* output) {
    if (!input || !output || !input->data || !output->data) {
        log_error("Invalid tensors for softmax forward");
        return;
    }
    
    int batch_size = input->batch_size;
    int num_classes = input->channels;  // Assuming shape is [batch, classes, 1]
    
    for (int b = 0; b < batch_size; b++) {
        float* in_batch = input->data + b * num_classes;
        float* out_batch = output->data + b * num_classes;
        
        // Find maximum for numerical stability
        float max_val = in_batch[0];
        for (int i = 1; i < num_classes; i++) {
            if (in_batch[i] > max_val) max_val = in_batch[i];
        }
        
        // Compute exp and sum
        float sum_exp = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            out_batch[i] = expf(in_batch[i] - max_val);
            sum_exp += out_batch[i];
        }
        
        // Normalize
        for (int i = 0; i < num_classes; i++) {
            out_batch[i] /= sum_exp;
        }
    }
}

float cross_entropy_loss(const tensor_t* predictions, const int* true_labels, int num_samples) {
    if (!predictions || !true_labels || !predictions->data) {
        log_error("Invalid arguments for cross_entropy_loss");
        return -1.0f;
    }
    
    float total_loss = 0.0f;
    int num_classes = predictions->channels;
    
    for (int i = 0; i < num_samples; i++) {
        int true_label = true_labels[i];
        if (true_label < 0 || true_label >= num_classes) {
            log_error("Invalid label %d for sample %d", true_label, i);
            continue;
        }
        
        float predicted_prob = predictions->data[i * num_classes + true_label];
        
        // Avoid log(0) by clamping to small positive value
        if (predicted_prob < 1e-7f) predicted_prob = 1e-7f;
        
        total_loss -= logf(predicted_prob);
    }
    
    return total_loss / num_samples;
}

void cross_entropy_backward(const tensor_t* predictions, const int* true_labels, 
                           int num_samples, tensor_t* grad_output) {
    if (!predictions || !true_labels || !grad_output || 
        !predictions->data || !grad_output->data) {
        log_error("Invalid arguments for cross_entropy_backward");
        return;
    }
    
    int num_classes = predictions->channels;
    
    // Initialize gradients
    tensor_fill_zeros(grad_output);
    
    for (int i = 0; i < num_samples; i++) {
        int true_label = true_labels[i];
        if (true_label < 0 || true_label >= num_classes) continue;
        
        float* grad_batch = grad_output->data + i * num_classes;
        float* pred_batch = predictions->data + i * num_classes;
        
        // Gradient of cross-entropy w.r.t. softmax output
        for (int j = 0; j < num_classes; j++) {
            if (j == true_label) {
                grad_batch[j] = (pred_batch[j] - 1.0f) / num_samples;
            } else {
                grad_batch[j] = pred_batch[j] / num_samples;
            }
        }
    }
}