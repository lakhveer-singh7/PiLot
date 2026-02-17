#include "lw_pilot_sim.h"
#include "nn_types.h"
#include "config_types.h"
#include "shared_memory.h"

// External configuration
extern device_json_config_t* g_device_config;
extern model_config_t* g_model_config;

// For shutdown detection - declared in shared_memory.c
extern volatile pipeline_phase_t* g_pipeline_ctrl;

// Forward declarations from nn layers
void dual_pooling1d(const tensor_t* input, tensor_t* output);
void global_average_pooling1d_backward(const tensor_t* grad_output, const tensor_t* input, tensor_t* grad_input);
void global_max_pooling1d_backward(const tensor_t* grad_output, const tensor_t* input, tensor_t* grad_input);
void fully_connected_forward(const tensor_t* input, const fc_config_t* config, tensor_t* output);
void fully_connected_backward(const tensor_t* grad_output, const tensor_t* input, 
                             const fc_config_t* config, tensor_t* grad_input,
                             float* grad_weights, float* grad_bias);
fc_config_t* create_fc_config(int in_features, int out_features);
void free_fc_config(fc_config_t* config);
void softmax_forward(tensor_t* input, tensor_t* output);
float cross_entropy_loss(const tensor_t* predictions, const int* true_labels, int num_samples);
void cross_entropy_backward(const tensor_t* predictions, const int* true_labels, 
                           int num_samples, tensor_t* grad_output);

typedef struct {
    int device_id;
    int num_classes;
    
    // Neural network layers
    fc_config_t* fc_config;
    
    // Forward pass tensors
    tensor_t* conv_features;    // Input from worker device
    tensor_t* pooled_features;  // After dual pooling
    tensor_t* fc_output;        // Fully connected output
    tensor_t* probabilities;    // After softmax
    
    // Backward pass tensors
    tensor_t* grad_fc_output;   // Gradient from loss
    tensor_t* grad_pooled;      // Gradient for pooled features
    tensor_t* grad_conv;        // Gradient to send back to worker
    
    // Gradient buffers (reusable)
    float* grad_weights;
    float* grad_bias;
    size_t grad_weights_size;
    size_t grad_bias_size;
    
    // Statistics
    int samples_processed;
    int correct_predictions;
    float total_loss;
    
} tail_context_t;

static void sgd_update(float* weights, const float* grads, size_t size_bytes, float lr) {
    if (!weights || !grads) return;
    size_t count = size_bytes / sizeof(float);
    for (size_t i = 0; i < count; i++) {
        weights[i] -= lr * grads[i];
    }
}

tail_context_t* create_tail_context(int device_id, int num_classes, int input_channels, int input_length) {
    tail_context_t* ctx = (tail_context_t*)sim_malloc(sizeof(tail_context_t));
    if (!ctx) {
        return NULL;
    }
    
    memset(ctx, 0, sizeof(tail_context_t));
    ctx->device_id = device_id;
    ctx->num_classes = num_classes;
    
    // Create fully connected layer: (input_channels * 2) → num_classes
    // Factor of 2 because dual pooling concatenates GAP and GMP
    int fc_input_size = input_channels * 2;
    ctx->fc_config = create_fc_config(fc_input_size, num_classes);
    if (!ctx->fc_config) {
        sim_free_tracked(ctx, sizeof(tail_context_t));
        return NULL;
    }
    
    // Allocate tensors
    ctx->conv_features = tensor_create(1, input_channels, input_length);
    ctx->pooled_features = tensor_create(1, fc_input_size, 1);  // Flattened pooled features
    ctx->fc_output = tensor_create(1, num_classes, 1);
    ctx->probabilities = tensor_create(1, num_classes, 1);
    ctx->grad_fc_output = tensor_create(1, num_classes, 1);
    ctx->grad_pooled = tensor_create(1, fc_input_size, 1);
    ctx->grad_conv = tensor_create(1, input_channels, input_length);
    
    if (!ctx->conv_features || !ctx->pooled_features || !ctx->fc_output || 
        !ctx->probabilities || !ctx->grad_fc_output || !ctx->grad_pooled || !ctx->grad_conv) {
        if (ctx->conv_features) tensor_free(ctx->conv_features);
        if (ctx->pooled_features) tensor_free(ctx->pooled_features);
        if (ctx->fc_output) tensor_free(ctx->fc_output);
        if (ctx->probabilities) tensor_free(ctx->probabilities);
        if (ctx->grad_fc_output) tensor_free(ctx->grad_fc_output);
        if (ctx->grad_pooled) tensor_free(ctx->grad_pooled);
        if (ctx->grad_conv) tensor_free(ctx->grad_conv);
        free_fc_config(ctx->fc_config);
        sim_free_tracked(ctx, sizeof(tail_context_t));
        return NULL;
    }
    
    // Allocate gradient buffers once (reusable across all samples)
    ctx->grad_weights_size = ctx->fc_config->weights_size;
    ctx->grad_bias_size = ctx->fc_config->bias_size;
    ctx->grad_weights = (float*)sim_malloc(ctx->grad_weights_size);
    ctx->grad_bias = (float*)sim_malloc(ctx->grad_bias_size);
    
    if (!ctx->grad_weights || !ctx->grad_bias) {
        log_error("Failed to allocate gradient buffers");
        if (ctx->grad_weights) sim_free_tracked(ctx->grad_weights, ctx->grad_weights_size);
        if (ctx->grad_bias) sim_free_tracked(ctx->grad_bias, ctx->grad_bias_size);
        tensor_free(ctx->conv_features);
        tensor_free(ctx->pooled_features);
        tensor_free(ctx->fc_output);
        tensor_free(ctx->probabilities);
        tensor_free(ctx->grad_fc_output);
        tensor_free(ctx->grad_pooled);
        tensor_free(ctx->grad_conv);
        free_fc_config(ctx->fc_config);
        sim_free_tracked(ctx, sizeof(tail_context_t));
        return NULL;
    }
    
    log_info("Created tail context for device %d: %d classes, %d input channels",
             device_id, num_classes, input_channels);
    
    return ctx;
}

void free_tail_context(tail_context_t* ctx) {
    if (ctx) {
        if (ctx->conv_features) tensor_free(ctx->conv_features);
        if (ctx->pooled_features) tensor_free(ctx->pooled_features);
        if (ctx->fc_output) tensor_free(ctx->fc_output);
        if (ctx->probabilities) tensor_free(ctx->probabilities);
        if (ctx->grad_fc_output) tensor_free(ctx->grad_fc_output);
        if (ctx->grad_pooled) tensor_free(ctx->grad_pooled);
        if (ctx->grad_conv) tensor_free(ctx->grad_conv);
        if (ctx->grad_weights) sim_free_tracked(ctx->grad_weights, ctx->grad_weights_size);
        if (ctx->grad_bias) sim_free_tracked(ctx->grad_bias, ctx->grad_bias_size);
        if (ctx->fc_config) free_fc_config(ctx->fc_config);
        sim_free_tracked(ctx, sizeof(tail_context_t));
    }
}

int tail_forward_pass(tail_context_t* ctx, int true_label) {
    // 1. Dual Pooling (GAP + GMP)
    dual_pooling1d(ctx->conv_features, ctx->pooled_features);
    
    // 2. Fully Connected layer
    fully_connected_forward(ctx->pooled_features, ctx->fc_config, ctx->fc_output);
    
    // 3. Softmax
    softmax_forward(ctx->fc_output, ctx->probabilities);
    
    // 4. Compute loss
    float loss = cross_entropy_loss(ctx->probabilities, &true_label, 1);
    ctx->total_loss += loss;
    
    // 5. Get prediction
    int predicted_label = 0;
    float max_prob = ctx->probabilities->data[0];
    for (int i = 1; i < ctx->num_classes; i++) {
        if (ctx->probabilities->data[i] > max_prob) {
            max_prob = ctx->probabilities->data[i];
            predicted_label = i;
        }
    }
    
    if (predicted_label == true_label) {
        ctx->correct_predictions++;
    }
    
    log_debug("Tail forward: predicted=%d, true=%d, loss=%.4f", 
              predicted_label, true_label, loss);
    
    return predicted_label;
}

void tail_backward_pass(tail_context_t* ctx, int true_label) {
    const float learning_rate = 0.001f;
    // 1. Compute gradient from cross-entropy loss
    cross_entropy_backward(ctx->probabilities, &true_label, 1, ctx->grad_fc_output);
    
    // 2. Backward through fully connected layer
    fully_connected_backward(ctx->grad_fc_output, ctx->pooled_features, 
                            ctx->fc_config, ctx->grad_pooled,
                            ctx->grad_weights, ctx->grad_bias);

    // SGD update on FC weights/bias
    sgd_update(ctx->fc_config->weights, ctx->grad_weights, ctx->fc_config->weights_size, learning_rate);
    sgd_update(ctx->fc_config->bias, ctx->grad_bias, ctx->fc_config->bias_size, learning_rate);
    
    // 3. Backward through dual pooling (GAP + GMP)
    int input_channels = ctx->conv_features->channels;
    int input_length = ctx->conv_features->length;
    
    // Split gradient: first half is GAP, second half is GMP
    for (int c = 0; c < input_channels; c++) {
        float gap_grad = ctx->grad_pooled->data[c] / (float)input_length;
        float gmp_grad = ctx->grad_pooled->data[c + input_channels];
        
        // Find max position for GMP gradient
        int max_pos = 0;
        float max_val = ctx->conv_features->data[c * input_length];
        for (int l = 1; l < input_length; l++) {
            int idx = c * input_length + l;
            if (ctx->conv_features->data[idx] > max_val) {
                max_val = ctx->conv_features->data[idx];
                max_pos = l;
            }
        }
        
        // Apply gradients
        for (int l = 0; l < input_length; l++) {
            int idx = c * input_length + l;
            ctx->grad_conv->data[idx] = gap_grad;  // GAP contribution
            if (l == max_pos) {
                ctx->grad_conv->data[idx] += gmp_grad;  // Add GMP contribution
            }
        }
    }
    
    log_debug("Tail backward: gradients computed");
}

int run_tail_device(int device_id, int num_classes) {
    log_info("Starting Tail Device %d (Classifier) with %d classes", device_id, num_classes);
    
    // Initialize shared memory system
    if (shm_init() < 0) {
        log_error("Failed to initialize shared memory");
        return -1;
    }
    
    // Get configuration from model config
    if (!g_model_config) {
        log_error("Model configuration is required for tail device");
        return -1;
    }
    
    // Find the last conv layer to determine input
    int last_conv_layer_id = -1;
    int total_input_channels = 0;
    
    for (int i = 0; i < g_model_config->num_layers; i++) {
        if (strcmp(g_model_config->layers[i].type, "conv1d") == 0) {
            last_conv_layer_id = i;
            total_input_channels = g_model_config->layers[i].out_channels;
        }
    }
    
    if (last_conv_layer_id < 0) {
        log_error("No convolutional layers found in model config");
        return -1;
    }
    
    log_info("Tail will read from Layer %d output (%d channels)", 
             last_conv_layer_id + 1, total_input_channels);
    
    // Configuration for shared memory-based communication
    int num_workers = g_model_config->layers[last_conv_layer_id].num_devices;
    int max_input_length = g_model_config->input_length;
    
    // Calculate actual input length after all conv layers
    for (int i = 0; i <= last_conv_layer_id; i++) {
        if (strcmp(g_model_config->layers[i].type, "conv1d") == 0) {
            int kernel_size = g_model_config->layers[i].kernel_size;
            int stride = g_model_config->layers[i].stride; 
            int padding = g_model_config->layers[i].padding;
            max_input_length = (max_input_length + 2 * padding - kernel_size) / stride + 1;
        }
    }
    
    // Determine which shared memory layer to read from
    shm_layer_id_t input_layer = (shm_layer_id_t)(last_conv_layer_id + 1);
    
    log_info("Tail reading from shared memory layer %d (%d channels, length %d)", 
             input_layer, total_input_channels, max_input_length);
    
    // Attach to previous layer output shared memory
    if (shm_create_segment(input_layer, total_input_channels, max_input_length, 1) < 0) {
        log_error("Failed to attach to layer %d shared memory", input_layer);
        shm_cleanup();
        return -1;
    }

    // Attach to gradient segment for backpropagation 
    // CORRECTED: Attach to existing segment created by Layer workers
    shm_layer_id_t grad_layer = (shm_layer_id_t)(SHM_GRAD_LAYER0 - last_conv_layer_id);
    
    // Gradient segment sizing: Gradients w.r.t. last layer's OUTPUT (which is Tail's input)
    // Tail writes gradients w.r.t. its input (= last conv layer's output)
    int grad_output_channels = total_input_channels;  // Last layer's OUTPUT channels
    int grad_contributors = 1;  // Only Tail writes to this segment
    
    log_info("Tail: Attaching to existing gradient segment %d with: grad_output_channels=%d, contributors=%d, length=%d",
             grad_layer, grad_output_channels, grad_contributors, max_input_length);
    
    // Attach to existing gradient segment (created by Layer Worker 0)
    if (shm_create_segment(grad_layer, grad_output_channels, max_input_length, grad_contributors) < 0) {
        log_error("Failed to attach to gradient segment %d", grad_layer);
        shm_cleanup();
        return -1;
    }
    log_info("Attached to gradient segment %d for backpropagation", grad_layer);
    
    // Create tail context
    tail_context_t* ctx = create_tail_context(device_id, num_classes, total_input_channels, max_input_length);
    if (!ctx) {
        log_error("Failed to create tail context");
        shm_cleanup();
        return -1;
    }
    
    // Initialize pipeline control
    if (shm_init_pipeline_control() < 0) {
        log_error("Failed to initialize pipeline control");
        free_tail_context(ctx);
        shm_cleanup();
        return -1;
    }
    
    log_info("Tail device ready for classification");
    
    // **MAIN SEQUENTIAL PROCESSING LOOP WITH P2P COORDINATION**
    int expected_sample_id = 1;
    int last_processed_sample_id = 0;
    
    while (1) {
        // **FORWARD PHASE - Wait for full Layer 3 forward completion**
        log_info("Tail: Waiting for input data from Layer %d...", last_conv_layer_id);
        if (shm_wait_for_forward_complete(input_layer) < 0) {
            log_info("Tail: Shutdown during forward wait");
            break;
        }
        
        log_info("Tail: Input data ready, starting classification");
        
        // Check for shutdown
        if (g_pipeline_ctrl && *g_pipeline_ctrl == PHASE_DONE) {
            log_info("Tail: Training complete signal received");
            break;
        }
        
        int current_sample_id = shm_get_current_sample_id(input_layer);
        int current_label = shm_get_current_label(input_layer);
        if (current_sample_id <= last_processed_sample_id) {
            // Avoid reprocessing stale sample when forward_ready remains asserted.
            usleep(50);
            continue;
        }
        if (current_sample_id != expected_sample_id) {
            log_error("Tail: Sample ID mismatch! expected=%d got=%d",
                      expected_sample_id, current_sample_id);
            expected_sample_id = current_sample_id;
        }
        
        // Read features from shared memory
        if (shm_read_tensor(input_layer, ctx->conv_features) < 0) {
            log_error("Tail: Failed to read features tensor");
            break;
        }
        // Tail is the sole consumer of Layer 4 output; consume and clear forward_ready.
        shm_consume_forward_ready(input_layer);
        
        // Log features received from final conv layer
        log_info("Tail: Sample %d, Input[0-4]: %.4f %.4f %.4f %.4f %.4f",
                 current_sample_id,
                 ctx->conv_features->data[0], ctx->conv_features->data[1],
                 ctx->conv_features->data[2], ctx->conv_features->data[3],
                 ctx->conv_features->data[4]);
        
        // Use true label propagated through shared memory
        int true_label = current_label;
        
        // Forward pass
        int predicted_label = tail_forward_pass(ctx, true_label);
        
        // Log prediction results
        log_info("Tail: Sample %d, Predicted=%d, True=%d, Probs[0-2]: %.4f %.4f %.4f",
                 current_sample_id, predicted_label, true_label,
                 ctx->probabilities->data[0], ctx->probabilities->data[1],
                 ctx->probabilities->data[2]);
        
        log_debug("Tail: Forward complete (predicted=%d, true=%d)", predicted_label, true_label);
        
        // Flag will be overwritten by next layer output
        // No need to clear
        
        // **BACKWARD PHASE - P2P: Start immediately (no waiting for signal)**
        log_debug("Tail: Starting backward pass");
        
        // Backward pass
        tail_backward_pass(ctx, true_label);
        
        // Write gradients to final layer gradient segment
        // Map layer ID to correct gradient segment
        shm_layer_id_t final_grad_layer = SHM_PIPELINE_CTRL;  // Invalid default
        switch (last_conv_layer_id) {
            case 0: final_grad_layer = SHM_GRAD_LAYER0; break;
            case 1: final_grad_layer = SHM_GRAD_LAYER1; break;
            case 2: final_grad_layer = SHM_GRAD_LAYER2; break;
            case 3: final_grad_layer = SHM_GRAD_LAYER3; break;
            default:
                log_error("Invalid layer ID %d for gradient mapping (only layers 0-3 supported)", last_conv_layer_id);
                free_tail_context(ctx);
                shm_cleanup();
                return -1;
        }
        
        log_debug("Tail: Mapped last_conv_layer_id=%d to final_grad_layer=%d", 
                 last_conv_layer_id, final_grad_layer);
        
        // Write gradients back to the conv layer
        if (shm_write_tensor(final_grad_layer, 0, ctx->grad_conv) < 0) {
            log_error("Failed to write gradients to layer %d", final_grad_layer);
            break;
        }
        
        // **Signal backward completion - Tail is sole writer**
        if (shm_signal_backward_complete(final_grad_layer, current_sample_id) < 0) {
            log_error("Tail: Failed to signal backward completion");
        }
        log_debug("Tail: Signaled Layer %d gradients ready", last_conv_layer_id);
        
        // Layer 0 worker signals sample completion to Head after its backward pass.
        log_info("Tail: Sample %d backward complete", current_sample_id);
        
        // Log gradients sent back
        log_info("Tail: Sample %d, Gradients[0-4]: %.6f %.6f %.6f %.6f %.6f",
                 current_sample_id,
                 ctx->grad_conv->data[0], ctx->grad_conv->data[1],
                 ctx->grad_conv->data[2], ctx->grad_conv->data[3],
                 ctx->grad_conv->data[4]);
        
        expected_sample_id++;  // Move to next expected sample
        last_processed_sample_id = current_sample_id;
        ctx->samples_processed++;
        
        log_debug("Tail: Completed sample %d, now expecting %d", 
                 expected_sample_id - 1, expected_sample_id);
        
        log_debug("Tail: Backward complete, gradients written");
        
        // Check if training is complete
        if (shm_get_phase() == PHASE_DONE) {
            log_info("Tail: Training complete, exiting");
            break;
        }
        
        // Progress reporting
        if ((expected_sample_id - 1) % 10 == 0) {
            float accuracy = (float)ctx->correct_predictions / ctx->samples_processed * 100.0f;
            float avg_loss = ctx->total_loss / ctx->samples_processed;
            log_info("Tail: %d samples - Accuracy: %.1f%%, Avg Loss: %.4f", 
                     expected_sample_id - 1, accuracy, avg_loss);
        }
    }
    
    log_info("Tail device processing completed");
    
    // Print final statistics
    float final_accuracy = (float)ctx->correct_predictions / ctx->samples_processed * 100.0f;
    float final_avg_loss = ctx->total_loss / ctx->samples_processed;
    log_info("Final Tail Statistics:");
    log_info("  Samples: %d", ctx->samples_processed);
    log_info("  Correct: %d", ctx->correct_predictions);
    log_info("  Accuracy: %.2f%%", final_accuracy);
    log_info("  Avg Loss: %.4f", final_avg_loss);
    
    // Cleanup
    free_tail_context(ctx);
    shm_cleanup();
    
    log_info("Tail device %d completed successfully", device_id);
    return 0;
}
