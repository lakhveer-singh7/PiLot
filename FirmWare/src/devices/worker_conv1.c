#include "lw_pilot_sim.h"
#include "nn_types.h"
#include "comm_types.h"
#include "config_types.h"
#include "worker_threads.h"
#include "shared_memory.h"

// External configuration
extern device_json_config_t* g_device_config;
extern model_config_t* g_model_config;
extern volatile pipeline_phase_t* g_pipeline_ctrl;  // For shutdown detection

// External runtime parameters (from command line)
extern int g_upstream_port;
extern int g_downstream_port;
extern int g_backward_port;
extern int g_backward_connect_id;
extern int g_backward_connect_port;
extern int g_in_channels;
extern int g_out_channels;
extern int g_kernel_size;
extern int g_stride;
extern int g_padding;
extern int g_layer_id;
extern int g_worker_id;
extern int g_num_workers;

// Helper function to get count of downstream workers that write gradients to this layer
static int get_downstream_worker_count(int layer_id) {
    switch (layer_id) {
        case 0: return 2;  // Layer 1 has 2 workers
        case 1: return 3;  // Layer 2 has 3 workers  
        case 2: return 4;  // Layer 3 has 4 workers
        case 3: return 1;  // Tail device
        default:
            log_error("Invalid layer_id %d for downstream count", layer_id);
            return 1;  // Safe fallback
    }
}

static int get_current_layer_worker_count(int layer_id) {
    switch (layer_id) {
        case 0: return 1;  // Layer 0 has 1 worker
        case 1: return 2;  // Layer 1 has 2 workers  
        case 2: return 3;  // Layer 2 has 3 workers
        case 3: return 4;  // Layer 3 has 4 workers
        default:
            log_error("Invalid layer_id %d for current layer count", layer_id);
            return 1;  // Safe fallback
    }
}

int run_worker_device(int device_id) {
    log_info("Starting Worker Device %d (Conv1D processor)", device_id);
    
    // Shared memory parameters (required)
    int layer_id = g_layer_id;  // Which layer (0-4)
    int worker_id = g_worker_id;  // Which worker within layer (0-N)
    int num_workers = g_num_workers;  // Total workers in this layer
    
    if (layer_id < 0 || worker_id < 0 || num_workers <= 0) {
        log_error("Shared memory parameters required: --layer-id, --worker-id, --num-workers");
        return -1;
    }
    
    // Get layer configuration from model config
    if (!g_model_config) {
        log_error("Model configuration is required for worker devices");
        return -1;
    }
    
    // Get input channels from model config based on layer_id
    int in_channels = get_layer_input_channels(g_model_config, layer_id);
    if (in_channels < 0) {
        log_error("Invalid layer_id %d for model configuration", layer_id);
        return -1;
    }
    
    // Get layer parameters from model config  
    int out_channels = 16;  // Each worker outputs 16 channels (uniform)
    int kernel_size = 5;
    int stride = 1; 
    int padding = 2;
    
    if (layer_id < g_model_config->num_layers && 
        strcmp(g_model_config->layers[layer_id].type, "conv1d") == 0) {
        // Use model config parameters
        kernel_size = g_model_config->layers[layer_id].kernel_size;
        stride = g_model_config->layers[layer_id].stride;
        padding = g_model_config->layers[layer_id].padding;
        
        // Total output channels for layer divided by number of workers
        int total_out_channels = g_model_config->layers[layer_id].out_channels;
        int layer_num_workers = g_model_config->layers[layer_id].num_devices;
        out_channels = total_out_channels / layer_num_workers;
    }
    
    // Allow command-line overrides
    if (g_in_channels >= 0) in_channels = g_in_channels;
    if (g_out_channels >= 0) out_channels = g_out_channels;
    if (g_kernel_size >= 0) kernel_size = g_kernel_size;
    if (g_stride >= 0) stride = g_stride;
    if (g_padding >= 0) padding = g_padding;
    
    log_info("Worker configuration: Layer %d, Worker %d/%d, Conv1D %d→%d channels (kernel=%d, stride=%d, padding=%d)",
             layer_id, worker_id, num_workers, in_channels, out_channels, kernel_size, stride, padding);
    
    // Initialize shared memory system
    if (shm_init() < 0) {
        log_error("Failed to initialize shared memory system");
        return -1;
    }
    
    // Calculate dimensions for input (from previous layer) and output
    int input_length = 300;  // Initial input length from HEAD
    int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;
    
    // For layer_id > 0, input channels come from previous layer's total output
    // For now, assume previous layer has same number of workers outputting 16 channels each
    // Layer 0 (HEAD): 1 channel  
    // Layer 1 (1 worker): 16 channels from 1 worker = 16
    // Layer 2 (2 workers): 16 channels from 2 workers = 32
    // etc.
    
    // Attach to input layer's shared memory (read-only access)
    shm_layer_id_t input_layer = (shm_layer_id_t)layer_id;
    if (shm_create_segment(input_layer, in_channels, input_length, 1) < 0) {
        log_error("Failed to attach to input layer %d shared memory", layer_id);
        shm_cleanup();
        return -1;
    }
    
    // Create output layer's shared memory segment  
    // Total output channels = num_workers * out_channels (16 channels per worker)
    shm_layer_id_t output_layer = (shm_layer_id_t)(layer_id + 1);
    int total_output_channels = num_workers * out_channels;
    if (shm_create_segment(output_layer, total_output_channels, output_length, num_workers) < 0) {
        log_error("Failed to create output layer %d shared memory", layer_id + 1);
        shm_cleanup();
        return -1;
    }
    
    // Create gradient shared memory segment for backward pass
    // Map layer ID to correct gradient segment
    shm_layer_id_t grad_segment;
    switch (layer_id) {
        case 0: grad_segment = SHM_GRAD_LAYER0; break;
        case 1: grad_segment = SHM_GRAD_LAYER1; break;
        case 2: grad_segment = SHM_GRAD_LAYER2; break;
        case 3: grad_segment = SHM_GRAD_LAYER3; break;
        default:
            log_error("Invalid layer_id %d for gradient mapping", layer_id);
            shm_cleanup();
            return -1;
    }
    
    // **CORRECTED GRADIENT SIZING: Only Worker 0 creates, others attach**
    // Gradient segment for THIS layer receives gradients from DOWNSTREAM layer workers
    // Contributors = number of workers in the NEXT layer (who write gradients back to this layer)
    
    int downstream_workers = get_downstream_worker_count(layer_id);
    int gradient_channels = total_output_channels;  // **CORRECTED: Use TOTAL layer output channels, not per-worker**
    int current_layer_workers = get_current_layer_worker_count(layer_id);  // Workers in THIS layer
    int gradient_contributors = downstream_workers;  // How many DOWNSTREAM workers write gradients to THIS layer
    
    log_info("Worker %d: %s gradient segment %d with: total_out_channels=%d, contributors=%d, length=%d", 
             device_id, (worker_id == 0) ? "Creating" : "Attaching to", grad_segment, 
             gradient_channels, gradient_contributors, output_length);
    
    // Only Worker 0 creates gradient segment, others attach to existing
    if (shm_create_segment(grad_segment, gradient_channels, output_length, gradient_contributors) < 0) {
        log_error("Failed to create gradient shared memory for layer %d", layer_id);
        shm_cleanup();
        return -1;
    }
    
    // **CRITICAL FIX: Workers also need to attach to PREVIOUS layer gradient segment to write gradients**
    if (layer_id > 0) {
        shm_layer_id_t prev_grad_segment;
        switch (layer_id - 1) {
            case 0: prev_grad_segment = SHM_GRAD_LAYER0; break;
            case 1: prev_grad_segment = SHM_GRAD_LAYER1; break;
            case 2: prev_grad_segment = SHM_GRAD_LAYER2; break;
            case 3: prev_grad_segment = SHM_GRAD_LAYER3; break;
            default:
                log_error("Invalid previous layer_id %d for gradient mapping", layer_id - 1);
                shm_cleanup();
                return -1;
        }
        
        // Previous layer dimensions (where we write gradients to)
        // CORRECTED: Match the gradient segment the previous layer created
        int prev_layer_id = layer_id - 1;
        int prev_out_channels = get_layer_output_channels(g_model_config, prev_layer_id);  // **Previous layer's OUTPUT channels**
        int prev_contributors = get_downstream_worker_count(prev_layer_id);  // How many downstream workers (us) write gradients
        int prev_length = input_length;   // Previous layer's output length
        
        log_info("Worker %d: Attaching to prev gradient segment %d with: out_channels=%d, contributors=%d, length=%d",
                 device_id, prev_grad_segment, prev_out_channels, prev_contributors, prev_length);
        
        // Attach to previous layer gradient segment (write-only) - use SAME parameters as that layer used to create it
        if (shm_create_segment(prev_grad_segment, prev_out_channels, prev_length, prev_contributors) < 0) {
            log_error("Failed to attach to previous layer gradient segment %d", prev_grad_segment);
            shm_cleanup();
            return -1;
        }
        
        log_info("Worker attached to previous layer gradient segment %d for backward pass", prev_grad_segment);
    }

    log_info("Worker shared memory initialized successfully (forward + backward)");
    
    // **NEW: Sequential processing instead of threading**
    
    // Initialize pipeline control  
    if (shm_init_pipeline_control() < 0) {
        log_error("Failed to initialize pipeline control");
        shm_cleanup();
        return -1;
    }
    
    // Create worker context for Conv1D processing
    worker_context_t* ctx = create_worker_context(device_id, layer_id, in_channels, out_channels, 
                                                   kernel_size, stride, padding);
    if (!ctx) {
        log_error("Failed to create worker context");
        shm_cleanup();
        return -1;
    }
    
    // Set shared memory parameters
    ctx->layer_id = layer_id;
    ctx->worker_id = worker_id;
    ctx->num_workers = num_workers;
    
    log_info("Worker device configured: %d→%d Conv1D layer (sequential mode)", in_channels, out_channels);
    
    // **MAIN SEQUENTIAL TRAINING LOOP WITH SAMPLE ID COORDINATION**
    int expected_sample_id = 1;
    while (1) {
        // **FORWARD PHASE - P2P: Wait for correct sample ID**
        
        log_debug("Worker %d (Layer %d): Waiting for sample %d...", device_id, layer_id, expected_sample_id);
        
        // **Wait for ALL upstream workers to complete writing**
        log_debug("Worker %d: Waiting for all upstream workers to complete forward write", device_id);
        if (shm_wait_for_forward_complete(input_layer) < 0) {
            log_info("Worker %d: Shutdown during forward wait", device_id);
            break;
        }
        
        // **Check sample ID after all writes complete**
        int current_sample = shm_get_current_sample_id(input_layer);
        if (current_sample != expected_sample_id) {
            log_error("Worker %d: Sample ID mismatch! Expected %d, got %d", 
                     device_id, expected_sample_id, current_sample);
            // Continue anyway to maintain pipeline flow
        }
        
        log_debug("Worker %d (Layer %d): All upstream complete, processing sample %d", 
                 device_id, layer_id, expected_sample_id);
        
        // **Read COMPLETE input from previous layer**
        // All workers in this layer receive the same complete input
        if (shm_read_complete_layer_output(input_layer, ctx->forward_input) < 0) {
            log_error("Worker %d: Failed to read complete input tensor", device_id);
            break;
        }
        
        // **Clear forward_ready after reading (each worker clears independently)**
        shm_clear_forward_ready(input_layer);
        
        // Set sample ID in input tensor
        ctx->forward_input->sample_id = expected_sample_id;
        
        // Log input data received
        log_info("Worker %d (Layer %d): Sample %d, Input[0-4]: %.4f %.4f %.4f %.4f %.4f",
                 device_id, layer_id, expected_sample_id,
                 ctx->forward_input->data[0], ctx->forward_input->data[1],
                 ctx->forward_input->data[2], ctx->forward_input->data[3],
                 ctx->forward_input->data[4]);
        
        // Forward computation: Conv1D + GroupNorm + ReLU
        conv1d_forward(ctx->forward_input, ctx->conv_config, ctx->forward_temp);
        
        // Group normalization  
        int num_groups = 8;
        if (out_channels % num_groups != 0) num_groups = 4;
        if (out_channels % num_groups != 0) num_groups = 1;
        group_norm_forward(ctx->forward_temp, ctx->forward_output, num_groups);
        
        // ReLU activation
        relu_forward(ctx->forward_output, ctx->forward_output);
        
        // **CRITICAL: Set sample ID in output tensor for next layer**
        ctx->forward_output->sample_id = expected_sample_id;
        
        // Write output to shared memory at worker offset
        if (shm_write_tensor(output_layer, worker_id, ctx->forward_output) < 0) {
            log_error("Worker %d: Failed to write forward output", device_id);
            break;
        }
        
        // Log output data produced
        log_info("Worker %d (Layer %d): Sample %d, Output[0-4]: %.4f %.4f %.4f %.4f %.4f",
                 device_id, layer_id, expected_sample_id,
                 ctx->forward_output->data[0], ctx->forward_output->data[1],
                 ctx->forward_output->data[2], ctx->forward_output->data[3],
                 ctx->forward_output->data[4]);
        
        log_debug("Worker %d: Forward complete, sample %d", device_id, expected_sample_id);
        
        // **Signal completion - last worker will set forward_ready and update sample_id**
        if (shm_signal_forward_complete(output_layer, expected_sample_id) < 0) {
            log_error("Worker %d: Failed to signal forward completion", device_id);
            break;
        }
        
        log_info("Worker %d (Layer %d): Sample %d forward signaled",
                 device_id, layer_id, expected_sample_id);
        
        // **BACKWARD PHASE - P2P: Wait for downstream layer gradients**
        
        // Map layer ID to correct gradient segment
        shm_layer_id_t grad_layer;
        switch (layer_id) {
            case 0: grad_layer = SHM_GRAD_LAYER0; break;
            case 1: grad_layer = SHM_GRAD_LAYER1; break;
            case 2: grad_layer = SHM_GRAD_LAYER2; break;
            case 3: grad_layer = SHM_GRAD_LAYER3; break;
            default:
                log_error("Worker %d: Invalid layer_id %d", device_id, layer_id);
                grad_layer = SHM_GRAD_LAYER0;
                break;
        }
        
        log_debug("Worker %d: Waiting for all downstream workers to complete gradient write...", device_id);
        
        // **Wait for ALL downstream workers to complete writing gradients**
        if (shm_wait_for_backward_complete(grad_layer) < 0) {
            log_info("Worker %d: Shutdown during backward wait", device_id);
            break;
        }
        
        log_debug("Worker %d: All downstream gradients complete, starting backward", device_id);
        
        // **CORRECTED: Read and aggregate gradients from ALL downstream workers**
        if (shm_read_aggregated_gradients(grad_layer, ctx->backward_grad_output) < 0) {
            log_error("Worker %d: Failed to read aggregated gradients", device_id);
            break;
        }
        
        // **Clear backward_ready after reading (each worker clears independently)**
        shm_clear_backward_ready(grad_layer);
        
        // Log gradients received
        log_info("Worker %d (Layer %d): Sample %d, GradIn[0-4]: %.6f %.6f %.6f %.6f %.6f",
                 device_id, layer_id, expected_sample_id,
                 ctx->backward_grad_output->data[0], ctx->backward_grad_output->data[1],
                 ctx->backward_grad_output->data[2], ctx->backward_grad_output->data[3],
                 ctx->backward_grad_output->data[4]);
        
        // For Layer 0: Signal sample completion after backward pass
        if (layer_id == 0) {
            shm_signal_sample_complete(expected_sample_id);
            log_info("Worker %d: Signaled sample %d completion to Head", device_id, expected_sample_id);
        }
        
        // Simplified backward computation (mock gradients for now)
        // Full implementation would compute weight gradients and input gradients
        if (layer_id > 0) {
            // **CORRECTED: Use backward_grad_input dimensions, not forward_input**
            // Create mock gradients for previous layer (scaled input)
            int grad_elements = ctx->backward_grad_input->channels * ctx->backward_grad_input->length;
            for (int i = 0; i < grad_elements; i++) {
                // Use modulo to safely handle dimension differences
                int input_idx = i % (ctx->forward_input->channels * ctx->forward_input->length);
                ctx->backward_grad_input->data[i] = ctx->forward_input->data[input_idx] * 0.01f;
            }
            
            log_debug("Worker %d: Generated %d gradient elements from %d input elements", 
                     device_id, grad_elements, ctx->forward_input->channels * ctx->forward_input->length);
            
            // Map previous layer ID to gradient segment
            shm_layer_id_t prev_grad_layer;
            switch (layer_id - 1) {
                case 0: prev_grad_layer = SHM_GRAD_LAYER0; break;
                case 1: prev_grad_layer = SHM_GRAD_LAYER1; break;
                case 2: prev_grad_layer = SHM_GRAD_LAYER2; break;
                case 3: prev_grad_layer = SHM_GRAD_LAYER3; break;
                default:
                    log_error("Worker %d: Invalid prev layer_id %d", device_id, layer_id - 1);
                    break;
            }
            
            // **CORRECTED: Convert device_id to contributor index for gradient writing**
            // Layer 0: device 1 → contributor 0
            // Layer 1: devices 2,3 → contributors 0,1  
            // Layer 2: devices 4,5,6 → contributors 0,1,2
            // Layer 3: devices 7,8,9,10 → contributors 0,1,2,3
            int contributor_index = worker_id;  // Use worker_id (0-based) instead of device_id
            
            log_debug("Worker %d (worker_id=%d): Writing gradient contribution %d to prev layer %d", 
                      device_id, worker_id, contributor_index, prev_grad_layer);
            
            // **CORRECTED: Write gradient contribution to previous layer**
            // Each worker writes complete gradients w.r.t. the input it received
            if (shm_write_gradient_contribution(prev_grad_layer, contributor_index, ctx->backward_grad_input) < 0) {
                log_error("Worker %d: Failed to write gradient contribution", device_id);
                // Continue processing instead of breaking - allows forward pipeline to continue
                log_info("Worker %d: Continuing without gradient contribution for sample %d", device_id, expected_sample_id);
            } else {
                log_debug("Worker %d: Successfully wrote gradient contribution for sample %d", device_id, expected_sample_id);
            }
            
            log_debug("Worker %d: Wrote gradient contribution to layer %d", device_id, layer_id - 1);
            
            // **Signal completion - last worker will set backward_ready**
            if (shm_signal_backward_complete(prev_grad_layer, expected_sample_id) < 0) {
                log_error("Worker %d: Failed to signal backward completion", device_id);
            }
        }
        
        log_debug("Worker %d: Backward complete", device_id);
        
        // Check if training epoch is complete
        if (g_pipeline_ctrl && *g_pipeline_ctrl == PHASE_DONE) {
            log_info("Worker %d: Epoch complete, exiting", device_id);
            break;
        }
        
        // Move to next sample
        log_debug("Worker %d: Moving to next sample %d", device_id, expected_sample_id + 1);
        expected_sample_id++;
        
        log_debug("Worker %d: Starting loop iteration for sample %d", device_id, expected_sample_id);
    }
    
    log_info("Worker device processing completed");
    
    // Print statistics
    log_info("Worker %d statistics: %d samples processed", device_id, expected_sample_id - 1);
    
    // Cleanup
    free_worker_context(ctx);
    shm_cleanup();
    
    log_info("Worker device %d completed successfully", device_id);
    return 0;
}