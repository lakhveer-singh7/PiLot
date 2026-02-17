#include "lw_pilot_sim.h"
#include "nn_types.h"
#include "comm_types.h"
#include "config_types.h"
#include "shared_memory.h"

// Forward declarations from ucr_loader.c
typedef struct {
    float* data;
    int* labels;
    int num_samples;
    int sample_length;
    int num_classes;
} dataset_t;

dataset_t* load_ucr_dataset(const char* filename);
void free_dataset(dataset_t* dataset);
tensor_t* get_dataset_sample(const dataset_t* dataset, int sample_idx);
int get_dataset_label(const dataset_t* dataset, int sample_idx);
void normalize_dataset(dataset_t* dataset);

// External configuration
extern device_json_config_t* g_device_config;
extern model_config_t* g_model_config;

 int run_head_device(int device_id, const char* dataset_name) {
    log_info("Starting Head Device %d with dataset: %s", device_id, dataset_name);
    
    // Initialize shared memory system
    if (shm_init() < 0) {
        log_error("Failed to initialize shared memory");
        return -1;
    }
    
    // Get Layer 1 configuration from config
    int num_layer1_workers = 1;
    int layer1_channels = 16;  // Default
    int input_length = 300;    // Default
    
    if (g_device_config && g_device_config->num_downstream > 0) {
        num_layer1_workers = g_device_config->num_downstream;
        log_info("Config-based setup: %d Layer 1 workers", num_layer1_workers);
        
        // Each worker outputs fixed channels (e.g., 16)
        // Total channels for layer 1 = num_workers * channels_per_worker
        layer1_channels = num_layer1_workers * 16;
    } else {
        log_info("Using default configuration: 1 worker, 16 channels");
    }
    
    // Create shared memory for Layer 0 (head output = input data)
    // Head writes raw samples here for Layer 1 workers to read
    //shm_layer_id_t layer_id, int channels, int length, int num_workers
    if (shm_create_segment(SHM_LAYER0_INPUT, 1, input_length, 1) < 0) {
        log_error("Failed to create shared memory for Layer 0");
        return -1;
    }
    
    log_info("Shared memory initialized successfully for head device");
    
    // Initialize pipeline control shared memory segment only
    // Gradient segments will be created by worker layers when needed
    if (shm_create_segment(SHM_PIPELINE_CTRL, 1, 4, 1) < 0) {
        log_error("Failed to create pipeline control segment");
        shm_cleanup();
        return -1;
    }
    
    // Load dataset
    char dataset_filename[512];
    snprintf(dataset_filename, sizeof(dataset_filename), "/mnt/d/New folder/UCR_DATASETS/%s/%s_TRAIN", dataset_name, dataset_name);
    
    dataset_t* dataset = load_ucr_dataset(dataset_filename);
    if (!dataset) {
        log_error("Failed to load dataset: %s", dataset_filename);
        shm_cleanup();
        return -1;
    }
    
    // Normalize dataset
    normalize_dataset(dataset);
    
    log_info("Dataset loaded: %d samples, %d time points, %d classes",
             dataset->num_samples, dataset->sample_length, dataset->num_classes);
    
    // Check memory constraint
    size_t dataset_memory = dataset->num_samples * dataset->sample_length * sizeof(float) +
                           dataset->num_samples * sizeof(int);
    log_info("Dataset memory usage: %.1f KB", dataset_memory / 1024.0f);
    
    if (dataset_memory > MEMORY_LIMIT_BYTES * 0.8) {  // Use 80% of limit for safety
        log_error("Dataset too large for memory constraint (%.1f KB > %.1f KB)",
                  dataset_memory / 1024.0f, (MEMORY_LIMIT_BYTES * 0.8) / 1024.0f);
        free_dataset(dataset);
        shm_cleanup();
        return -1;
    }
    
    int total_samples = dataset->num_samples;
    int total_rounds = 500;
    
    log_info("Starting round-based training with %d samples per round (total rounds: %d)", total_samples, total_rounds);
    
    // **NEW: Initialize pipeline control**
    if (shm_init_pipeline_control() < 0) {
        log_error("Failed to initialize pipeline control");
        free_dataset(dataset);
        shm_cleanup();
        return -1;
    }
    
    log_info("Head will feed samples sequentially and wait for complete pipeline processing");
    
    // **STARTUP SYNCHRONIZATION**: Wait for all devices to initialize
    log_info("Head: Waiting for all devices to initialize...");
    sleep(6);  // Give workers and tail 6 seconds to initialize (tail starts last at ~5s)
    log_info("Head: All devices should be ready, starting sequential training");
    
    // **MAIN SEQUENTIAL SAMPLE FEEDING LOOP**
    int global_sample_count = 0;
    int max_samples = total_rounds;  // One sample per round
    
    log_info("Head: Will process %d total samples (round-based)", max_samples);
    
    while (global_sample_count < max_samples) {
        // Get sample (cycle through dataset)
        int dataset_idx = global_sample_count % total_samples;
        tensor_t* sample = get_dataset_sample(dataset, dataset_idx);
        if (!sample) {
            log_error("Failed to get sample %d", dataset_idx);
            global_sample_count++;
            continue;
        }
        
        int label = get_dataset_label(dataset, dataset_idx);
        
        // **WAIT FOR PREVIOUS SAMPLE PIPELINE COMPLETION**
        if (global_sample_count > 0) {
            log_info("Head: Waiting for complete pipeline processing of sample %d...", global_sample_count);
            if (shm_wait_for_sample_completion(global_sample_count) < 0) {
                log_error("Head: Failed to wait for sample %d completion", global_sample_count);
                tensor_free(sample);
                break;
            }
            log_info("Head: Pipeline completed sample %d", global_sample_count);
        }
        
        // **WRITE NEW SAMPLE TO PIPELINE WITH SAMPLE ID**
        global_sample_count++; // Increment before setting sample ID
        sample->sample_id = global_sample_count;  // Use incremented count as sample ID
        shm_set_current_sample_id(SHM_LAYER0_INPUT, sample->sample_id);
        shm_set_current_label(SHM_LAYER0_INPUT, label);
        
        // **WRITE NEW SAMPLE TO PIPELINE**
        if (shm_write_tensor(SHM_LAYER0_INPUT, 0, sample) < 0) {
            log_error("Failed to write sample to shared memory");
            tensor_free(sample);
           //---->>>> global_sample_count++;
            continue;
        }
        
        // **INCREMENT SAMPLES SENT COUNTER**
        int sent_count = shm_increment_samples_sent();
        if (sent_count != global_sample_count) {
            log_debug("Head: Sample counter mismatch (sent=%d, expected=%d)", sent_count, global_sample_count);
        }
        
        // Log current sample info
        log_info("Head: Sample %d (dataset[%d]), Label=%d, Input[0-4]: %.4f %.4f %.4f %.4f %.4f",
                 global_sample_count, dataset_idx, label,
                 sample->data[0], sample->data[1], sample->data[2], 
                 sample->data[3], sample->data[4]);
        
        // **SIGNAL PIPELINE TO START PROCESSING**
        // Head is the sole writer to Layer 0, so it immediately completes
        if (shm_signal_forward_complete_with_label(SHM_LAYER0_INPUT, global_sample_count, label) < 0) {
            log_error("Head: Failed to signal forward completion for sample %d", global_sample_count);
            tensor_free(sample);
            break;
        }
        log_debug("Head: Signaled pipeline to start processing sample %d", global_sample_count);
        
        tensor_free(sample);
        // global_sample_count already incremented above
        
        // Progress reporting
        if (global_sample_count % 10 == 0) {
            log_info("Head: Processed %d/%d rounds", global_sample_count, total_rounds);
        }
    }
    
    // Signal training complete
    shm_set_phase(PHASE_DONE);
    log_info("Training completed successfully!");

cleanup:
    free_dataset(dataset);
    shm_cleanup();
    
    log_info("Head device finished");
    return 0;
}
