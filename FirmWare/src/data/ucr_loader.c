#include "nn_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float* data;         // All samples flattened
    int* labels;         // Sample labels
    int num_samples;
    int sample_length;
    int num_classes;
} dataset_t;

// Parse a single line from UCR dataset file
static int parse_ucr_line(const char* line, float* sample, int* label, int sample_length) {
    if (!line || !sample || !label) return -1;
    
    char line_copy[8192];
    strncpy(line_copy, line, sizeof(line_copy) - 1);
    line_copy[sizeof(line_copy) - 1] = '\0';
    
    // Parse comma-separated values
    char* token = strtok(line_copy, ",");
    if (!token) {
        log_error("Failed to parse label from line: %.50s", line);
        return -1;
    }
    
    // First value is the class label
    *label = (int)atoi(token);
    
    // Parse the rest as sample values
    for (int i = 0; i < sample_length; i++) {
        token = strtok(NULL, ",");
        if (!token) {
            log_error("Failed to parse sample value %d", i);
            return -1;
        }
        sample[i] = atof(token);
    }
    
    return 0;
}

// Load UCR dataset from text file
dataset_t* load_ucr_dataset(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        log_error("Failed to open dataset file: %s", filename);
        return NULL;
    }
    
    // First pass: count lines and determine dimensions
    char line[8192];
    int num_lines = 0;
    int sample_length = 0;
    
    while (fgets(line, sizeof(line), file)) {
        if (strlen(line) > 1) {  // Skip empty lines
            num_lines++;
            if (sample_length == 0) {
                // Count comma-separated values in first line to determine sample length
                char line_copy[8192];
                strncpy(line_copy, line, sizeof(line_copy) - 1);
                line_copy[sizeof(line_copy) - 1] = '\0';
                
                char* token = strtok(line_copy, ",");
                token = strtok(NULL, ",");  // Skip label
                while (token != NULL) {
                    sample_length++;
                    token = strtok(NULL, ",");
                }
            }
        }
    }
    
    if (num_lines == 0 || sample_length == 0) {
        log_error("Invalid dataset file: %s", filename);
        fclose(file);
        return NULL;
    }
    
    // Limit samples to fit memory constraint (256KB per device)
    // Each sample: sample_length * 4 bytes + metadata ≈ sample_length * 4.2 bytes  
    int max_samples = MEMORY_LIMIT_BYTES / (sample_length * sizeof(float) * 2);  // Conservative limit
    if (num_lines > max_samples) {
        log_info("Limiting dataset: %d→%d samples to fit memory constraint", num_lines, max_samples);
        num_lines = max_samples;
    }
    
    // Use more samples for actual training (up to 100 samples)
    int training_limit = 100;  // Reasonable limit for distributed training
    if (num_lines > training_limit) {
        log_info("Limiting to %d samples for distributed training", training_limit);
        num_lines = training_limit;
    }
    
    log_info("Dataset: %d samples, %d time points per sample", num_lines, sample_length);
    
    // Allocate dataset structure
    dataset_t* dataset = (dataset_t*)sim_malloc(sizeof(dataset_t));
    if (!dataset) {
        log_error("Failed to allocate dataset structure");
        fclose(file);
        return NULL;
    }
    log_info("Allocated dataset structure");
    
    dataset->num_samples = num_lines;
    dataset->sample_length = sample_length;
    dataset->num_classes = 0;  // Will be calculated
    
    // Calculate memory requirements
    size_t data_size = num_lines * sample_length * sizeof(float);
    size_t label_size = num_lines * sizeof(int);
    log_info("Allocating %.1f KB for data, %.1f KB for labels", 
             data_size / 1024.0f, label_size / 1024.0f);
    
    dataset->data = (float*)sim_malloc(data_size);
    dataset->labels = (int*)sim_malloc(label_size);
    
    if (!dataset->data || !dataset->labels) {
        log_error("Failed to allocate memory for dataset content");
        if (dataset->data) sim_free_tracked(dataset->data, data_size);
        if (dataset->labels) sim_free_tracked(dataset->labels, label_size);
        sim_free_tracked(dataset, sizeof(dataset_t));
        fclose(file);
        return NULL;
    }
    
    log_info("Successfully allocated dataset memory");
    
    // Second pass: load data
    rewind(file);
    int sample_idx = 0;
    int max_label = 0;
    
    while (fgets(line, sizeof(line), file) && sample_idx < num_lines) {
        if (strlen(line) > 1) {
            float* sample_data = &dataset->data[sample_idx * sample_length];
            int label;
            
            if (parse_ucr_line(line, sample_data, &label, sample_length) == 0) {
                dataset->labels[sample_idx] = label;
                if (label > max_label) max_label = label;
                sample_idx++;
            }
        }
    }
    
    dataset->num_classes = max_label + 1;  // Assuming 0-based labels
    
    fclose(file);
    
    log_info("Loaded dataset: %d samples, %d classes, %d time points", 
             dataset->num_samples, dataset->num_classes, dataset->sample_length);
    
    return dataset;
}

void free_dataset(dataset_t* dataset) {
    if (dataset) {
        if (dataset->data) {
            size_t data_size = dataset->num_samples * dataset->sample_length * sizeof(float);
            sim_free_tracked(dataset->data, data_size);
        }
        if (dataset->labels) {
            size_t label_size = dataset->num_samples * sizeof(int);
            sim_free_tracked(dataset->labels, label_size);
        }
        sim_free_tracked(dataset, sizeof(dataset_t));
    }
}

// Get a single sample from the dataset
tensor_t* get_dataset_sample(const dataset_t* dataset, int sample_idx) {
    if (!dataset || sample_idx < 0 || sample_idx >= dataset->num_samples) {
        return NULL;
    }
    
    // Create tensor for single sample: batch=1, channels=1, length=sample_length
    tensor_t* tensor = tensor_create(1, 1, dataset->sample_length);
    if (!tensor) {
        return NULL;
    }
    
    // Copy sample data
    const float* sample_data = &dataset->data[sample_idx * dataset->sample_length];
    memcpy(tensor->data, sample_data, dataset->sample_length * sizeof(float));
    
    return tensor;
}

int get_dataset_label(const dataset_t* dataset, int sample_idx) {
    if (!dataset || sample_idx < 0 || sample_idx >= dataset->num_samples) {
        return -1;
    }
    
    return dataset->labels[sample_idx];
}

// Simple Z-score normalization
void normalize_dataset(dataset_t* dataset) {
    if (!dataset || !dataset->data) return;
    
    int total_elements = dataset->num_samples * dataset->sample_length;
    
    // Calculate mean
    double sum = 0.0;
    for (int i = 0; i < total_elements; i++) {
        sum += dataset->data[i];
    }
    float mean = (float)(sum / total_elements);
    
    // Calculate standard deviation
    double variance_sum = 0.0;
    for (int i = 0; i < total_elements; i++) {
        double diff = dataset->data[i] - mean;
        variance_sum += diff * diff;
    }
    float std_dev = (float)sqrt(variance_sum / total_elements);
    
    // Normalize
    if (std_dev > 1e-8) {  // Avoid division by zero
        for (int i = 0; i < total_elements; i++) {
            dataset->data[i] = (dataset->data[i] - mean) / std_dev;
        }
    }
    
    log_info("Normalized dataset: mean=%.6f, std=%.6f", mean, std_dev);
}