# Shared Memory Communication - Code Examples

This document provides practical code examples demonstrating key patterns in the PiLot shared memory communication system.

## 1. Creating a Shared Memory Segment

```c
// Example: Create shared memory segment for Layer 1 input (16 channels, 300 length, 2 workers)
#include "shared_memory.h"

int create_layer1_segment(void) {
    shm_layer_id_t layer_id = SHM_LAYER1_INPUT;
    int channels = 16;
    int length = 300;
    int num_workers = 2;  // Layer 1 has 2 workers
    
    // Create or attach to shared memory segment
    if (shm_create_segment(layer_id, channels, length, num_workers) < 0) {
        log_error("Failed to create Layer 1 shared memory segment");
        return -1;
    }
    
    log_info("Layer 1 segment created: %d channels × %d length for %d workers", 
             channels, length, num_workers);
    
    return 0;
}
```

**What happens under the hood:**
1. Calculates segment size: `sizeof(shm_header_t) + (16 × 300 × sizeof(float))` = 64 + 19,200 = 19,264 bytes
2. Creates POSIX shared memory with key `0x1235` (0x1234 + layer_id)
3. Maps segment into process address space with `shmat()`
4. Initializes header fields (ready flags, counters) to zero
5. Creates POSIX barriers for 2 workers (if num_workers > 1)

## 2. Writing Data to Shared Memory (Producer)

```c
// Example: Layer 0 worker writes its output to shared memory for Layer 1 to read
#include "shared_memory.h"

int layer0_write_output(tensor_t* output) {
    shm_layer_id_t layer_id = SHM_LAYER1_INPUT;  // Layer 0 writes to Layer 1 input
    int worker_id = 0;  // Layer 0 has only 1 worker
    
    // Write tensor to shared memory
    if (shm_write_tensor(layer_id, worker_id, output) < 0) {
        log_error("Failed to write Layer 0 output");
        return -1;
    }
    
    log_info("Layer 0 wrote %d channels to shared memory", output->channels);
    
    // Signal completion (atomic increment + set ready flag if last worker)
    shm_signal_forward_complete(layer_id, output->sample_id);
    
    return 0;
}
```

**Memory layout after write:**
```
Shared Memory Segment (SHM_LAYER1_INPUT):
┌─────────────────────────────────────┐
│ shm_header_t (64 bytes)             │
│  - forward_ready = 1 (after signal) │
│  - forward_contributors_completed=1 │
├─────────────────────────────────────┤
│ Worker 0 Output (16 × 300 floats)  │  ← Layer 0 output written here
│ [ch0_t0, ch0_t1, ..., ch15_t299]   │
└─────────────────────────────────────┘
```

## 3. Reading Data from Shared Memory (Consumer)

```c
// Example: Layer 1 workers read their input from shared memory
#include "shared_memory.h"

int layer1_read_input(tensor_t* input, int layer_id_value) {
    shm_layer_id_t layer_id = SHM_LAYER1_INPUT;
    
    // Wait for data to be ready (polling on forward_ready flag)
    shm_wait_for_forward_ready(layer_id);
    
    // Read entire layer output (all workers read the same data)
    if (shm_read_tensor(layer_id, input) < 0) {
        log_error("Failed to read Layer 1 input");
        return -1;
    }
    
    log_info("Layer 1 worker read %d channels from shared memory", input->channels);
    
    // Clear ready flag (consumers mark data as consumed)
    shm_clear_forward_ready(layer_id);
    
    return 0;
}
```

**Wait loop under the hood:**
```c
void shm_wait_for_forward_ready(shm_layer_id_t layer_id) {
    shm_segment_t* seg = &g_shm_manager.layers[layer_id];
    
    // Busy-wait polling (with shutdown detection)
    while (__sync_fetch_and_add(&seg->header->forward_ready, 0) == 0) {
        usleep(100);  // Sleep 100 microseconds between checks
        
        // Check for shutdown signal
        if (*g_pipeline_ctrl == PHASE_DONE) {
            return;  // Abort wait
        }
    }
}
```

## 4. Multi-Worker Coordination with Barriers

```c
// Example: Layer 2 workers coordinate using barriers
#include "shared_memory.h"

int layer2_worker_process(tensor_t* input, tensor_t* output, int worker_id) {
    shm_layer_id_t input_layer = SHM_LAYER2_INPUT;
    shm_layer_id_t output_layer = SHM_LAYER3_INPUT;
    
    // Step 1: All workers read shared input
    shm_read_tensor(input_layer, input);
    
    // Step 2: Perform Conv1D computation (independent for each worker)
    conv1d_forward(input, output, worker_id);  // Each worker processes 32→16 channels
    
    // Step 3: Each worker writes its output to disjoint channel range
    shm_write_tensor(output_layer, worker_id, output);
    
    // Step 4: Wait for ALL workers to finish writing (barrier synchronization)
    shm_barrier_forward(output_layer);
    
    // Step 5: Last worker sets ready flag (atomic counter + conditional set)
    shm_signal_forward_complete(output_layer, input->sample_id);
    
    log_info("Layer 2 Worker %d completed forward pass", worker_id);
    
    return 0;
}
```

**Barrier behavior:**
```
Worker 0: writes channels 0-15    → barrier_wait() → BLOCKS
Worker 1: writes channels 16-31   → barrier_wait() → BLOCKS
Worker 2: writes channels 32-47   → barrier_wait() → ALL RELEASED
```

After barrier, atomic counter logic:
```c
int completed = __sync_add_and_fetch(&forward_contributors_completed, 1);
// Worker 0: completed = 1 (not last, return)
// Worker 1: completed = 2 (not last, return)
// Worker 2: completed = 3 (LAST! set forward_ready=1)
```

## 5. Pipeline Phase Coordination

```c
// Example: Head device coordinates pipeline phases
#include "shared_memory.h"

void head_coordinate_forward_pass(tensor_t* sample) {
    // Write sample to Layer 0 input
    shm_write_tensor(SHM_LAYER0_INPUT, 0, sample);
    
    // Signal Layer 0 to start processing
    shm_set_phase(PHASE_FORWARD_LAYER0);
    log_info("Head: Signaled PHASE_FORWARD_LAYER0");
    
    // Wait for Layer 0 to complete
    shm_wait_layer_forward_complete(SHM_LAYER0_INPUT);
    log_info("Head: Layer 0 completed");
    
    // Signal Layer 1 to start processing
    shm_set_phase(PHASE_FORWARD_LAYER1);
    log_info("Head: Signaled PHASE_FORWARD_LAYER1");
    
    // Wait for Layer 1 to complete
    shm_wait_layer_forward_complete(SHM_LAYER1_INPUT);
    log_info("Head: Layer 1 completed");
    
    // Continue for Layer 2, 3...
}

// Example: Worker waits for its phase
void layer1_worker_wait_for_phase(void) {
    pipeline_phase_t my_phase = PHASE_FORWARD_LAYER1;
    
    log_info("Layer 1 Worker: Waiting for PHASE_FORWARD_LAYER1...");
    shm_wait_for_phase(my_phase);
    log_info("Layer 1 Worker: Phase reached, starting computation");
    
    // Proceed with forward pass
}
```

**Phase progression timeline:**
```
Time →  Head Device         Layer 0 Worker      Layer 1 Workers
0ms:    set_phase(L0)    →  wait_phase(L0)      wait_phase(L1)
10ms:   wait_complete()     ✓ phase match       (still waiting)
20ms:   (waiting...)        process_data()      (still waiting)
30ms:   (waiting...)        write_output()      (still waiting)
40ms:   ✓ complete          signal_complete()   (still waiting)
41ms:   set_phase(L1)    →  (idle)           →  ✓ phase match
42ms:   wait_complete()     (idle)              process_data()
...
```

## 6. Gradient Aggregation (Backward Pass)

```c
// Example: Layer 2 worker aggregates gradients from 4 Layer 3 workers
#include "shared_memory.h"

int layer2_aggregate_gradients(tensor_t* aggregated_grads) {
    shm_layer_id_t grad_layer = SHM_GRAD_LAYER2;
    
    // Read and sum gradients from all downstream contributors
    if (shm_read_aggregated_gradients(grad_layer, aggregated_grads) < 0) {
        log_error("Failed to aggregate gradients");
        return -1;
    }
    
    log_info("Layer 2: Aggregated gradients from %d Layer 3 workers", 4);
    
    return 0;
}
```

**Under the hood (gradient aggregation):**
```c
int shm_read_aggregated_gradients(shm_layer_id_t grad_layer_id, tensor_t* aggregated) {
    shm_segment_t* seg = &g_shm_manager.layers[grad_layer_id];
    int num_contributors = get_gradient_contributors(grad_layer_id);  // 4 for Layer 2
    int input_channels = get_gradient_input_channels(grad_layer_id);  // 32 for Layer 2
    
    // Initialize to zero
    memset(aggregated->data, 0, aggregated->channels * aggregated->length * sizeof(float));
    
    // Sum contributions from all 4 Layer 3 workers
    for (int contrib_id = 0; contrib_id < num_contributors; contrib_id++) {
        size_t offset = contrib_id * input_channels * seg->length * sizeof(float);
        float* contrib_data = (float*)((char*)seg->shm_ptr + sizeof(shm_header_t) + offset);
        
        // Add this contribution to aggregated gradients
        for (int i = 0; i < input_channels * seg->length; i++) {
            aggregated->data[i] += contrib_data[i];
        }
    }
    
    return 0;
}
```

**Memory layout during aggregation:**
```
SHM_GRAD_LAYER2 Buffer:
┌─────────────────────────────────────────────┐
│ Header (64 bytes)                           │
├─────────────────────────────────────────────┤
│ Contributor 0 (Worker 7): 32 ch × 300 len  │  Layer 3 Worker 0
│ [grad_0_0, grad_0_1, ..., grad_0_9599]     │
├─────────────────────────────────────────────┤
│ Contributor 1 (Worker 8): 32 ch × 300 len  │  Layer 3 Worker 1
│ [grad_1_0, grad_1_1, ..., grad_1_9599]     │
├─────────────────────────────────────────────┤
│ Contributor 2 (Worker 9): 32 ch × 300 len  │  Layer 3 Worker 2
│ [grad_2_0, grad_2_1, ..., grad_2_9599]     │
├─────────────────────────────────────────────┤
│ Contributor 3 (Worker 10): 32 ch × 300 len │  Layer 3 Worker 3
│ [grad_3_0, grad_3_1, ..., grad_3_9599]     │
└─────────────────────────────────────────────┘

Aggregation:
aggregated[i] = grad_0[i] + grad_1[i] + grad_2[i] + grad_3[i]
for i = 0 to 9599 (32 channels × 300 length)
```

## 7. Atomic Operations for Synchronization

```c
// Example: Worker signals completion using atomic increment
#include "shared_memory.h"

int worker_signal_completion(shm_layer_id_t layer_id, int sample_id) {
    shm_segment_t* seg = &g_shm_manager.layers[layer_id];
    
    // Atomically increment completion counter (GCC built-in atomic)
    int completed = __sync_add_and_fetch(&seg->header->forward_contributors_completed, 1);
    int expected = seg->header->forward_expected_contributors;
    
    log_info("Worker signaled completion (%d/%d)", completed, expected);
    
    // Last worker sets ready flag
    if (completed >= expected) {
        // Reset counter for next sample
        seg->header->forward_contributors_completed = 0;
        
        // Memory barrier ensures all writes are visible
        __sync_synchronize();
        
        // Set ready flag (atomic write to volatile variable)
        seg->header->forward_ready = 1;
        
        log_info("Last worker set forward_ready=1");
    }
    
    return 0;
}
```

**Atomic operations used:**
- `__sync_add_and_fetch(&var, 1)`: Atomic increment, returns new value
- `__sync_synchronize()`: Full memory barrier (prevents reordering)
- `__sync_lock_test_and_set(&var, val)`: Atomic swap
- `__sync_fetch_and_add(&var, 0)`: Atomic read (volatile load)

**Why atomic operations?**
- **Thread-safe:** Multiple workers can increment counter simultaneously without races
- **Lock-free:** No blocking, no contention, no deadlock risk
- **Memory ordering:** Ensures writes are visible across CPUs/cores

## 8. Cleanup and Resource Management

```c
// Example: Device cleanup on shutdown
#include "shared_memory.h"
#include <signal.h>

void signal_handler(int signum) {
    log_info("Caught signal %d, shutting down...", signum);
    
    // Set pipeline phase to PHASE_DONE (notifies all devices)
    shm_set_phase(PHASE_DONE);
    
    // Cleanup shared memory resources
    shm_cleanup();
    
    exit(0);
}

int main(int argc, char** argv) {
    // Register signal handlers
    signal(SIGINT, signal_handler);   // Ctrl+C
    signal(SIGTERM, signal_handler);  // Kill command
    
    // Initialize shared memory
    shm_init();
    
    // Create segments
    shm_create_segment(SHM_LAYER1_INPUT, 16, 300, 2);
    
    // ... device processing ...
    
    // Normal cleanup
    shm_cleanup();
    
    return 0;
}
```

**Cleanup sequence:**
```c
void shm_cleanup(void) {
    for (int i = 0; i < MAX_SHM_LAYERS; i++) {
        shm_segment_t* seg = &g_shm_manager.layers[i];
        
        // 1. Detach from shared memory
        if (seg->shm_ptr != NULL) {
            shmdt(seg->shm_ptr);
        }
        
        // 2. Mark segment for deletion (actual deletion when all detach)
        if (seg->shm_id > 0) {
            shmctl(seg->shm_id, IPC_RMID, NULL);
        }
        
        // 3. Destroy barriers
        if (seg->forward_barrier.initialized) {
            pthread_barrier_destroy(&seg->forward_barrier.barrier);
        }
    }
}
```

## 9. Complete Example: Layer Worker Main Loop

```c
// Example: Complete Layer 2 Worker 1 implementation
#include "shared_memory.h"
#include "nn_types.h"

int run_layer2_worker(int worker_id) {
    // Configuration
    shm_layer_id_t input_layer = SHM_LAYER2_INPUT;
    shm_layer_id_t output_layer = SHM_LAYER3_INPUT;
    shm_layer_id_t grad_input_layer = SHM_GRAD_LAYER2;
    shm_layer_id_t grad_output_layer = SHM_GRAD_LAYER1;
    
    int input_channels = 32;
    int output_channels = 16;
    int length = 300;
    
    // Initialize shared memory
    shm_init();
    shm_create_segment(input_layer, input_channels, length, 3);
    shm_create_segment(output_layer, 48, length, 3);  // Combined 3×16 = 48
    shm_create_segment(grad_input_layer, input_channels, length, 4);  // From 4 Layer 3 workers
    shm_create_segment(grad_output_layer, input_channels, length, 3);  // To Layer 1 (3 writers)
    
    // Initialize pipeline control
    shm_init_pipeline_control();
    
    // Allocate tensors
    tensor_t* input = tensor_create(1, input_channels, length);
    tensor_t* output = tensor_create(1, output_channels, length);
    tensor_t* grad_output = tensor_create(1, output_channels, length);
    tensor_t* grad_input = tensor_create(1, input_channels, length);
    
    log_info("Layer 2 Worker %d initialized", worker_id);
    
    // Main processing loop
    int sample_count = 0;
    while (1) {
        // ========== FORWARD PASS ==========
        
        // Wait for forward phase signal
        shm_wait_for_phase(PHASE_FORWARD_LAYER2);
        
        // Check for shutdown
        if (shm_get_phase() == PHASE_DONE) {
            break;
        }
        
        // Read input from shared memory (all workers read same data)
        shm_read_tensor(input_layer, input);
        
        // Perform Conv1D forward pass
        conv1d_forward(input, output, worker_id);
        
        // Write output to shared memory (disjoint channel range)
        shm_write_tensor(output_layer, worker_id, output);
        
        // Wait for all workers to finish writing
        shm_barrier_forward(output_layer);
        
        // Signal completion (last worker sets ready flag)
        shm_signal_forward_complete(output_layer, sample_count);
        
        log_info("Layer 2 Worker %d: Forward pass complete", worker_id);
        
        // ========== BACKWARD PASS ==========
        
        // Wait for backward phase signal
        shm_wait_for_phase(PHASE_BACKWARD_LAYER2);
        
        // Aggregate gradients from Layer 3 workers
        shm_read_aggregated_gradients(grad_input_layer, grad_output);
        
        // Perform Conv1D backward pass
        conv1d_backward(input, output, grad_output, grad_input, worker_id);
        
        // Write gradients to shared memory for Layer 1
        shm_write_gradient_contribution(grad_output_layer, worker_id, grad_input);
        
        // Wait for all workers to finish writing gradients
        shm_barrier_backward(output_layer);
        
        // Signal backward completion
        shm_signal_backward_complete(output_layer, sample_count);
        
        log_info("Layer 2 Worker %d: Backward pass complete", worker_id);
        
        sample_count++;
    }
    
    // Cleanup
    tensor_free(input);
    tensor_free(output);
    tensor_free(grad_output);
    tensor_free(grad_input);
    shm_cleanup();
    
    log_info("Layer 2 Worker %d shutdown complete", worker_id);
    
    return 0;
}
```

## 10. Performance Profiling Example

```c
// Example: Measure synchronization overhead
#include <time.h>

uint64_t get_timestamp_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000ULL;
}

void profile_synchronization(shm_layer_id_t layer_id) {
    uint64_t start, end, wait_time;
    
    // Measure phase wait time
    start = get_timestamp_us();
    shm_wait_for_phase(PHASE_FORWARD_LAYER1);
    end = get_timestamp_us();
    wait_time = end - start;
    
    log_info("Phase wait time: %llu microseconds", wait_time);
    
    // Measure barrier synchronization time
    start = get_timestamp_us();
    shm_barrier_forward(layer_id);
    end = get_timestamp_us();
    
    log_info("Barrier sync time: %llu microseconds", end - start);
    
    // Measure atomic completion time
    start = get_timestamp_us();
    shm_signal_forward_complete(layer_id, 0);
    end = get_timestamp_us();
    
    log_info("Atomic signal time: %llu microseconds", end - start);
}
```

---

## Summary of Key Patterns

1. **Create segments before use:** `shm_create_segment()` in each device's initialization
2. **Wait for phases:** Workers poll `shm_wait_for_phase()` to synchronize with head
3. **Write to disjoint ranges:** Forward pass workers write to separate channel ranges
4. **Use barriers for intra-layer sync:** `shm_barrier_forward/backward()` ensures all workers finish
5. **Atomic counters for completion:** `shm_signal_forward/backward_complete()` coordinates multi-worker layers
6. **Aggregate gradients:** Backward pass sums contributions from all downstream workers
7. **Cleanup on exit:** `shm_cleanup()` detaches and marks segments for deletion

---

**Document Version:** 1.0  
**Last Updated:** February 17, 2026  
**Companion to:** SHARED_MEMORY_ANALYSIS.md, ARCHITECTURE_DIAGRAMS.md
