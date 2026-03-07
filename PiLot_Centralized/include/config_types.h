#ifndef CONFIG_TYPES_H
#define CONFIG_TYPES_H

#include "logging.h"
#include <stdint.h>

#define MAX_LAYERS 10

/* ---------- Per-layer description from JSON ---------- */
typedef struct {
    int  id;
    char type[32];        /* "conv1d" or "fc" */
    int  in_channels;
    int  out_channels;
    int  kernel_size;
    int  stride;
    int  padding;
    int  num_devices;     /* kept for compat; ignored in centralized mode */
    int  input_length;
    int  output_length;
    /* FC-specific */
    int  in_features;
    int  out_features;
} model_layer_t;

/* ---------- Complete model configuration ---------- */
typedef struct {
    char  name[256];
    char  version[32];
    char  dataset[64];
    int   epochs;
    int   num_classes;
    int   input_length;
    int   memory_limit_bytes;   /* ignored in centralized mode */
    float learning_rate;
    int   num_layers;
    model_layer_t layers[MAX_LAYERS];
} model_config_t;

/* ---------- API ---------- */
model_config_t* load_model_config(const char* path);
void            free_model_config(model_config_t* c);

#endif /* CONFIG_TYPES_H */
