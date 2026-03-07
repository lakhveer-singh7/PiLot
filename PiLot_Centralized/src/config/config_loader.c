#include "config_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* ------------------------------------------------------------------ */
/*  Minimal JSON helpers (same parser as distributed version)          */
/* ------------------------------------------------------------------ */
static char* read_file_contents(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) { log_error("Cannot open config: %s", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc(sz + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t nread = fread(buf, 1, sz, f);
    buf[nread] = '\0';
    fclose(f);
    return buf;
}

static char* find_value(const char* json, const char* key) {
    char skey[256];
    snprintf(skey, sizeof(skey), "\"%s\"", key);
    char* pos = strstr(json, skey);
    if (!pos) return NULL;
    pos = strchr(pos, ':');
    if (!pos) return NULL;
    pos++;
    while (*pos && isspace((unsigned char)*pos)) pos++;

    char* val = (char*)malloc(256);
    int i = 0;
    if (*pos == '"') {
        pos++;
        while (*pos && *pos != '"' && i < 255) val[i++] = *pos++;
    } else if (*pos == '[') {
        int d = 1; val[i++] = *pos++;
        while (*pos && d > 0 && i < 255) {
            if (*pos == '[') d++;
            if (*pos == ']') d--;
            val[i++] = *pos++;
        }
    } else {
        while (*pos && !isspace((unsigned char)*pos) &&
               *pos != ',' && *pos != '}' && i < 255)
            val[i++] = *pos++;
    }
    val[i] = '\0';
    return val;
}

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */
model_config_t* load_model_config(const char* path) {
    char* json = read_file_contents(path);
    if (!json) return NULL;

    model_config_t* c = (model_config_t*)calloc(1, sizeof(model_config_t));
    if (!c) { free(json); return NULL; }

    char* v;
    /* name */
    if ((v = find_value(json, "name"))) {
        strncpy(c->name, v, sizeof(c->name) - 1); free(v);
    }
    /* version */
    if ((v = find_value(json, "version"))) {
        strncpy(c->version, v, sizeof(c->version) - 1); free(v);
    }
    /* dataset */
    if ((v = find_value(json, "dataset"))) {
        strncpy(c->dataset, v, sizeof(c->dataset) - 1); free(v);
    }
    if ((v = find_value(json, "epochs"))) {
        c->epochs = atoi(v); free(v);
    }
    if ((v = find_value(json, "learning_rate"))) {
        c->learning_rate = (float)atof(v); free(v);
    }
    if ((v = find_value(json, "num_classes"))) {
        c->num_classes = atoi(v); free(v);
    }
    if ((v = find_value(json, "input_length"))) {
        c->input_length = atoi(v); free(v);
    }
    if ((v = find_value(json, "memory_limit_bytes"))) {
        c->memory_limit_bytes = atoi(v); free(v);
    }

    /* ---------- Parse layers array ---------- */
    char* layers_start = strstr(json, "\"layers\"");
    if (layers_start) {
        layers_start = strchr(layers_start, '[');
        if (layers_start) {
            layers_start++;
            int cnt = 0;
            char* lp = strstr(layers_start, "\"id\"");
            char* arr_end = strstr(layers_start, "]");

            while (lp && cnt < MAX_LAYERS && lp < arr_end) {
                model_layer_t* L = &c->layers[cnt];

                if ((v = find_value(lp, "id")))           { L->id = atoi(v); free(v); }
                if ((v = find_value(lp, "type")))          { strncpy(L->type, v, sizeof(L->type)-1); free(v); }
                if ((v = find_value(lp, "in_channels")))   { L->in_channels   = atoi(v); free(v); }
                if ((v = find_value(lp, "out_channels")))  { L->out_channels  = atoi(v); free(v); }
                if ((v = find_value(lp, "kernel_size")))   { L->kernel_size   = atoi(v); free(v); }
                if ((v = find_value(lp, "stride")))        { L->stride        = atoi(v); free(v); }
                if ((v = find_value(lp, "padding")))       { L->padding       = atoi(v); free(v); }
                if ((v = find_value(lp, "num_devices")))   { L->num_devices   = atoi(v); free(v); }
                if ((v = find_value(lp, "in_features")))   { L->in_features   = atoi(v); free(v); }
                if ((v = find_value(lp, "out_features")))  { L->out_features  = atoi(v); free(v); }
                if ((v = find_value(lp, "input_length")))  { L->input_length  = atoi(v); free(v); }
                if ((v = find_value(lp, "output_length"))) { L->output_length = atoi(v); free(v); }

                cnt++;
                lp = strstr(lp + 1, "\"id\"");
            }
            c->num_layers = cnt;
        }
    }

    free(json);
    log_info("Config: %s v%s — %d layers, dataset=%s, lr=%.4f",
             c->name, c->version, c->num_layers, c->dataset, c->learning_rate);
    return c;
}

void free_model_config(model_config_t* c) { free(c); }
