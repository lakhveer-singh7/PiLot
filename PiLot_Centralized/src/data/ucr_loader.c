#include "nn_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/*  UCR dataset type                                                   */
/* ------------------------------------------------------------------ */
typedef struct {
    float* data;        /* [num_samples * sample_length] */
    int*   labels;
    int    num_samples;
    int    sample_length;
    int    num_classes;
} dataset_t;

/* ------------------------------------------------------------------ */
/*  Parse a single CSV line:  label, v0, v1, ...                       */
/* ------------------------------------------------------------------ */
#define MAX_LINE_LEN 65536

static int parse_ucr_line(const char* line, float* sample,
                          int* label, int sample_length) {
    if (!line || !sample || !label) return -1;

    char buf[MAX_LINE_LEN];
    strncpy(buf, line, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    char* tok = strtok(buf, ",");
    if (!tok) return -1;
    *label = atoi(tok);

    for (int i = 0; i < sample_length; i++) {
        tok = strtok(NULL, ",");
        if (!tok) return -1;
        sample[i] = (float)atof(tok);
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Load UCR dataset (no sample-count limit)                           */
/* ------------------------------------------------------------------ */
dataset_t* load_ucr_dataset(const char* filename) {
    FILE* f = fopen(filename, "r");
    if (!f) { log_error("Cannot open dataset: %s", filename); return NULL; }

    char line[MAX_LINE_LEN];
    int num_lines = 0, sample_length = 0;

    /* First pass: count lines & determine length */
    while (fgets(line, sizeof(line), f)) {
        if (strlen(line) <= 1) continue;
        num_lines++;
        if (sample_length == 0) {
            char tmp[MAX_LINE_LEN];
            strncpy(tmp, line, sizeof(tmp) - 1);
            tmp[sizeof(tmp) - 1] = '\0';
            char* tok = strtok(tmp, ",");  /* label */
            tok = strtok(NULL, ",");
            while (tok) { sample_length++; tok = strtok(NULL, ","); }
        }
    }
    if (num_lines == 0 || sample_length == 0) {
        log_error("Invalid dataset file: %s", filename);
        fclose(f);
        return NULL;
    }

    dataset_t* ds = (dataset_t*)calloc(1, sizeof(dataset_t));
    if (!ds) { fclose(f); return NULL; }

    ds->num_samples   = num_lines;
    ds->sample_length = sample_length;
    ds->data   = (float*)malloc((size_t)num_lines * sample_length * sizeof(float));
    ds->labels = (int*)malloc((size_t)num_lines * sizeof(int));
    if (!ds->data || !ds->labels) {
        free(ds->data); free(ds->labels); free(ds);
        fclose(f); return NULL;
    }

    /* Second pass: load data */
    rewind(f);
    int idx = 0, max_label = -1;
    while (fgets(line, sizeof(line), f) && idx < num_lines) {
        if (strlen(line) <= 1) continue;
        float* row = &ds->data[idx * sample_length];
        int lbl;
        if (parse_ucr_line(line, row, &lbl, sample_length) == 0) {
            lbl -= 1;  /* UCR labels are 1-based → 0-based */
            if (lbl < 0) continue;
            ds->labels[idx] = lbl;
            if (lbl > max_label) max_label = lbl;
            idx++;
        }
    }
    fclose(f);
    ds->num_samples = idx;

    /* Shuffle (Fisher-Yates) */
    {
        static int seeded = 0;
        if (!seeded) { srand((unsigned)time(NULL)); seeded = 1; }
        float* tmp = (float*)malloc((size_t)sample_length * sizeof(float));
        for (int i = idx - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            memcpy(tmp, &ds->data[i * sample_length],
                   (size_t)sample_length * sizeof(float));
            memcpy(&ds->data[i * sample_length],
                   &ds->data[j * sample_length],
                   (size_t)sample_length * sizeof(float));
            memcpy(&ds->data[j * sample_length], tmp,
                   (size_t)sample_length * sizeof(float));
            int tl = ds->labels[i];
            ds->labels[i] = ds->labels[j];
            ds->labels[j] = tl;
        }
        free(tmp);
    }

    /* Count unique classes */
    int* seen = (int*)calloc(max_label + 1, sizeof(int));
    int unique = 0;
    for (int i = 0; i < idx; i++) {
        if (!seen[ds->labels[i]]) { seen[ds->labels[i]] = 1; unique++; }
    }
    free(seen);
    ds->num_classes = unique;

    log_info("Loaded %s: %d samples, %d classes, length=%d",
             filename, ds->num_samples, ds->num_classes, ds->sample_length);
    return ds;
}

void free_dataset(dataset_t* ds) {
    if (ds) { free(ds->data); free(ds->labels); free(ds); }
}

/* Return a freshly-allocated 1×1×L tensor for sample_idx */
tensor_t* get_dataset_sample(const dataset_t* ds, int sample_idx) {
    if (!ds || sample_idx < 0 || sample_idx >= ds->num_samples) return NULL;
    tensor_t* t = tensor_create(1, 1, ds->sample_length);
    if (!t) return NULL;
    memcpy(t->data, &ds->data[sample_idx * ds->sample_length],
           (size_t)ds->sample_length * sizeof(float));
    return t;
}

int get_dataset_label(const dataset_t* ds, int idx) {
    if (!ds || idx < 0 || idx >= ds->num_samples) return -1;
    return ds->labels[idx];
}

/* Per-sample z-normalisation */
void normalize_dataset(dataset_t* ds) {
    if (!ds || !ds->data) return;
    for (int s = 0; s < ds->num_samples; s++) {
        float* row = ds->data + s * ds->sample_length;
        double sum = 0;
        for (int i = 0; i < ds->sample_length; i++) sum += row[i];
        float mean = (float)(sum / ds->sample_length);
        double var = 0;
        for (int i = 0; i < ds->sample_length; i++) {
            double d = row[i] - mean;
            var += d * d;
        }
        float std = (float)sqrt(var / ds->sample_length);
        if (std > 1e-8f)
            for (int i = 0; i < ds->sample_length; i++)
                row[i] = (row[i] - mean) / std;
    }
}
