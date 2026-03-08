/*
 * PiLot Centralized — Single-process CNN training
 *
 * Same model architecture, datasets, and training features as the distributed
 * PiLot, but everything runs in one process with direct function calls.
 * No IPC, no shared memory, no memory budget.
 *
 * Usage:
 *   ./pilot_centralized --config=configs/model_config.json \
 *                       [--dataset=ECG5000] [--epochs=100] [--debug]
 */

#include "nn_types.h"
#include "config_types.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sys/resource.h>

/* ---- Forward declarations from ucr_loader.c ---- */
typedef struct {
    float* data;
    int*   labels;
    int    num_samples;
    int    sample_length;
    int    num_classes;
} dataset_t;

dataset_t* load_ucr_dataset(const char* filename);
void       free_dataset(dataset_t* ds);
tensor_t*  get_dataset_sample(const dataset_t* ds, int idx);
int        get_dataset_label(const dataset_t* ds, int idx);
void       normalize_dataset(dataset_t* ds);

/* ================================================================== */
/*  Per-conv-layer runtime state                                       */
/* ================================================================== */
typedef struct {
    conv1d_config_t* conv;
    int  in_channels;
    int  out_channels;
    int  input_length;
    int  output_length;
    int  num_groups;       /* for GroupNorm */
    /* intermediate tensors */
    tensor_t* conv_out;    /* after conv */
    tensor_t* gn_out;      /* after group-norm (pre-relu) */
    tensor_t* act_out;     /* after relu (forwarded to next layer) */
    /* backward intermediates */
    tensor_t* grad_act;    /* grad before relu */
    tensor_t* grad_gn;     /* grad after relu-backward */
    tensor_t* grad_conv;   /* grad after gn-backward */
    tensor_t* grad_input;  /* grad w.r.t. layer input */
    /* param gradients */
    float* grad_w;
    float* grad_b;
    /* Adam state */
    float* m_w;
    float* v_w;
    float* m_b;
    float* v_b;
    int    num_w;
    int    num_b;
} conv_layer_state_t;

/* ================================================================== */
/*  Helpers                                                            */
/* ================================================================== */
static int argmax(const float* x, int n) {
    int idx = 0; float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) { mx = x[i]; idx = i; }
    return idx;
}

/* ================================================================== */
/*  Build layer stack from config                                      */
/* ================================================================== */
static int build_conv_layers(const model_config_t* cfg,
                             conv_layer_state_t** out_layers,
                             int* out_count) {
    int num_conv = 0;
    for (int i = 0; i < cfg->num_layers; i++)
        if (strcmp(cfg->layers[i].type, "conv1d") == 0) num_conv++;

    conv_layer_state_t* layers = (conv_layer_state_t*)calloc(num_conv,
                                        sizeof(conv_layer_state_t));
    if (!layers) return -1;

    int idx = 0;
    for (int i = 0; i < cfg->num_layers; i++) {
        const model_layer_t* L = &cfg->layers[i];
        if (strcmp(L->type, "conv1d") != 0) continue;

        conv_layer_state_t* s = &layers[idx];
        s->in_channels   = L->in_channels;
        s->out_channels  = L->out_channels;
        s->input_length  = L->input_length;
        s->output_length = L->output_length;
        s->num_groups    = 8;
        if (s->out_channels % s->num_groups != 0)
            s->num_groups = 1;   /* fallback if not divisible */

        /* Conv config */
        s->conv = create_conv1d_config(L->in_channels, L->out_channels,
                                       L->kernel_size, L->stride, L->padding);
        if (!s->conv) { free(layers); return -1; }

        int OL = s->output_length;
        int OC = s->out_channels;
        int IC = s->in_channels;
        int IL = s->input_length;

        /* Forward intermediates */
        s->conv_out = tensor_create(1, OC, OL);
        s->gn_out   = tensor_create(1, OC, OL);
        s->act_out  = tensor_create(1, OC, OL);

        /* Backward intermediates */
        s->grad_act   = tensor_create(1, OC, OL);
        s->grad_gn    = tensor_create(1, OC, OL);
        s->grad_conv  = tensor_create(1, OC, OL);
        s->grad_input = tensor_create(1, IC, IL);

        /* Parameter gradients */
        s->grad_w = (float*)calloc(s->conv->weights_size / sizeof(float), sizeof(float));
        s->grad_b = (float*)calloc(s->conv->bias_size / sizeof(float), sizeof(float));

        /* Adam state */
        s->num_w = s->conv->weights_size / (int)sizeof(float);
        s->num_b = s->conv->bias_size    / (int)sizeof(float);
        s->m_w = (float*)calloc(s->num_w, sizeof(float));
        s->v_w = (float*)calloc(s->num_w, sizeof(float));
        s->m_b = (float*)calloc(s->num_b, sizeof(float));
        s->v_b = (float*)calloc(s->num_b, sizeof(float));

        idx++;
    }

    *out_layers = layers;
    *out_count  = num_conv;
    return 0;
}

static void free_conv_layers(conv_layer_state_t* layers, int n) {
    for (int i = 0; i < n; i++) {
        conv_layer_state_t* s = &layers[i];
        free_conv1d_config(s->conv);
        tensor_free(s->conv_out);
        tensor_free(s->gn_out);
        tensor_free(s->act_out);
        tensor_free(s->grad_act);
        tensor_free(s->grad_gn);
        tensor_free(s->grad_conv);
        tensor_free(s->grad_input);
        free(s->grad_w); free(s->grad_b);
        free(s->m_w); free(s->v_w);
        free(s->m_b); free(s->v_b);
    }
    free(layers);
}

/* ================================================================== */
/*  main                                                               */
/* ================================================================== */
int main(int argc, char* argv[]) {
    srand((unsigned)time(NULL));

    /* ---------- CLI parsing ---------- */
    char config_file[512] = "";
    char dataset_override[256] = "";
    char log_dir[512] = "logs";
    int  epochs_override = -1;
    int  debug = 0;

    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--config=", 9) == 0)
            strncpy(config_file, argv[i] + 9, sizeof(config_file) - 1);
        else if (strncmp(argv[i], "--dataset=", 10) == 0)
            strncpy(dataset_override, argv[i] + 10, sizeof(dataset_override) - 1);
        else if (strncmp(argv[i], "--epochs=", 9) == 0)
            epochs_override = atoi(argv[i] + 9);
        else if (strncmp(argv[i], "--log-dir=", 10) == 0)
            strncpy(log_dir, argv[i] + 10, sizeof(log_dir) - 1);
        else if (strcmp(argv[i], "--debug") == 0)
            debug = 1;
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s --config=<json> [--dataset=<name>] "
                   "[--epochs=N] [--log-dir=<dir>] [--debug]\n", argv[0]);
            return 0;
        }
    }
    if (config_file[0] == '\0') {
        fprintf(stderr, "Error: --config=<json> is required\n");
        return 1;
    }
    if (debug) set_log_level_debug();

    /* Initialize log file */
    {
        char mkdir_cmd[600];
        snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p %s", log_dir);
        system(mkdir_cmd);
        char log_path[600];
        snprintf(log_path, sizeof(log_path), "%s/pilot_centralized.log", log_dir);
        log_init(log_path);
    }

    /* ---------- Load model config ---------- */
    model_config_t* cfg = load_model_config(config_file);
    if (!cfg) { log_error("Failed to load config"); return 1; }

    const char* ds_name = (dataset_override[0] != '\0')
                          ? dataset_override : cfg->dataset;
    int max_epochs = (epochs_override > 0) ? epochs_override : cfg->epochs;
    if (max_epochs <= 0) max_epochs = 100;

    log_info("=== PiLot Centralized ===");
    log_info("Config : %s", config_file);
    log_info("Dataset: %s,  Epochs: %d,  LR: %.4f", ds_name, max_epochs, cfg->learning_rate);

    /* ---------- Load datasets ---------- */
    const char* data_root = getenv("UCR_DATA_ROOT");
    if (!data_root) data_root = "/mnt/d/New folder/UCR_DATASETS";

    char train_path[512], test_path[512];
    snprintf(train_path, sizeof(train_path), "%s/%s/%s_TRAIN", data_root, ds_name, ds_name);
    snprintf(test_path, sizeof(test_path), "%s/%s/%s_TEST", data_root, ds_name, ds_name);

    dataset_t* train_ds = load_ucr_dataset(train_path);
    dataset_t* test_ds  = load_ucr_dataset(test_path);
    if (!train_ds || !test_ds) {
        log_error("Failed to load datasets");
        free_dataset(train_ds); free_dataset(test_ds);
        free_model_config(cfg);
        return 1;
    }
    normalize_dataset(train_ds);
    normalize_dataset(test_ds);

    int num_classes = cfg->num_classes;
    log_info("Train: %d samples,  Test: %d samples,  Classes: %d", train_ds->num_samples, test_ds->num_samples, num_classes);

    /* ---------- Build conv layer stack ---------- */
    conv_layer_state_t* conv_layers = NULL;
    int num_conv = 0;
    if (build_conv_layers(cfg, &conv_layers, &num_conv) < 0 || num_conv == 0) {
        log_error("Failed to build conv layer stack or no conv layers found"); return 1;
    }
    log_info("Built %d conv layers", num_conv);

    /* ---------- Build FC head ---------- */
    /* Find the FC layer in config */
    int fc_in = -1, fc_out = -1;
    for (int i = 0; i < cfg->num_layers; i++) {
        if (strcmp(cfg->layers[i].type, "fc") == 0) {
            fc_in  = cfg->layers[i].in_features;
            fc_out = cfg->layers[i].out_features;
            break;
        }
    }
    if (fc_in < 0) {
        /* Derive from last conv layer: dual pooling doubles channels */
        int last_ch = conv_layers[num_conv - 1].out_channels;
        fc_in  = last_ch * 2;
        fc_out = num_classes;
    }

    /* Validate FC in_features matches dual-pooled output */
    {
        int expected = conv_layers[num_conv - 1].out_channels * 2;
        if (fc_in != expected) {
            log_error("FC in_features (%d) != pooled channels (%d)", fc_in, expected);
            return 1;
        }
    }

    fc_config_t* fc = create_fc_config(fc_in, fc_out);
    if (!fc) { log_error("Failed to create FC"); return 1; }

    /* FC Adam state */
    int fc_nw = fc->weights_size / (int)sizeof(float);
    int fc_nb = fc->bias_size    / (int)sizeof(float);
    float* fc_gw = (float*)calloc(fc_nw, sizeof(float));
    float* fc_gb = (float*)calloc(fc_nb, sizeof(float));
    float* fc_mw = (float*)calloc(fc_nw, sizeof(float));
    float* fc_vw = (float*)calloc(fc_nw, sizeof(float));
    float* fc_mb = (float*)calloc(fc_nb, sizeof(float));
    float* fc_vb = (float*)calloc(fc_nb, sizeof(float));
    if (!fc_gw || !fc_gb || !fc_mw || !fc_vw || !fc_mb || !fc_vb) {
        log_error("FC optimizer buffer allocation failed"); return 1;
    }

    /* ---------- Classifier tensors ---------- */
    int last_out_ch  = conv_layers[num_conv - 1].out_channels;
    int last_out_len = conv_layers[num_conv - 1].output_length;
    int pool_features = last_out_ch * 2;   /* dual pooling */

    tensor_t* pooled      = tensor_create(1, pool_features, 1);
    tensor_t* dropout_out = tensor_create(1, pool_features, 1);
    float*    dropout_mask= (float*)calloc(pool_features, sizeof(float));
    tensor_t* fc_out_t    = tensor_create(1, fc_out, 1);
    tensor_t* probs       = tensor_create(1, num_classes, 1);

    /* Backward classifier tensors */
    tensor_t* grad_logits  = tensor_create(1, fc_out, 1);
    tensor_t* grad_fc_in   = tensor_create(1, pool_features, 1);
    tensor_t* grad_dropout = tensor_create(1, pool_features, 1);
    tensor_t* grad_pool_in = tensor_create(1, last_out_ch, last_out_len);

    /* Augmentation buffer */
    int input_length = cfg->input_length;
    float* aug_buf = (float*)malloc((size_t)input_length * sizeof(float));
    if (!aug_buf) { log_error("Augmentation buffer allocation failed"); return 1; }

    /* ---------- Training hyperparams ---------- */
    float adam_b1 = 0.9f, adam_b2 = 0.999f, adam_eps = 1e-8f;
    float weight_decay = 0.0003f;
    float dropout_rate = 0.2f;
    int   adam_t = 0;

    /* Early stopping */
    float best_test_acc = 0.0f;
    int   best_epoch = 0;
    int   patience = 30;
    int   no_improve = 0;
    float* best_fc_w = (float*)malloc((size_t)fc->weights_size);
    float* best_fc_b = (float*)malloc((size_t)fc->bias_size);
    if (!best_fc_w || !best_fc_b) { log_error("Best-weight buffer allocation failed"); return 1; }
    memcpy(best_fc_w, fc->weights, (size_t)fc->weights_size);
    memcpy(best_fc_b, fc->bias,    (size_t)fc->bias_size);

    /* Best conv layer weight storage for full-model early stopping */
    float** best_conv_w = (float**)malloc((size_t)num_conv * sizeof(float*));
    float** best_conv_b = (float**)malloc((size_t)num_conv * sizeof(float*));
    for (int i = 0; i < num_conv; i++) {
        best_conv_w[i] = (float*)malloc((size_t)conv_layers[i].conv->weights_size);
        best_conv_b[i] = (float*)malloc((size_t)conv_layers[i].conv->bias_size);
        memcpy(best_conv_w[i], conv_layers[i].conv->weights, (size_t)conv_layers[i].conv->weights_size);
        memcpy(best_conv_b[i], conv_layers[i].conv->bias,    (size_t)conv_layers[i].conv->bias_size);
    }

    /* Timing */
    struct timespec t_start, t_epoch;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    /* ================================================================ */
    /*  EPOCH LOOP                                                      */
    /* ================================================================ */
    for (int epoch = 1; epoch <= max_epochs; epoch++) {
        /* ---------- Training phase ---------- */
        int train_correct = 0, train_total = 0;
        float train_loss_sum = 0.0f;

        for (int s = 0; s < train_ds->num_samples; s++) {
            int label = get_dataset_label(train_ds, s);

            /* Copy & augment sample */
            memcpy(aug_buf, &train_ds->data[s * input_length],
                   (size_t)input_length * sizeof(float));
            apply_augmentation(aug_buf, input_length);

            /* Build input tensor (reuse stack-like pointer trick) */
            tensor_t input_t;
            input_t.batch_size = 1;
            input_t.channels   = conv_layers[0].in_channels;
            input_t.length     = conv_layers[0].input_length;
            input_t.stride     = input_t.length;
            input_t.data       = aug_buf;
            input_t.allocated_size = 0;  /* not owned */

            /* ---- Forward: Conv layers ---- */
            tensor_t* x = &input_t;
            for (int i = 0; i < num_conv; i++) {
                conv_layer_state_t* cl = &conv_layers[i];
                conv1d_forward(x, cl->conv, cl->conv_out);
                group_norm_forward(cl->conv_out, cl->gn_out, cl->num_groups);
                relu_forward(cl->gn_out, cl->act_out);
                x = cl->act_out;
            }

            /* ---- Forward: Classifier ---- */
            dual_pooling1d(x, pooled);
            dropout_forward(pooled, dropout_out, dropout_mask,
                            dropout_rate, 1 /* training */);
            fully_connected_forward(dropout_out, fc, fc_out_t);
            softmax_forward(fc_out_t, probs);

            float loss = cross_entropy_loss(probs, &label, 1);
            train_loss_sum += loss;
            int pred = argmax(probs->data, num_classes);
            if (pred == label) train_correct++;
            train_total++;

            /* ---- Backward: Classifier ---- */
            cross_entropy_backward(probs, &label, 1, grad_logits);
            fully_connected_backward(grad_logits, dropout_out, fc,
                                     grad_fc_in, fc_gw, fc_gb);
            dropout_backward(grad_fc_in, dropout_mask, grad_dropout);
            dual_pooling1d_backward(grad_dropout, x, grad_pool_in);

            /* ---- Backward: Conv layers (reverse) ---- */
            tensor_t* dL_dx = grad_pool_in;
            for (int i = num_conv - 1; i >= 0; i--) {
                conv_layer_state_t* cl = &conv_layers[i];
                /* The input to this layer's relu was cl->gn_out */
                relu_backward(dL_dx, cl->gn_out, cl->grad_act);
                group_norm_backward(cl->conv_out, cl->grad_act,
                                    cl->grad_gn, cl->num_groups);

                /* Input to this conv layer */
                tensor_t* layer_in;
                tensor_t input_ref;
                if (i == 0) {
                    input_ref = input_t;
                    layer_in  = &input_ref;
                } else {
                    layer_in = conv_layers[i - 1].act_out;
                }

                conv1d_backward(cl->grad_gn, layer_in, cl->conv,
                                cl->grad_input, cl->grad_w, cl->grad_b);
                dL_dx = cl->grad_input;
            }

            /* ---- Adam updates ---- */
            adam_t++;
            float lr = lr_cosine_annealing(cfg->learning_rate, epoch, 60, 1e-5f);

            for (int i = 0; i < num_conv; i++) {
                conv_layer_state_t* cl = &conv_layers[i];
                clip_gradients(cl->grad_w, cl->num_w, 5.0f);
                clip_gradients(cl->grad_b, cl->num_b, 5.0f);
                adam_update(cl->conv->weights, cl->grad_w,
                            cl->m_w, cl->v_w, cl->num_w,
                            lr, adam_b1, adam_b2, adam_eps, adam_t, weight_decay);
                adam_update_bias(cl->conv->bias, cl->grad_b,
                                cl->m_b, cl->v_b, cl->num_b,
                                lr, adam_b1, adam_b2, adam_eps, adam_t);
            }
            clip_gradients(fc_gw, fc_nw, 5.0f);
            clip_gradients(fc_gb, fc_nb, 5.0f);
            adam_update(fc->weights, fc_gw, fc_mw, fc_vw, fc_nw,
                        lr, adam_b1, adam_b2, adam_eps, adam_t, weight_decay);
            adam_update_bias(fc->bias, fc_gb, fc_mb, fc_vb, fc_nb,
                            lr, adam_b1, adam_b2, adam_eps, adam_t);
        }

        float train_acc = (train_total > 0)
                          ? (float)train_correct / train_total * 100.0f : 0.0f;

        /* ---------- Testing phase ---------- */
        int test_correct = 0, test_total = 0;
        float test_loss_sum = 0.0f;

        struct timespec ts_test_start, ts_test_end;
        clock_gettime(CLOCK_MONOTONIC, &ts_test_start);

        for (int s = 0; s < test_ds->num_samples; s++) {
            int label = get_dataset_label(test_ds, s);

            tensor_t input_t;
            input_t.batch_size = 1;
            input_t.channels   = conv_layers[0].in_channels;
            input_t.length     = conv_layers[0].input_length;
            input_t.stride     = input_t.length;
            input_t.data       = &test_ds->data[s * input_length];
            input_t.allocated_size = 0;

            tensor_t* x = &input_t;
            for (int i = 0; i < num_conv; i++) {
                conv_layer_state_t* cl = &conv_layers[i];
                conv1d_forward(x, cl->conv, cl->conv_out);
                group_norm_forward(cl->conv_out, cl->gn_out, cl->num_groups);
                relu_forward(cl->gn_out, cl->act_out);
                x = cl->act_out;
            }

            dual_pooling1d(x, pooled);
            dropout_forward(pooled, dropout_out, dropout_mask, dropout_rate,
                            0 /* testing: no dropout */);
            fully_connected_forward(dropout_out, fc, fc_out_t);
            softmax_forward(fc_out_t, probs);

            float loss = cross_entropy_loss(probs, &label, 1);
            test_loss_sum += loss;
            int pred = argmax(probs->data, num_classes);
            if (pred == label) test_correct++;
            test_total++;
        }

        clock_gettime(CLOCK_MONOTONIC, &ts_test_end);
        double test_time = (ts_test_end.tv_sec - ts_test_start.tv_sec)
                         + (ts_test_end.tv_nsec - ts_test_start.tv_nsec) / 1e9;
        double avg_infer_ms = (test_total > 0)
                              ? test_time / test_total * 1000.0 : 0.0;

        float test_acc  = (test_total > 0)
                          ? (float)test_correct / test_total * 100.0f : 0.0f;
        float avg_loss  = (test_total > 0)
                          ? test_loss_sum / test_total : 0.0f;

        clock_gettime(CLOCK_MONOTONIC, &t_epoch);
        double elapsed = (t_epoch.tv_sec - t_start.tv_sec)
                       + (t_epoch.tv_nsec - t_start.tv_nsec) / 1e9;

        log_info("Epoch %3d | Train Acc %.2f%% | Test Acc %.2f%% (%d/%d) | "
                 "Loss %.4f | LR %.6f | Infer %.3fms | %.1fs",
                 epoch, train_acc, test_acc, test_correct, test_total,
                 avg_loss,
                 lr_cosine_annealing(cfg->learning_rate, epoch, 60, 1e-5f),
                 avg_infer_ms, elapsed);

        /* Standardized metrics line for automated parsing */
        {
            struct rusage usage;
            getrusage(RUSAGE_SELF, &usage);
            long mem_kb = usage.ru_maxrss;
            log_info("[METRICS] Epoch=%d | Timespan=%.1fs | Train_Acc=%.2f%% | "
                     "Test_Acc=%.2f%% | Infer_Latency=%.3fms | Memory=%ldKB",
                     epoch, elapsed, train_acc, test_acc, avg_infer_ms, mem_kb);
        }

        /* ---- Early stopping ---- */
        if (test_acc > best_test_acc) {
            best_test_acc = test_acc;
            best_epoch    = epoch;
            no_improve    = 0;
            memcpy(best_fc_w, fc->weights, (size_t)fc->weights_size);
            memcpy(best_fc_b, fc->bias,    (size_t)fc->bias_size);
            for (int i = 0; i < num_conv; i++) {
                memcpy(best_conv_w[i], conv_layers[i].conv->weights, (size_t)conv_layers[i].conv->weights_size);
                memcpy(best_conv_b[i], conv_layers[i].conv->bias,    (size_t)conv_layers[i].conv->bias_size);
            }
            log_info("  ** New best: %.2f%% at epoch %d", best_test_acc, epoch);
        } else {
            no_improve++;
            if (no_improve >= patience) {
                log_info("Early stopping at epoch %d (best %.2f%% at epoch %d)",
                         epoch, best_test_acc, best_epoch);
                memcpy(fc->weights, best_fc_w, (size_t)fc->weights_size);
                memcpy(fc->bias,    best_fc_b, (size_t)fc->bias_size);
                for (int i = 0; i < num_conv; i++) {
                    memcpy(conv_layers[i].conv->weights, best_conv_w[i], (size_t)conv_layers[i].conv->weights_size);
                    memcpy(conv_layers[i].conv->bias,    best_conv_b[i], (size_t)conv_layers[i].conv->bias_size);
                }
                break;
            }
        }
    }

    log_info("=== Training complete ===");
    log_info("Best test accuracy: %.2f%% at epoch %d", best_test_acc, best_epoch);

    /* ---------- Cleanup ---------- */
    free(aug_buf);
    free(dropout_mask);
    tensor_free(pooled);
    tensor_free(dropout_out);
    tensor_free(fc_out_t);
    tensor_free(probs);
    tensor_free(grad_logits);
    tensor_free(grad_fc_in);
    tensor_free(grad_dropout);
    tensor_free(grad_pool_in);
    free(fc_gw); free(fc_gb);
    free(fc_mw); free(fc_vw);
    free(fc_mb); free(fc_vb);
    for (int i = 0; i < num_conv; i++) { free(best_conv_w[i]); free(best_conv_b[i]); }
    free(best_conv_w); free(best_conv_b);
    free(best_fc_w); free(best_fc_b);
    free_fc_config(fc);
    free_conv_layers(conv_layers, num_conv);
    free_dataset(train_ds);
    free_dataset(test_ds);
    free_model_config(cfg);
    log_cleanup();

    return 0;
}
