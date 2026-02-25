#include "nn_types.h"

void sgd_update(float* weights,const float* grad_weights,int size,float learning_rate){
    if (!weights || !grad_weights) {
        return;
    }

    for (int i = 0; i < size; i++) {
        weights[i] -= learning_rate * grad_weights[i];
    }
}


void sgd_update_bias(float* bias,const float* grad_bias,int size,float learning_rate){
    if (!bias || !grad_bias) {
        return;
    }

    for (int i = 0; i < size; i++) {
        bias[i] -= learning_rate * grad_bias[i];
    }
}


// --- SGD with Momentum ---
// v = momentum * v + grad
// w -= lr * v
void sgd_momentum_update(float* weights, const float* grad_weights, float* velocity,
                         int size, float learning_rate, float momentum) {
    if (!weights || !grad_weights || !velocity) return;
    for (int i = 0; i < size; i++) {
        velocity[i] = momentum * velocity[i] + grad_weights[i];
        weights[i] -= learning_rate * velocity[i];
    }
}

void sgd_momentum_update_bias(float* bias, const float* grad_bias, float* velocity,
                              int size, float learning_rate, float momentum) {
    if (!bias || !grad_bias || !velocity) return;
    for (int i = 0; i < size; i++) {
        velocity[i] = momentum * velocity[i] + grad_bias[i];
        bias[i] -= learning_rate * velocity[i];
    }
}

// --- Learning Rate Schedule: warmup + exponential decay ---
float lr_schedule(float base_lr, int epoch) {
    if (epoch < 5) {
        // Linear warmup: 0.1*base_lr -> base_lr over 5 epochs
        return base_lr * (0.1f + 0.9f * ((float)epoch / 5.0f));
    } else {
        // Exponential decay: base_lr * 0.95^(epoch - 5)
        return base_lr * powf(0.95f, (float)(epoch - 5));
    }
}
