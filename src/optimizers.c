#include "optimizers.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// Error handling
#define OPTIMIZER_ERROR(msg) do { \
    fprintf(stderr, "\n*** OPTIMIZER ERROR ***\n"); \
    fprintf(stderr, "Message: %s\n", msg); \
    fprintf(stderr, "File: %s\n", __FILE__); \
    fprintf(stderr, "Line: %d\n", __LINE__); \
    fprintf(stderr, "Function: %s\n", __func__); \
    exit(EXIT_FAILURE); \
} while(0)

#define OPTIMIZER_CHECK(condition, msg) do { \
    if (!(condition)) { \
        OPTIMIZER_ERROR(msg); \
    } \
} while(0)

// Create optimizer
OptimizerState* create_optimizer(Optimizer type, double learning_rate) {
    OPTIMIZER_CHECK(learning_rate > 0, "Learning rate must be positive");
    
    OptimizerState* optimizer = (OptimizerState*)malloc(sizeof(OptimizerState));
    OPTIMIZER_CHECK(optimizer != NULL, "Memory allocation failed for optimizer");
    
    optimizer->type = type;
    optimizer->learning_rate = learning_rate;
    optimizer->timestep = 0;
    optimizer->max_layers = 100; // Support up to 100 layers
    
    if (type == ADAM) {
        optimizer->beta1 = 0.9;
        optimizer->beta2 = 0.999;
        optimizer->epsilon = 1e-8;
        
        // Allocate arrays for moments
        optimizer->m_weights = (Matrix***)calloc(optimizer->max_layers, sizeof(Matrix**));
        optimizer->v_weights = (Matrix***)calloc(optimizer->max_layers, sizeof(Matrix**));
        optimizer->m_biases = (Matrix***)calloc(optimizer->max_layers, sizeof(Matrix**));
        optimizer->v_biases = (Matrix***)calloc(optimizer->max_layers, sizeof(Matrix**));
    } else {
        optimizer->m_weights = NULL;
        optimizer->v_weights = NULL;
        optimizer->m_biases = NULL;
        optimizer->v_biases = NULL;
    }
    
    return optimizer;
}

// Free optimizer memory
void free_optimizer(OptimizerState* optimizer) {
    if (!optimizer) return;
    
    if (optimizer->m_weights) {
        for (int i = 0; i < optimizer->max_layers; i++) {
            if (optimizer->m_weights[i]) free_matrix(*optimizer->m_weights[i]);
            if (optimizer->v_weights[i]) free_matrix(*optimizer->v_weights[i]);
            if (optimizer->m_biases[i]) free_matrix(*optimizer->m_biases[i]);
            if (optimizer->v_biases[i]) free_matrix(*optimizer->v_biases[i]);
        }
        free(optimizer->m_weights);
        free(optimizer->v_weights);
        free(optimizer->m_biases);
        free(optimizer->v_biases);
    }
    
    free(optimizer);
}

// Initialize Adam moments for a layer
void initialize_adam_moments(OptimizerState* optimizer, Layer* layer, int layer_index) {
    OPTIMIZER_CHECK(optimizer != NULL, "Optimizer cannot be NULL");
    OPTIMIZER_CHECK(layer != NULL, "Layer cannot be NULL");
    OPTIMIZER_CHECK(layer_index < optimizer->max_layers, "Layer index out of bounds");
    
    if (optimizer->m_weights[layer_index] == NULL) {
        optimizer->m_weights[layer_index] = (Matrix**)malloc(sizeof(Matrix*));
        optimizer->v_weights[layer_index] = (Matrix**)malloc(sizeof(Matrix*));
        optimizer->m_biases[layer_index] = (Matrix**)malloc(sizeof(Matrix*));
        optimizer->v_biases[layer_index] = (Matrix**)malloc(sizeof(Matrix*));
        
        *optimizer->m_weights[layer_index] = create_matrix(layer->weights->rows, layer->weights->cols);
        *optimizer->v_weights[layer_index] = create_matrix(layer->weights->rows, layer->weights->cols);
        *optimizer->m_biases[layer_index] = create_matrix(layer->biases->rows, layer->biases->cols);
        *optimizer->v_biases[layer_index] = create_matrix(layer->biases->rows, layer->biases->cols);
        
        // Initialize to zeros
        for (int i = 0; i < layer->weights->rows; i++) {
            for (int j = 0; j < layer->weights->cols; j++) {
                (*optimizer->m_weights[layer_index])->data[i][j] = 0.0;
                (*optimizer->v_weights[layer_index])->data[i][j] = 0.0;
            }
        }
        for (int i = 0; i < layer->biases->rows; i++) {
            (*optimizer->m_biases[layer_index])->data[i][0] = 0.0;
            (*optimizer->v_biases[layer_index])->data[i][0] = 0.0;
        }
    }
}

// Update weights using SGD
void update_weights_sgd(Layer* layer, OptimizerState* optimizer) {
    OPTIMIZER_CHECK(layer != NULL, "Layer cannot be NULL");
    OPTIMIZER_CHECK(optimizer != NULL, "Optimizer cannot be NULL");
    
    // Update weights: w = w - lr * dw
    for (int i = 0; i < layer->weights->rows; i++) {
        for (int j = 0; j < layer->weights->cols; j++) {
            layer->weights->data[i][j] -= optimizer->learning_rate * layer->dweights->data[i][j];
        }
    }
    
    // Update biases: b = b - lr * db
    for (int i = 0; i < layer->biases->rows; i++) {
        layer->biases->data[i][0] -= optimizer->learning_rate * layer->dbiases->data[i][0];
    }
}

// Update weights using Adam
void update_weights_adam(Layer* layer, OptimizerState* optimizer, int layer_index) {
    OPTIMIZER_CHECK(layer != NULL, "Layer cannot be NULL");
    OPTIMIZER_CHECK(optimizer != NULL, "Optimizer cannot be NULL");
    
    // Initialize moments if needed
    initialize_adam_moments(optimizer, layer, layer_index);
    
    optimizer->timestep++;
    
    double beta1 = optimizer->beta1;
    double beta2 = optimizer->beta2;
    double epsilon = optimizer->epsilon;
    double lr = optimizer->learning_rate;
    
    Matrix* m_weights = *optimizer->m_weights[layer_index];
    Matrix* v_weights = *optimizer->v_weights[layer_index];
    Matrix* m_biases = *optimizer->m_biases[layer_index];
    Matrix* v_biases = *optimizer->v_biases[layer_index];
    
    // Update weights
    for (int i = 0; i < layer->weights->rows; i++) {
        for (int j = 0; j < layer->weights->cols; j++) {
            double grad = layer->dweights->data[i][j];
            
            // Update biased first moment estimate
            m_weights->data[i][j] = beta1 * m_weights->data[i][j] + (1 - beta1) * grad;
            // Update biased second raw moment estimate
            v_weights->data[i][j] = beta2 * v_weights->data[i][j] + (1 - beta2) * grad * grad;
            
            // Compute bias-corrected first moment estimate
            double m_hat = m_weights->data[i][j] / (1 - pow(beta1, optimizer->timestep));
            // Compute bias-corrected second raw moment estimate
            double v_hat = v_weights->data[i][j] / (1 - pow(beta2, optimizer->timestep));
            
            // Update parameters
            layer->weights->data[i][j] -= lr * m_hat / (sqrt(v_hat) + epsilon);
        }
    }
    
    // Update biases
    for (int i = 0; i < layer->biases->rows; i++) {
        double grad = layer->dbiases->data[i][0];
        
        // Update biased first moment estimate
        m_biases->data[i][0] = beta1 * m_biases->data[i][0] + (1 - beta1) * grad;
        // Update biased second raw moment estimate
        v_biases->data[i][0] = beta2 * v_biases->data[i][0] + (1 - beta2) * grad * grad;
        
        // Compute bias-corrected first moment estimate
        double m_hat = m_biases->data[i][0] / (1 - pow(beta1, optimizer->timestep));
        // Compute bias-corrected second raw moment estimate
        double v_hat = v_biases->data[i][0] / (1 - pow(beta2, optimizer->timestep));
        
        // Update parameters
        layer->biases->data[i][0] -= lr * m_hat / (sqrt(v_hat) + epsilon);
    }
}

// Update weights based on optimizer type
void update_weights(Layer* layer, OptimizerState* optimizer, int layer_index) {
    OPTIMIZER_CHECK(layer != NULL, "Layer cannot be NULL");
    OPTIMIZER_CHECK(optimizer != NULL, "Optimizer cannot be NULL");
    
    switch (optimizer->type) {
        case SGD:
            update_weights_sgd(layer, optimizer);
            break;
        case ADAM:
            update_weights_adam(layer, optimizer, layer_index);
            break;
        default:
            OPTIMIZER_ERROR("Unknown optimizer type");
    }
}