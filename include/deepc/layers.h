#ifndef LAYERS_H
#define LAYERS_H

#include "matrix.h"

typedef enum {
    LINEAR,
    SIGMOID,
    RELU,
    TANH,
    SOFTMAX
} Activation;

typedef struct Layer {
    char* name;
    Activation activation;
    int input_size;
    int output_size;
    struct Layer* next;
    
    // Parameters and gradients
    Matrix* weights;
    Matrix* biases;
    Matrix* dweights;
    Matrix* dbiases;
    
    // Forward pass cache
    Matrix* input;
    Matrix* output;
    Matrix* z;
} Layer;

// Layer creation
Layer* Dense(int units, Activation activation, int input_dim);

// Forward and backward passes
Matrix* forward_pass(Layer* layer, const Matrix* input);
Matrix* backward_pass(Layer* layer, const Matrix* gradient);

// Activation functions
Matrix* apply_activation(const Matrix* input, Activation activation);
Matrix* apply_activation_derivative(const Matrix* input, Activation activation);

// Initialization
void initialize_weights_xavier(Matrix* weights, int input_size);

// Memory management
void free_layer(Layer* layer);

// Utility
int matrix_has_nan(const Matrix* m);


// Layer serialization
void save_layer(Layer* layer, FILE* file);
Layer* load_layer(FILE* file);

#endif
