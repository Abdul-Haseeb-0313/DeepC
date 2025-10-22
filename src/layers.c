#include "layers.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Error handling
#define LAYER_ERROR(msg) do { \
    fprintf(stderr, "\n*** LAYER ERROR ***\n"); \
    fprintf(stderr, "Message: %s\n", msg); \
    fprintf(stderr, "File: %s\n", __FILE__); \
    fprintf(stderr, "Line: %d\n", __LINE__); \
    fprintf(stderr, "Function: %s\n", __func__); \
    exit(EXIT_FAILURE); \
} while(0)

#define LAYER_CHECK(condition, msg) do { \
    if (!(condition)) { \
        LAYER_ERROR(msg); \
    } \
} while(0)

// Create a Dense layer
Layer* Dense(int units, Activation activation, int input_dim) {
    LAYER_CHECK(units > 0, "Units must be positive");
    LAYER_CHECK(input_dim > 0, "Input dimension must be positive");
    
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    LAYER_CHECK(layer != NULL, "Memory allocation failed for layer");
    
    layer->name = "dense";
    layer->activation = activation;
    layer->input_size = input_dim;
    layer->output_size = units;
    layer->next = NULL;
    
    // Initialize weights and biases with correct dimensions
    // weights: [output_size, input_size] - so we can do: output = input * weights^T + bias
    layer->weights = create_matrix(units, input_dim);
    layer->biases = create_matrix(units, 1);
    layer->dweights = create_matrix(units, input_dim);
    layer->dbiases = create_matrix(units, 1);
    
    // Initialize cache matrices
    layer->input = NULL;
    layer->output = NULL;
    layer->z = NULL;
    
    // Initialize weights using Xavier initialization
    initialize_weights_xavier(layer->weights, input_dim);
    
    // Initialize biases to zeros
    for (int i = 0; i < units; i++) {
        layer->biases->data[i][0] = 0.0;
    }
    
    // Initialize gradients to zero
    for (int i = 0; i < units; i++) {
        for (int j = 0; j < input_dim; j++) {
            layer->dweights->data[i][j] = 0.0;
        }
        layer->dbiases->data[i][0] = 0.0;
    }
    
    return layer;
}

// Free layer memory
void free_layer(Layer* layer) {
    if (!layer) return;
    
    if (layer->weights) free_matrix(layer->weights);
    if (layer->biases) free_matrix(layer->biases);
    if (layer->dweights) free_matrix(layer->dweights);
    if (layer->dbiases) free_matrix(layer->dbiases);
    if (layer->input) free_matrix(layer->input);
    if (layer->output) free_matrix(layer->output);
    if (layer->z) free_matrix(layer->z);
    
    free(layer);
}

// CORRECTED Forward pass through a single layer
Matrix* forward_pass(Layer* layer, const Matrix* input) {
    LAYER_CHECK(layer != NULL, "Layer cannot be NULL");
    LAYER_CHECK(input != NULL, "Input matrix cannot be NULL");
    LAYER_CHECK(input->cols == layer->input_size, 
                "Input dimension mismatch: expected %d, got %d");
    
    // Store input for backpropagation
    if (layer->input) free_matrix(layer->input);
    layer->input = copy_matrix(input);
    
    // Calculate z = input * weights^T + bias
    // input: [batch_size, input_size]
    // weights: [output_size, input_size]
    // weights_transpose: [input_size, output_size]
    // z = input * weights_transpose: [batch_size, output_size]
    
    Matrix* weights_transpose = transpose(layer->weights);
    LAYER_CHECK(weights_transpose != NULL, "Failed to transpose weights");
    
    Matrix* z = dot(input, weights_transpose);
    free_matrix(weights_transpose);
    LAYER_CHECK(z != NULL, "Dot product failed in forward pass");
    
    // Add bias to each sample in the batch
    // z: [batch_size, output_size], biases: [output_size, 1]
    for (int i = 0; i < z->rows; i++) {
        for (int j = 0; j < z->cols; j++) {
            z->data[i][j] += layer->biases->data[j][0];
        }
    }
    
    // Store z for backpropagation
    if (layer->z) free_matrix(layer->z);
    layer->z = copy_matrix(z);
    
    // Apply activation function
    Matrix* output = apply_activation(z, layer->activation);
    free_matrix(z);
    
    // Store output
    if (layer->output) free_matrix(layer->output);
    layer->output = copy_matrix(output);
    
    return output;
}

// Backward pass through a single layer
Matrix* backward_pass(Layer* layer, const Matrix* gradient) {
    LAYER_CHECK(layer != NULL, "Layer cannot be NULL");
    LAYER_CHECK(gradient != NULL, "Gradient cannot be NULL");
    LAYER_CHECK(layer->z != NULL, "Layer cache is empty - run forward pass first");
    LAYER_CHECK(layer->input != NULL, "Layer input cache is empty");
    
    // gradient: dL/doutput [batch_size, output_size]
    
    // 1. Compute activation derivative: doutput/dz
    Matrix* activation_deriv = apply_activation_derivative(layer->z, layer->activation);
    
    // 2. Compute delta: dL/dz = dL/doutput * doutput/dz [batch_size, output_size]
    Matrix* delta = create_matrix(gradient->rows, gradient->cols);
    for (int i = 0; i < gradient->rows; i++) {
        for (int j = 0; j < gradient->cols; j++) {
            delta->data[i][j] = gradient->data[i][j] * activation_deriv->data[i][j];
        }
    }
    
    // 3. Compute weight gradients: dL/dW = delta^T * input / batch_size
    Matrix* input_transpose = transpose(layer->input);
    for (int i = 0; i < layer->dweights->rows; i++) {
        for (int j = 0; j < layer->dweights->cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < delta->rows; k++) {
                sum += delta->data[k][i] * input_transpose->data[j][k];
            }
            layer->dweights->data[i][j] = sum / delta->rows;
        }
    }
    
    // 4. Compute bias gradients: dL/db = mean(delta, axis=0)
    for (int i = 0; i < layer->dbiases->rows; i++) {
        double sum = 0.0;
        for (int k = 0; k < delta->rows; k++) {
            sum += delta->data[k][i];
        }
        layer->dbiases->data[i][0] = sum / delta->rows;
    }
    
    // 5. Compute gradient for previous layer: dL/dinput = delta * weights
    Matrix* prev_gradient = dot(delta, layer->weights);
    
    // Cleanup
    free_matrix(activation_deriv);
    free_matrix(delta);
    free_matrix(input_transpose);
    
    return prev_gradient;
}

// Apply activation function to matrix (unchanged, but included for completeness)
Matrix* apply_activation(const Matrix* input, Activation activation) {
    LAYER_CHECK(input != NULL, "Input matrix cannot be NULL");
    
    Matrix* output = create_matrix(input->rows, input->cols);
    LAYER_CHECK(output != NULL, "Failed to create activation output matrix");
    
    switch (activation) {
        case LINEAR:
            for (int i = 0; i < input->rows; i++) {
                for (int j = 0; j < input->cols; j++) {
                    output->data[i][j] = input->data[i][j];
                }
            }
            break;
            
        case SIGMOID:
            for (int i = 0; i < input->rows; i++) {
                for (int j = 0; j < input->cols; j++) {
                    output->data[i][j] = 1.0 / (1.0 + exp(-input->data[i][j]));
                }
            }
            break;
            
        case RELU:
            for (int i = 0; i < input->rows; i++) {
                for (int j = 0; j < input->cols; j++) {
                    output->data[i][j] = input->data[i][j] > 0 ? input->data[i][j] : 0;
                }
            }
            break;
            
        case TANH:
            for (int i = 0; i < input->rows; i++) {
                for (int j = 0; j < input->cols; j++) {
                    output->data[i][j] = tanh(input->data[i][j]);
                }
            }
            break;
            
        case SOFTMAX:
            for (int i = 0; i < input->rows; i++) {
                double max_val = -INFINITY;
                double sum = 0.0;
                
                // Find max for numerical stability
                for (int j = 0; j < input->cols; j++) {
                    if (input->data[i][j] > max_val) {
                        max_val = input->data[i][j];
                    }
                }
                
                // Calculate exponentials and sum
                for (int j = 0; j < input->cols; j++) {
                    output->data[i][j] = exp(input->data[i][j] - max_val);
                    sum += output->data[i][j];
                }
                
                // Normalize
                for (int j = 0; j < input->cols; j++) {
                    output->data[i][j] /= sum;
                }
            }
            break;
    }
    
    return output;
}

// Apply activation derivative
Matrix* apply_activation_derivative(const Matrix* input, Activation activation) {
    LAYER_CHECK(input != NULL, "Input matrix cannot be NULL");
    
    Matrix* derivative = create_matrix(input->rows, input->cols);
    LAYER_CHECK(derivative != NULL, "Failed to create activation derivative matrix");
    
    switch (activation) {
        case LINEAR:
            for (int i = 0; i < input->rows; i++) {
                for (int j = 0; j < input->cols; j++) {
                    derivative->data[i][j] = 1.0;
                }
            }
            break;
            
        case SIGMOID:
            for (int i = 0; i < input->rows; i++) {
                for (int j = 0; j < input->cols; j++) {
                    double sig = 1.0 / (1.0 + exp(-input->data[i][j]));
                    derivative->data[i][j] = sig * (1 - sig);
                }
            }
            break;
            
        case RELU:
            for (int i = 0; i < input->rows; i++) {
                for (int j = 0; j < input->cols; j++) {
                    derivative->data[i][j] = input->data[i][j] > 0 ? 1.0 : 0.0;
                }
            }
            break;
            
        case TANH:
            for (int i = 0; i < input->rows; i++) {
                for (int j = 0; j < input->cols; j++) {
                    double tanh_val = tanh(input->data[i][j]);
                    derivative->data[i][j] = 1 - tanh_val * tanh_val;
                }
            }
            break;
            
        case SOFTMAX:
            // For softmax, we assume the derivative is handled in the loss function
            // This is typically used with categorical crossentropy
            for (int i = 0; i < input->rows; i++) {
                for (int j = 0; j < input->cols; j++) {
                    derivative->data[i][j] = 1.0;
                }
            }
            break;
    }
    
    return derivative;
}

// Initialize weights using Xavier initialization
void initialize_weights_xavier(Matrix* weights, int input_size) {
    LAYER_CHECK(weights != NULL, "Weights matrix cannot be NULL");
    
    static int seeded = 0;
    if (!seeded) {
        srand(time(NULL));
        seeded = 1;
    }
    
    double scale = sqrt(2.0 / (input_size + weights->rows));
    
    for (int i = 0; i < weights->rows; i++) {
        for (int j = 0; j < weights->cols; j++) {
            weights->data[i][j] = ((double)rand() / RAND_MAX - 0.5) * 2 * scale;
        }
    }
}

// Helper function to check matrix for NaN values
int matrix_has_nan(const Matrix* m) {
    if (!m) return 0;
    
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            if (isnan(m->data[i][j])) {
                return 1;
            }
        }
    }
    return 0;
}