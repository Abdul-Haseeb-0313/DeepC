#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include "matrix.h"
#include "layers.h"

typedef enum {
    SGD,
    ADAM
} Optimizer;

typedef struct OptimizerState {
    Optimizer type;
    double learning_rate;
    int timestep;
    
    // Adam parameters
    double beta1;
    double beta2;
    double epsilon;
    

    Matrix*** m_weights;  // Array of matrices for each layer
    Matrix*** v_weights;
    Matrix*** m_biases;
    Matrix*** v_biases;
    int max_layers;
} OptimizerState;

// Optimizer management
OptimizerState* create_optimizer(Optimizer type, double learning_rate);
void free_optimizer(OptimizerState* optimizer);

// Weight updates
void update_weights(Layer* layer, OptimizerState* optimizer, int layer_index);

#endif
