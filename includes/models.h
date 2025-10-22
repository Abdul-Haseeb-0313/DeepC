#ifndef MODELS_H
#define MODELS_H

#include "matrix.h"
#include "layers.h"
#include "losses.h"
#include "optimizers.h"

typedef struct SequentialModel {
    char* name;
    Layer* input_layer;
    Layer* output_layer;
    int num_layers;
    
    // Training parameters
    double learning_rate;
    LossFunction loss_function;
    Optimizer optimizer_type;
    OptimizerState* optimizer;
    int is_compiled;
} SequentialModel;

// Model creation and management
SequentialModel* create_model(const char* name);
void add_layer(SequentialModel* model, Layer* layer);
void free_model(SequentialModel* model);

// Model compilation and training
void compile(SequentialModel* model, Optimizer optimizer, LossFunction loss, double learning_rate);
Matrix* predict(SequentialModel* model, const Matrix* input);
void fit(SequentialModel* model, const Matrix* X, const Matrix* y, 
         int epochs, int batch_size, int verbose);

// Model evaluation
double evaluate(SequentialModel* model, const Matrix* X, const Matrix* y);

// Utility functions
void print_model_summary(const SequentialModel* model);


// BUILT-IN SAVE/LOAD FUNCTIONS
void save_model(SequentialModel* model, const char* filename);
SequentialModel* load_model(const char* filename);
void save_weights(SequentialModel* model, const char* filename);
void load_weights(SequentialModel* model, const char* filename);

#endif