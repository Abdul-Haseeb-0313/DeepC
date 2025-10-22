#include "models.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Error handling
#define MODEL_ERROR(msg) do { \
    fprintf(stderr, "\n*** MODEL ERROR ***\n"); \
    fprintf(stderr, "Message: %s\n", msg); \
    fprintf(stderr, "File: %s\n", __FILE__); \
    fprintf(stderr, "Line: %d\n", __LINE__); \
    fprintf(stderr, "Function: %s\n", __func__); \
    exit(EXIT_FAILURE); \
} while(0)

#define MODEL_CHECK(condition, msg) do { \
    if (!(condition)) { \
        MODEL_ERROR(msg); \
    } \
} while(0)

// Store layers in array for easy access during backpropagation
typedef struct {
    Layer** layers;
    int count;
} LayerArray;

LayerArray get_layers_array(SequentialModel* model) {
    LayerArray array;
    array.count = model->num_layers;
    array.layers = (Layer**)malloc(array.count * sizeof(Layer*));
    
    Layer* current = model->input_layer;
    for (int i = 0; i < array.count; i++) {
        array.layers[i] = current;
        current = current->next;
    }
    
    return array;
}

// Complete backpropagation through all layers
void backward_propagation(SequentialModel* model, const Matrix* loss_gradient) {
    MODEL_CHECK(model != NULL, "Model cannot be NULL");
    MODEL_CHECK(loss_gradient != NULL, "Loss gradient cannot be NULL");
    
    // Get layers in reverse order for backpropagation
    LayerArray array = get_layers_array(model);
    
    Matrix* gradient = copy_matrix(loss_gradient);
    
    // Backward pass through layers in reverse order
    for (int i = array.count - 1; i >= 0; i--) {
        Matrix* prev_gradient = backward_pass(array.layers[i], gradient);
        free_matrix(gradient);
        gradient = prev_gradient;
    }
    
    free_matrix(gradient);
    free(array.layers);
}

// Update all layers' weights
void update_model_weights(SequentialModel* model) {
    MODEL_CHECK(model != NULL, "Model cannot be NULL");
    MODEL_CHECK(model->optimizer != NULL, "Optimizer cannot be NULL");
    
    Layer* current = model->input_layer;
    int layer_index = 0;
    
    while (current) {
        update_weights(current, model->optimizer, layer_index);
        current = current->next;
        layer_index++;
    }
}

// Create a sequential model
SequentialModel* create_model(const char* name) {
    SequentialModel* model = (SequentialModel*)malloc(sizeof(SequentialModel));
    MODEL_CHECK(model != NULL, "Memory allocation failed for model");
    
    model->name = name ? strdup(name) : strdup("sequential_model");
    model->input_layer = NULL;
    model->output_layer = NULL;
    model->num_layers = 0;
    model->learning_rate = 0.01;
    model->loss_function = MEAN_SQUARED_ERROR;
    model->optimizer_type = SGD;
    model->optimizer = NULL;
    model->is_compiled = 0;
    
    return model;
}

// Add layer to model
void add_layer(SequentialModel* model, Layer* layer) {
    MODEL_CHECK(model != NULL, "Model cannot be NULL");
    MODEL_CHECK(layer != NULL, "Layer cannot be NULL");
    
    if (model->input_layer == NULL) {
        // First layer
        model->input_layer = layer;
        model->output_layer = layer;
    } else {
        // Check dimension compatibility
        MODEL_CHECK(model->output_layer->output_size == layer->input_size,
                   "Layer dimension mismatch");
        
        model->output_layer->next = layer;
        model->output_layer = layer;
    }
    
    model->num_layers++;
}

// Free model memory
void free_model(SequentialModel* model) {
    if (!model) return;
    
    Layer* current = model->input_layer;
    while (current) {
        Layer* next = current->next;
        free_layer(current);
        current = next;
    }
    
    if (model->optimizer) free_optimizer(model->optimizer);
    if (model->name) free(model->name);
    free(model);
}

// Compile the model
void compile(SequentialModel* model, Optimizer optimizer, LossFunction loss, double learning_rate) {
    MODEL_CHECK(model != NULL, "Model cannot be NULL");
    MODEL_CHECK(model->input_layer != NULL, "Model has no layers");
    MODEL_CHECK(learning_rate > 0, "Learning rate must be positive");
    
    model->optimizer_type = optimizer;
    model->loss_function = loss;
    model->learning_rate = learning_rate;
    model->optimizer = create_optimizer(optimizer, learning_rate);
    model->is_compiled = 1;
    
    printf("Model compiled successfully!\n");
    printf("  Name: %s\n", model->name);
    printf("  Layers: %d\n", model->num_layers);
    printf("  Optimizer: %s\n", optimizer == SGD ? "SGD" : "Adam");
    printf("  Loss: %s\n", 
           loss == MEAN_SQUARED_ERROR ? "MSE" : 
           loss == BINARY_CROSSENTROPY ? "BinaryCE" : "CategoricalCE");
    printf("  Learning rate: %.4f\n", learning_rate);
}

// Predict using the entire model
Matrix* predict(SequentialModel* model, const Matrix* input) {
    MODEL_CHECK(model != NULL, "Model cannot be NULL");
    MODEL_CHECK(input != NULL, "Input matrix cannot be NULL");
    MODEL_CHECK(model->input_layer != NULL, "Model has no layers");
    
    Matrix* current_output = copy_matrix(input);
    Layer* current_layer = model->input_layer;
    
    while (current_layer) {
        Matrix* next_output = forward_pass(current_layer, current_output);
        free_matrix(current_output);
        current_output = next_output;
        current_layer = current_layer->next;
    }
    
    return current_output;
}

// Train the model
void fit(SequentialModel* model, const Matrix* X, const Matrix* y, 
         int epochs, int batch_size, int verbose) {
    MODEL_CHECK(model != NULL, "Model cannot be NULL");
    MODEL_CHECK(X != NULL && y != NULL, "Data cannot be NULL");
    MODEL_CHECK(model->is_compiled, "Model must be compiled before training");
    MODEL_CHECK(X->rows == y->rows, "X and y must have same number of samples");
    
    int num_samples = X->rows;
    
    if (batch_size <= 0 || batch_size > num_samples) {
        batch_size = num_samples;
    }
    
    int num_batches = (num_samples + batch_size - 1) / batch_size;
    
    if (verbose) {
        printf("Starting training...\n");
        printf("  Samples: %d\n", num_samples);
        printf("  Batch size: %d\n", batch_size);
        printf("  Batches per epoch: %d\n", num_batches);
        printf("  Epochs: %d\n", epochs);
    }
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        
        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * batch_size;
            int end_idx = (batch + 1) * batch_size;
            if (end_idx > num_samples) end_idx = num_samples;
            
            int current_batch_size = end_idx - start_idx;
            
            // Create batch data
            Matrix* X_batch = create_matrix(current_batch_size, X->cols);
            Matrix* y_batch = create_matrix(current_batch_size, y->cols);
            
            for (int i = 0; i < current_batch_size; i++) {
                for (int j = 0; j < X->cols; j++) {
                    X_batch->data[i][j] = X->data[start_idx + i][j];
                }
                for (int j = 0; j < y->cols; j++) {
                    y_batch->data[i][j] = y->data[start_idx + i][j];
                }
            }
            
            // Forward pass
            Matrix* predictions = predict(model, X_batch);
            
            // Compute loss and gradient
            double batch_loss = compute_loss(y_batch, predictions, model->loss_function);
            total_loss += batch_loss * current_batch_size;
            
            Matrix* loss_gradient = compute_loss_gradient(y_batch, predictions, model->loss_function);
            
            // Backward pass
            backward_propagation(model, loss_gradient);
            
            // Update weights
            update_model_weights(model);
            
            // Cleanup
            free_matrix(predictions);
            free_matrix(loss_gradient);
            free_matrix(X_batch);
            free_matrix(y_batch);
        }
        
        double average_loss = total_loss / num_samples;
        
        if (verbose && (epoch % 10 == 0 || epoch == epochs - 1)) {
            printf("Epoch %d/%d - Loss: %.6f\n", epoch + 1, epochs, average_loss);
        }
        
    }
    
    if (verbose) {
        printf("Training completed!\n");
    }
}

// Evaluate model on test data
double evaluate(SequentialModel* model, const Matrix* X, const Matrix* y) {
    MODEL_CHECK(model != NULL, "Model cannot be NULL");
    MODEL_CHECK(X != NULL && y != NULL, "Data cannot be NULL");
    
    Matrix* predictions = predict(model, X);
    double loss = compute_loss(y, predictions, model->loss_function);
    
    free_matrix(predictions);
    return loss;
}

// Print model summary
void print_model_summary(const SequentialModel* model) {
    MODEL_CHECK(model != NULL, "Model cannot be NULL");
    
    printf("\n=== Model Summary: %s ===\n", model->name);
    printf("Layers: %d\n", model->num_layers);
    
    if (model->is_compiled) {
        printf("Compiled: Yes\n");
        printf("Optimizer: %s\n", model->optimizer_type == SGD ? "SGD" : "Adam");
        printf("Loss: %s\n", 
               model->loss_function == MEAN_SQUARED_ERROR ? "MSE" : 
               model->loss_function == BINARY_CROSSENTROPY ? "BinaryCE" : "CategoricalCE");
        printf("Learning rate: %.4f\n", model->learning_rate);
    } else {
        printf("Compiled: No\n");
    }
    
    Layer* current = model->input_layer;
    int layer_num = 1;
    int total_params = 0;
    
    printf("\nLayer Details:\n");
    printf("-------------------------------------------------\n");
    while (current) {
        int weights_params = current->weights->rows * current->weights->cols;
        int bias_params = current->biases->rows;
        int layer_params = weights_params + bias_params;
        total_params += layer_params;
        
        printf("Layer %d: Dense(%d -> %d) ", 
               layer_num, current->input_size, current->output_size);
        
        switch (current->activation) {
            case LINEAR: printf("Linear"); break;
            case SIGMOID: printf("Sigmoid"); break;
            case RELU: printf("ReLU"); break;
            case TANH: printf("Tanh"); break;
            case SOFTMAX: printf("Softmax"); break;
        }
        
        printf(" - Params: %d\n", layer_params);
        
        current = current->next;
        layer_num++;
    }
    
    printf("-------------------------------------------------\n");
    printf("Total parameters: %d\n", total_params);
    printf("=================================================\n\n");
}