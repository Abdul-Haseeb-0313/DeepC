#include "deepc/models.h"
#include "deepc/layers.h"
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
        if (model->output_layer->output_size != layer->input_size) {
            fprintf(stderr, "Layer dimension mismatch: expected %d, got %d\n",
                    model->output_layer->output_size, layer->input_size);
            MODEL_ERROR("Layer dimension mismatch");
        }
        
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
    if (!model || !X || !y) {
        printf("ERROR: Model or data is NULL in fit\n");
        return;
    }
    
    if (!model->is_compiled) {
        printf("ERROR: Model must be compiled before training\n");
        return;
    }
    
    if (X->rows != y->rows) {
        printf("ERROR: X and y must have same number of samples\n");
        return;
    }
    
    int num_samples = X->rows;
    
    // Use full batch if batch_size is invalid
    if (batch_size <= 0 || batch_size > num_samples) {
        batch_size = num_samples;
    }
    
    int num_batches = (num_samples + batch_size - 1) / batch_size;
    
    if (verbose) {
        printf("Starting training...\n");
        printf("Samples: %d, Batch size: %d, Batches per epoch: %d, Epochs: %d\n",
               num_samples, batch_size, num_batches, epochs);
    }
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        int batches_processed = 0;
        
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
            if (!predictions) {
                printf("ERROR: Forward pass failed in batch %d\n", batch);
                free_matrix(X_batch);
                free_matrix(y_batch);
                continue;
            }
            
            // Compute loss
            double batch_loss = compute_loss(y_batch, predictions, model->loss_function);
            total_loss += batch_loss * current_batch_size;
            batches_processed++;
            
            // Compute gradient
            Matrix* loss_gradient = compute_loss_gradient(y_batch, predictions, model->loss_function);
            
            if (loss_gradient) {
                // Backward pass
                backward_propagation(model, loss_gradient);
                
                // Update weights
                update_model_weights(model);
                
                free_matrix(loss_gradient);
            }
            
            // Cleanup
            free_matrix(predictions);
            free_matrix(X_batch);
            free_matrix(y_batch);
            
            if (verbose && batch % 10 == 0) {
                printf("Epoch %d, Batch %d/%d - Loss: %.6f\n", 
                       epoch + 1, batch + 1, num_batches, batch_loss);
            }
        }
        
        double average_loss = total_loss / num_samples;
        
        if (verbose) {
            printf("Epoch %d/%d - Average Loss: %.6f\n", epoch + 1, epochs, average_loss);
        }
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

// Add these functions to the end of models.c

// Save model to file
void save_model(SequentialModel* model, const char* filename) {
    MODEL_CHECK(model != NULL, "Model cannot be NULL");
    MODEL_CHECK(filename != NULL, "Filename cannot be NULL");
    
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("ERROR: Cannot create file: %s\n", filename);
        return;
    }
    
    // Save model header
    fprintf(file, "DEEPC_MODEL_V2\n");
    fprintf(file, "%s\n", model->name);
    fprintf(file, "%d\n", model->num_layers);
    fprintf(file, "%d\n", model->is_compiled);
    fprintf(file, "%d\n", model->optimizer_type);
    fprintf(file, "%d\n", model->loss_function);
    fprintf(file, "%.17g\n", model->learning_rate);
    
    // Save each layer
    Layer* current = model->input_layer;
    int layer_idx = 0;
    
    while (current) {
        fprintf(file, "LAYER_START\n");
        fprintf(file, "%s\n", current->name);
        fprintf(file, "%d\n", current->input_size);
        fprintf(file, "%d\n", current->output_size);
        fprintf(file, "%d\n", current->activation);
        
        // Save weights
        fprintf(file, "WEIGHTS %d %d\n", current->weights->rows, current->weights->cols);
        for (int i = 0; i < current->weights->rows; i++) {
            for (int j = 0; j < current->weights->cols; j++) {
                fprintf(file, "%.17g\n", current->weights->data[i][j]);
            }
        }
        
        // Save biases
        fprintf(file, "BIASES %d %d\n", current->biases->rows, current->biases->cols);
        for (int i = 0; i < current->biases->rows; i++) {
            for (int j = 0; j < current->biases->cols; j++) {
                fprintf(file, "%.17g\n", current->biases->data[i][j]);
            }
        }
        
        fprintf(file, "LAYER_END\n");
        current = current->next;
        layer_idx++;
    }
    
    fclose(file);
    printf("Model saved: %s\n", filename);
}

SequentialModel* load_model(const char* filename) {
    MODEL_CHECK(filename != NULL, "Filename cannot be NULL");
    
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("ERROR: Cannot open file: %s\n", filename);
        return NULL;
    }
    
    char line[256];
    
    // Check file format
    if (!fgets(line, sizeof(line), file) || strcmp(line, "DEEPC_MODEL_V2\n") != 0) {
        printf("ERROR: Invalid model file format\n");
        fclose(file);
        return NULL;
    }
    
    // Read model header
    fgets(line, sizeof(line), file);
    line[strcspn(line, "\n")] = 0;
    char model_name[256];
    strcpy(model_name, line);
    
    fgets(line, sizeof(line), file);
    int num_layers = atoi(line);
    
    fgets(line, sizeof(line), file);
    int is_compiled = atoi(line);
    
    fgets(line, sizeof(line), file);
    int optimizer_type = atoi(line);
    
    fgets(line, sizeof(line), file);
    int loss_function = atoi(line);
    
    fgets(line, sizeof(line), file);
    double learning_rate = atof(line);
    
    printf("Loading model: %s (%d layers)\n", model_name, num_layers);
    
    // Create model
    SequentialModel* model = create_model(model_name);
    model->num_layers = num_layers;
    model->is_compiled = is_compiled;
    model->optimizer_type = optimizer_type;
    model->loss_function = loss_function;
    model->learning_rate = learning_rate;
    
    // Load layers
    for (int layer_idx = 0; layer_idx < num_layers; layer_idx++) {
        fgets(line, sizeof(line), file); // LAYER_START
        
        // Read layer info
        fgets(line, sizeof(line), file);
        line[strcspn(line, "\n")] = 0;
        char layer_name[256];
        strcpy(layer_name, line);
        
        fgets(line, sizeof(line), file);
        int input_size = atoi(line);
        
        fgets(line, sizeof(line), file);
        int output_size = atoi(line);
        
        fgets(line, sizeof(line), file);
        int activation = atoi(line);
        
        printf("Loading layer %d: %s (%d -> %d)\n", 
               layer_idx + 1, layer_name, input_size, output_size);
        
        // Create layer
        Layer* layer = Dense(output_size, activation, input_size);
        
        // Read weights
        fgets(line, sizeof(line), file); // WEIGHTS dimensions
        int w_rows, w_cols;
        sscanf(line, "WEIGHTS %d %d", &w_rows, &w_cols);
        
        for (int i = 0; i < w_rows; i++) {
            for (int j = 0; j < w_cols; j++) {
                fgets(line, sizeof(line), file);
                layer->weights->data[i][j] = atof(line);
            }
        }
        
        // Read biases
        fgets(line, sizeof(line), file); // BIASES dimensions
        int b_rows, b_cols;
        sscanf(line, "BIASES %d %d", &b_rows, &b_cols);
        
        for (int i = 0; i < b_rows; i++) {
            for (int j = 0; j < b_cols; j++) {
                fgets(line, sizeof(line), file);
                layer->biases->data[i][j] = atof(line);
            }
        }
        
        fgets(line, sizeof(line), file); // LAYER_END
        
        // Add layer to model
        if (model->input_layer == NULL) {
            model->input_layer = layer;
            model->output_layer = layer;
        } else {
            model->output_layer->next = layer;
            model->output_layer = layer;
        }
    }
    
    fclose(file);
    
    // Compile if the model was compiled
    if (is_compiled) {
        model->optimizer = create_optimizer(optimizer_type, learning_rate);
        printf("Model compiled: %s optimizer, %s loss, lr=%.4f\n",
               optimizer_type == SGD ? "SGD" : "Adam",
               loss_function == MEAN_SQUARED_ERROR ? "MSE" :
               loss_function == BINARY_CROSSENTROPY ? "BinaryCE" : "CategoricalCE",
               learning_rate);
    }
    
    printf("Model loaded successfully: %s\n", filename);
    return model;
}

void save_weights(SequentialModel* model, const char* filename) {
    MODEL_CHECK(model != NULL, "Model cannot be NULL");
    MODEL_CHECK(filename != NULL, "Filename cannot be NULL");
    
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("ERROR: Cannot create file: %s\n", filename);
        return;
    }
    
    fprintf(file, "DEEPC_WEIGHTS_V2\n");
    fprintf(file, "%d\n", model->num_layers);
    
    Layer* current = model->input_layer;
    while (current) {
        // Save weights
        fprintf(file, "WEIGHTS %d %d\n", current->weights->rows, current->weights->cols);
        for (int i = 0; i < current->weights->rows; i++) {
            for (int j = 0; j < current->weights->cols; j++) {
                fprintf(file, "%.17g\n", current->weights->data[i][j]);
            }
        }
        
        // Save biases
        fprintf(file, "BIASES %d %d\n", current->biases->rows, current->biases->cols);
        for (int i = 0; i < current->biases->rows; i++) {
            for (int j = 0; j < current->biases->cols; j++) {
                fprintf(file, "%.17g\n", current->biases->data[i][j]);
            }
        }
        
        current = current->next;
    }
    
    fclose(file);
    printf("Weights saved: %s\n", filename);
}

void load_weights(SequentialModel* model, const char* filename) {
    MODEL_CHECK(model != NULL, "Model cannot be NULL");
    MODEL_CHECK(filename != NULL, "Filename cannot be NULL");
    
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("ERROR: Cannot open file: %s\n", filename);
        return;
    }
    
    char line[256];
    
    // Check file format
    if (!fgets(line, sizeof(line), file) || strcmp(line, "DEEPC_WEIGHTS_V2\n") != 0) {
        printf("ERROR: Invalid weights file format\n");
        fclose(file);
        return;
    }
    
    fgets(line, sizeof(line), file);
    int num_layers = atoi(line);
    
    if (num_layers != model->num_layers) {
        printf("ERROR: Layer count mismatch: model has %d, file has %d\n", 
               model->num_layers, num_layers);
        fclose(file);
        return;
    }
    
    Layer* current = model->input_layer;
    for (int i = 0; i < num_layers; i++) {
        if (!current) {
            printf("ERROR: Model has fewer layers than weights file\n");
            fclose(file);
            return;
        }
        
        // Read weights
        fgets(line, sizeof(line), file);
        int w_rows, w_cols;
        sscanf(line, "WEIGHTS %d %d", &w_rows, &w_cols);
        
        if (w_rows != current->weights->rows || w_cols != current->weights->cols) {
            printf("ERROR: Weights dimension mismatch in layer %d\n", i);
            fclose(file);
            return;
        }
        
        for (int i = 0; i < w_rows; i++) {
            for (int j = 0; j < w_cols; j++) {
                fgets(line, sizeof(line), file);
                current->weights->data[i][j] = atof(line);
            }
        }
        
        // Read biases
        fgets(line, sizeof(line), file);
        int b_rows, b_cols;
        sscanf(line, "BIASES %d %d", &b_rows, &b_cols);
        
        if (b_rows != current->biases->rows || b_cols != current->biases->cols) {
            printf("ERROR: Biases dimension mismatch in layer %d\n", i);
            fclose(file);
            return;
        }
        
        for (int i = 0; i < b_rows; i++) {
            for (int j = 0; j < b_cols; j++) {
                fgets(line, sizeof(line), file);
                current->biases->data[i][j] = atof(line);
            }
        }
        
        current = current->next;
    }
    
    fclose(file);
    printf("Weights loaded: %s\n", filename);
}
