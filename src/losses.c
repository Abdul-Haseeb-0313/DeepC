#include "deepc/losses.h"

// Error handling
#define LOSS_ERROR(msg) do { \
    fprintf(stderr, "\n*** LOSS ERROR ***\n"); \
    fprintf(stderr, "Message: %s\n", msg); \
    fprintf(stderr, "File: %s\n", __FILE__); \
    fprintf(stderr, "Line: %d\n", __LINE__); \
    fprintf(stderr, "Function: %s\n", __func__); \
    exit(EXIT_FAILURE); \
} while(0)

#define LOSS_CHECK(condition, msg) do { \
    if (!(condition)) { \
        LOSS_ERROR(msg); \
    } \
} while(0)



// Compute loss based on the specified loss function
double compute_loss(const Matrix* y_true, const Matrix* y_pred, LossFunction loss_func) {
    LOSS_CHECK(y_true != NULL, "True labels cannot be NULL");
    LOSS_CHECK(y_pred != NULL, "Predictions cannot be NULL");
    LOSS_CHECK(y_true->rows == y_pred->rows && y_true->cols == y_pred->cols, 
               "True labels and predictions must have same dimensions");
    LOSS_CHECK(!matrix_has_nan(y_true), "NaN detected in true labels");
    LOSS_CHECK(!matrix_has_nan(y_pred), "NaN detected in predictions");
    
    switch (loss_func) {
        case MEAN_SQUARED_ERROR:
            return mse_loss(y_true, y_pred);
        case BINARY_CROSSENTROPY:
            return binary_crossentropy_loss(y_true, y_pred);
        case CATEGORICAL_CROSSENTROPY:
            return categorical_crossentropy_loss(y_true, y_pred);
        default:
            LOSS_ERROR("Unknown loss function");
    }
    
    return 0.0;
}

// Compute loss gradient based on the specified loss function
Matrix* compute_loss_gradient(const Matrix* y_true, const Matrix* y_pred, LossFunction loss_func) {
    LOSS_CHECK(y_true != NULL, "True labels cannot be NULL");
    LOSS_CHECK(y_pred != NULL, "Predictions cannot be NULL");
    LOSS_CHECK(y_true->rows == y_pred->rows && y_true->cols == y_pred->cols, 
               "True labels and predictions must have same dimensions");
    LOSS_CHECK(!matrix_has_nan(y_true), "NaN detected in true labels");
    LOSS_CHECK(!matrix_has_nan(y_pred), "NaN detected in predictions");
    
    switch (loss_func) {
        case MEAN_SQUARED_ERROR:
            return mse_gradient(y_true, y_pred);
        case BINARY_CROSSENTROPY:
            return binary_crossentropy_gradient(y_true, y_pred);
        case CATEGORICAL_CROSSENTROPY:
            return categorical_crossentropy_gradient(y_true, y_pred);
        default:
            LOSS_ERROR("Unknown loss function");
    }
    
    return NULL;
}

// Mean Squared Error loss
double mse_loss(const Matrix* y_true, const Matrix* y_pred) {
    double sum = 0.0;
    int total_elements = y_true->rows * y_true->cols;
    
    for (int i = 0; i < y_true->rows; i++) {
        for (int j = 0; j < y_true->cols; j++) {
            double diff = y_true->data[i][j] - y_pred->data[i][j];
            sum += diff * diff;
        }
    }
    
    return sum / total_elements;
}

// MSE gradient
Matrix* mse_gradient(const Matrix* y_true, const Matrix* y_pred) {
    Matrix* gradient = create_matrix(y_true->rows, y_true->cols);
    int total_elements = y_true->rows * y_true->cols;
    
    for (int i = 0; i < y_true->rows; i++) {
        for (int j = 0; j < y_true->cols; j++) {
            gradient->data[i][j] = 2.0 * (y_pred->data[i][j] - y_true->data[i][j]) / total_elements;
        }
    }
    
    return gradient;
}

// Binary Cross Entropy loss
double binary_crossentropy_loss(const Matrix* y_true, const Matrix* y_pred) {
    double sum = 0.0;
    int total_elements = y_true->rows * y_true->cols;
    double epsilon = 1e-7;  // To avoid log(0)
    
    for (int i = 0; i < y_true->rows; i++) {
        for (int j = 0; j < y_true->cols; j++) {
            double y_t = y_true->data[i][j];
            double y_p = y_pred->data[i][j];
            
            // Clip predictions to avoid log(0)
            if (y_p < epsilon) y_p = epsilon;
            if (y_p > 1 - epsilon) y_p = 1 - epsilon;
            
            sum += y_t * log(y_p) + (1 - y_t) * log(1 - y_p);
        }
    }
    
    return -sum / total_elements;
}

// Binary Cross Entropy gradient
Matrix* binary_crossentropy_gradient(const Matrix* y_true, const Matrix* y_pred) {
    Matrix* gradient = create_matrix(y_true->rows, y_true->cols);
    int total_elements = y_true->rows * y_true->cols;
    double epsilon = 1e-7;  // To avoid division by zero
    
    for (int i = 0; i < y_true->rows; i++) {
        for (int j = 0; j < y_true->cols; j++) {
            double y_t = y_true->data[i][j];
            double y_p = y_pred->data[i][j];
            
            // Clip predictions to avoid division by zero
            if (y_p < epsilon) y_p = epsilon;
            if (y_p > 1 - epsilon) y_p = 1 - epsilon;
            
            gradient->data[i][j] = (y_p - y_t) / (y_p * (1 - y_p)) / total_elements;
        }
    }
    
    return gradient;
}

// Categorical Cross Entropy loss
double categorical_crossentropy_loss(const Matrix* y_true, const Matrix* y_pred) {
    double sum = 0.0;
    int total_samples = y_true->rows;
    double epsilon = 1e-7;  // To avoid log(0)
    
    for (int i = 0; i < y_true->rows; i++) {
        for (int j = 0; j < y_true->cols; j++) {
            double y_t = y_true->data[i][j];
            double y_p = y_pred->data[i][j];
            
            // Clip predictions to avoid log(0)
            if (y_p < epsilon) y_p = epsilon;
            if (y_p > 1 - epsilon) y_p = 1 - epsilon;
            
            sum += y_t * log(y_p);
        }
    }
    
    return -sum / total_samples;
}

// Categorical Cross Entropy gradient
Matrix* categorical_crossentropy_gradient(const Matrix* y_true, const Matrix* y_pred) {
    Matrix* gradient = create_matrix(y_true->rows, y_true->cols);
    int total_samples = y_true->rows;
    double epsilon = 1e-7;  // To avoid division by zero
    
    for (int i = 0; i < y_true->rows; i++) {
        for (int j = 0; j < y_true->cols; j++) {
            double y_t = y_true->data[i][j];
            double y_p = y_pred->data[i][j];
            
            // Clip predictions to avoid division by zero
            if (y_p < epsilon) y_p = epsilon;
            if (y_p > 1 - epsilon) y_p = 1 - epsilon;
            
            gradient->data[i][j] = (y_p - y_t) / total_samples;
        }
    }
    
    return gradient;
}
