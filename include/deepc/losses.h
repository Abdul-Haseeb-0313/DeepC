#ifndef LOSSES_H
#define LOSSES_H

#include "matrix.h"

// Loss function types
typedef enum {
    MEAN_SQUARED_ERROR,
    BINARY_CROSSENTROPY,
    CATEGORICAL_CROSSENTROPY
} LossFunction;

// Loss calculation
double compute_loss(const Matrix* y_true, const Matrix* y_pred, LossFunction loss_func);
Matrix* compute_loss_gradient(const Matrix* y_true, const Matrix* y_pred, LossFunction loss_func);

// Individual loss functions
double mse_loss(const Matrix* y_true, const Matrix* y_pred);
Matrix* mse_gradient(const Matrix* y_true, const Matrix* y_pred);

double binary_crossentropy_loss(const Matrix* y_true, const Matrix* y_pred);
Matrix* binary_crossentropy_gradient(const Matrix* y_true, const Matrix* y_pred);

double categorical_crossentropy_loss(const Matrix* y_true, const Matrix* y_pred);
Matrix* categorical_crossentropy_gradient(const Matrix* y_true, const Matrix* y_pred);



#endif