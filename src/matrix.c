#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

// ===== Helper Functions =====
void matrix_set(Matrix* self, int row, int col, double value) {
    self->data[row * self->cols + col] = value;
}

double matrix_get(Matrix* self, int row, int col) {
    return self->data[row * self->cols + col];
}

Matrix* matrix_filter(Matrix* self, int (*predicate)(double)) {
    Matrix* result = create_matrix(1, self->rows * self->cols);
    int k = 0;
    for (int i = 0; i < self->rows * self->cols; i++) {
        double val = self->data[i];
        if (predicate(val)) {
            result->data[k++] = val;
        }
    }
    result->cols = k; // shrink to actual size
    return result;
}

void matrix_print(Matrix* self) {
    for (int i = 0; i < self->rows; i++) {
        for (int j = 0; j < self->cols; j++) {
            printf("%6.2f ", self->data[i * self->cols + j]);
        }
        printf("\n");
    }
}

void matrix_free(Matrix* self) {
    free(self->data);
    free(self);
}

int matrix_size(Matrix* self, int* rows, int* cols) {
    if (rows) *rows = self->rows;
    if (cols) *cols = self->cols;
    return self->rows * self->cols;
}

// ===== Constructor =====
Matrix* create_matrix(int rows, int cols) {
    Matrix* m = (Matrix*)malloc(sizeof(Matrix));
    m->rows = rows;
    m->cols = cols;
    m->data = (double*)calloc(rows * cols, sizeof(double));

    // Assign methods
    m->set = matrix_set;
    m->get = matrix_get;
    m->filter = matrix_filter;
    m->print = matrix_print;
    m->free = matrix_free;
    m->size = matrix_size;

    return m;
}
