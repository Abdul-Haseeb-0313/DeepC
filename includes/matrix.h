#ifndef MATRIX_H
#define MATRIX_H

typedef struct Matrix {
    int rows;
    int cols;
    double* data;

    // Methods (function pointers)
    void (*set)(struct Matrix* self, int row, int col, double value);
    double (*get)(struct Matrix* self, int row, int col);
    struct Matrix* (*filter)(struct Matrix* self, int (*predicate)(double));
    void (*print)(struct Matrix* self);
    void (*free)(struct Matrix* self);
    int (*size)(struct Matrix* self, int* rows, int* cols);
} Matrix;

// Constructor
Matrix* create_matrix(int rows, int cols);

#endif // MATRIX_H
