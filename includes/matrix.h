#ifndef MATRIX_H
#define MATRIX_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <execinfo.h>
#include <unistd.h>

// Matrix structure
typedef struct {
    double **data;
    int rows;
    int cols;
} Matrix;

// Stack trace function
void print_stack_trace(void);

// Creation and destruction
Matrix* create_matrix(int rows, int cols);
void free_matrix(Matrix *m);
Matrix* copy_matrix(const Matrix *src);

// Utility functions
Matrix* zeros(int rows, int cols);
Matrix* ones(int rows, int cols);
Matrix* rand_matrix(int rows, int cols);
void print_matrix(const Matrix *m);

// Row and column operations
Matrix* get_row(const Matrix *m, int row_index);
Matrix* get_col(const Matrix *m, int col_index);
void set_row(Matrix *m, int row_index, const Matrix *row_data);
void set_col(Matrix *m, int col_index, const Matrix *col_data);

// Basic linear algebra operations
Matrix* add(const Matrix *a, const Matrix *b);
Matrix* subtract(const Matrix *a, const Matrix *b);
Matrix* multiply(const Matrix *a, const Matrix *b);
Matrix* dot(const Matrix *a, const Matrix *b);
Matrix* scale(const Matrix *a, double scalar);
Matrix* transpose(const Matrix *a);
Matrix* apply_function(const Matrix *a, double (*func)(double));

// In-place operations (for efficiency)
void add_inplace(Matrix *a, const Matrix *b);
void subtract_inplace(Matrix *a, const Matrix *b);
void scale_inplace(Matrix *a, double scalar);

// Utility math functions for activation functions
double sigmoid(double x);
double relu(double x);
double tanh_func(double x);

#endif