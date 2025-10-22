#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <execinfo.h>

typedef struct {
    int rows;
    int cols;
    double **data;
} Matrix;

// Function declarations
Matrix* create_matrix(int rows, int cols);
void free_matrix(Matrix *m);
Matrix* copy_matrix(const Matrix *src);
Matrix* zeros(int rows, int cols);
Matrix* ones(int rows, int cols);
Matrix* rand_matrix(int rows, int cols);
void print_matrix(const Matrix *m);
Matrix* get_row(const Matrix *m, int row_index);
Matrix* get_col(const Matrix *m, int col_index);
void set_row(Matrix *m, int row_index, const Matrix *row_data);
void set_col(Matrix *m, int col_index, const Matrix *col_data);
Matrix* add(const Matrix *a, const Matrix *b);
Matrix* subtract(const Matrix *a, const Matrix *b);
Matrix* multiply(const Matrix *a, const Matrix *b);
Matrix* dot(const Matrix *a, const Matrix *b);
Matrix* scale(const Matrix *a, double scalar);
Matrix* transpose(const Matrix *a);
Matrix* apply_function(const Matrix *a, double (*func)(double));
void add_inplace(Matrix *a, const Matrix *b);
void subtract_inplace(Matrix *a, const Matrix *b);
void scale_inplace(Matrix *a, double scalar);

// Activation functions
double sigmoid(double x);
double relu(double x);
double tanh_func(double x);

// Helper functions
void print_stack_trace(void);
int matrix_has_nan(const Matrix* m);

// Add these to matrix.h
Matrix* get_features(const Matrix* data_with_labels, int label_column);
Matrix* get_labels(const Matrix* data_with_labels, int label_column);
void print_class_distribution(const Matrix* labels);

#endif