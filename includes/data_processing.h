#ifndef CSV_LOADER_H
#define CSV_LOADER_H

#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

// Simple CSV loading
Matrix* load_csv(const char *filename, int has_header);
void fill_missing_with_mean(Matrix *m);
void fill_missing_with_zeros(Matrix *m);
int count_missing_values(const Matrix *m);
void print_matrix_stats(const Matrix *m);

// Train-test split
void train_test_split(const Matrix *X, const Matrix *y, double test_size, 
                     Matrix **X_train, Matrix **X_test, 
                     Matrix **y_train, Matrix **y_test);

// BUILT-IN DATA PROCESSING FUNCTIONS
Matrix* split_features_labels(const Matrix *data, int label_column);
Matrix* one_hot_encode_labels(const Matrix *labels, int num_classes);
Matrix* normalize_matrix(Matrix *X);
Matrix* standardize_matrix(Matrix *X);
void shuffle_dataset(Matrix *X, Matrix *y);

#endif