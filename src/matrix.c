#include "matrix.h"

// Stack trace function
void print_stack_trace(void) {
#ifdef EXECINFO_AVAILABLE
    void *buffer[100];
    char **strings;
    int nptrs;
    
    printf("\n=== STACK TRACE ===\n");
    nptrs = backtrace(buffer, 100);
    strings = backtrace_symbols(buffer, nptrs);
    
    if (strings == NULL) {
        perror("backtrace_symbols");
        return;
    }
    
    for (int i = 1; i < nptrs; i++) {
        printf("#%d %s\n", i-1, strings[i]);
    }
    
    free(strings);
    printf("===================\n\n");
#endif
}

// Error handling macro with stack trace
#define MATRIX_ERROR(msg) do { \
    fprintf(stderr, "\n*** MATRIX ERROR ***\n"); \
    fprintf(stderr, "Message: %s\n", msg); \
    fprintf(stderr, "File: %s\n", __FILE__); \
    fprintf(stderr, "Line: %d\n", __LINE__); \
    fprintf(stderr, "Function: %s\n", __func__); \
    print_stack_trace(); \
    exit(EXIT_FAILURE); \
} while(0)

#define MATRIX_CHECK(condition, msg) do { \
    if (!(condition)) { \
        MATRIX_ERROR(msg); \
    } \
} while(0)

// Create a new matrix
Matrix* create_matrix(int rows, int cols) {
    MATRIX_CHECK(rows > 0 && cols > 0, "Matrix dimensions must be positive");
    
    Matrix *m = (Matrix*)malloc(sizeof(Matrix));
    MATRIX_CHECK(m != NULL, "Memory allocation failed for matrix structure");
    
    m->rows = rows;
    m->cols = cols;
    
    // Allocate memory for row pointers
    m->data = (double**)malloc(rows * sizeof(double*));
    MATRIX_CHECK(m->data != NULL, "Memory allocation failed for matrix rows");
    
    // Allocate memory for each row
    for (int i = 0; i < rows; i++) {
        m->data[i] = (double*)malloc(cols * sizeof(double));
        if (!m->data[i]) {
            // Free previously allocated memory before terminating
            for (int j = 0; j < i; j++) {
                free(m->data[j]);
            }
            free(m->data);
            free(m);
            MATRIX_ERROR("Memory allocation failed for matrix row");
        }
        
        // Initialize to zero
        for (int j = 0; j < cols; j++) {
            m->data[i][j] = 0.0;
        }
    }
    
    return m;
}

// Free matrix memory
void free_matrix(Matrix *m) {
    if (m) {
        if (m->data) {
            for (int i = 0; i < m->rows; i++) {
                if (m->data[i]) {
                    free(m->data[i]);
                }
            }
            free(m->data);
        }
        free(m);
    }
}

// Create a deep copy of a matrix
Matrix* copy_matrix(const Matrix *src) {
    MATRIX_CHECK(src != NULL, "Source matrix is NULL");
    
    Matrix *dest = create_matrix(src->rows, src->cols);
    
    for (int i = 0; i < src->rows; i++) {
        for (int j = 0; j < src->cols; j++) {
            dest->data[i][j] = src->data[i][j];
        }
    }
    
    return dest;
}

// Create matrix filled with zeros
Matrix* zeros(int rows, int cols) {
    return create_matrix(rows, cols); // Already initialized to zeros
}

// Create matrix filled with ones
Matrix* ones(int rows, int cols) {
    Matrix *m = create_matrix(rows, cols);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            m->data[i][j] = 1.0;
        }
    }
    
    return m;
}

// Create matrix with random values between 0 and 1
Matrix* rand_matrix(int rows, int cols) {
    Matrix *m = create_matrix(rows, cols);
    
    // Seed random number generator only once
    static int seeded = 0;
    if (!seeded) {
        srand(time(NULL));
        seeded = 1;
    }
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            m->data[i][j] = (double)rand() / RAND_MAX;
        }
    }
    
    return m;
}

// Print matrix
void print_matrix(const Matrix *m) {
    MATRIX_CHECK(m != NULL, "Matrix is NULL");
    
    printf("Matrix (%d x %d):\n", m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            if (isnan(m->data[i][j])) {
                printf("     NaN ");
            } else {
                printf("%8.4f ", m->data[i][j]);
            }
        }
        printf("\n");
    }
    printf("\n");
}

// Get a specific row as a new 1 x cols matrix
Matrix* get_row(const Matrix *m, int row_index) {
    MATRIX_CHECK(m != NULL, "Matrix is NULL");
    MATRIX_CHECK(row_index >= 0 && row_index < m->rows, "Row index out of bounds");
    
    Matrix *row = create_matrix(1, m->cols);
    
    for (int j = 0; j < m->cols; j++) {
        row->data[0][j] = m->data[row_index][j];
    }
    
    return row;
}

// Get a specific column as a new rows x 1 matrix
Matrix* get_col(const Matrix *m, int col_index) {
    MATRIX_CHECK(m != NULL, "Matrix is NULL");
    MATRIX_CHECK(col_index >= 0 && col_index < m->cols, "Column index out of bounds");
    
    Matrix *col = create_matrix(m->rows, 1);
    
    for (int i = 0; i < m->rows; i++) {
        col->data[i][0] = m->data[i][col_index];
    }
    
    return col;
}

// Set a specific row from a 1 x cols matrix
void set_row(Matrix *m, int row_index, const Matrix *row_data) {
    MATRIX_CHECK(m != NULL, "Matrix is NULL");
    MATRIX_CHECK(row_data != NULL, "Row data is NULL");
    MATRIX_CHECK(row_index >= 0 && row_index < m->rows, "Row index out of bounds");
    MATRIX_CHECK(row_data->rows == 1 && row_data->cols == m->cols, 
                "Row data dimensions don't match");
    
    for (int j = 0; j < m->cols; j++) {
        m->data[row_index][j] = row_data->data[0][j];
    }
}

// Set a specific column from a rows x 1 matrix
void set_col(Matrix *m, int col_index, const Matrix *col_data) {
    MATRIX_CHECK(m != NULL, "Matrix is NULL");
    MATRIX_CHECK(col_data != NULL, "Column data is NULL");
    MATRIX_CHECK(col_index >= 0 && col_index < m->cols, "Column index out of bounds");
    MATRIX_CHECK(col_data->cols == 1 && col_data->rows == m->rows, 
                "Column data dimensions don't match");
    
    for (int i = 0; i < m->rows; i++) {
        m->data[i][col_index] = col_data->data[i][0];
    }
}

// Element-wise addition
Matrix* add(const Matrix *a, const Matrix *b) {
    MATRIX_CHECK(a != NULL && b != NULL, "Matrices cannot be NULL");
    MATRIX_CHECK(a->rows == b->rows && a->cols == b->cols, 
                "Matrix dimensions don't match for addition");
    
    Matrix *result = create_matrix(a->rows, a->cols);
    
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            result->data[i][j] = a->data[i][j] + b->data[i][j];
        }
    }
    
    return result;
}

// Element-wise subtraction
Matrix* subtract(const Matrix *a, const Matrix *b) {
    MATRIX_CHECK(a != NULL && b != NULL, "Matrices cannot be NULL");
    MATRIX_CHECK(a->rows == b->rows && a->cols == b->cols, 
                "Matrix dimensions don't match for subtraction");
    
    Matrix *result = create_matrix(a->rows, a->cols);
    
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            result->data[i][j] = a->data[i][j] - b->data[i][j];
        }
    }
    
    return result;
}

// Element-wise multiplication (Hadamard product)
Matrix* multiply(const Matrix *a, const Matrix *b) {
    MATRIX_CHECK(a != NULL && b != NULL, "Matrices cannot be NULL");
    MATRIX_CHECK(a->rows == b->rows && a->cols == b->cols, 
                "Matrix dimensions don't match for element-wise multiplication");
    
    Matrix *result = create_matrix(a->rows, a->cols);
    
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            result->data[i][j] = a->data[i][j] * b->data[i][j];
        }
    }
    
    return result;
}

Matrix* dot(const Matrix *a, const Matrix *b) {
    if (!a || !b) {
        printf("ERROR: One or both matrices are NULL in dot product\n");
        return NULL;
    }
    
    if (a->cols != b->rows) {
        printf("ERROR: Matrix dimension mismatch in dot product: ");
        printf("A(%d,%d) * B(%d,%d) - A cols (%d) != B rows (%d)\n", 
               a->rows, a->cols, b->rows, b->cols, a->cols, b->rows);
        return NULL;
    }
    
    Matrix *result = create_matrix(a->rows, b->cols);
    if (!result) {
        printf("ERROR: Failed to create result matrix in dot product\n");
        return NULL;
    }
    
    // Initialize to zero
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            result->data[i][j] = 0.0;
        }
    }
    
    // Perform matrix multiplication
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < a->cols; k++) {
                sum += a->data[i][k] * b->data[k][j];
            }
            result->data[i][j] = sum;
        }
    }
    
    return result;
}

// Scalar multiplication
Matrix* scale(const Matrix *a, double scalar) {
    MATRIX_CHECK(a != NULL, "Matrix cannot be NULL");
    
    Matrix *result = create_matrix(a->rows, a->cols);
    
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            result->data[i][j] = a->data[i][j] * scalar;
        }
    }
    
    return result;
}

// Matrix transpose
Matrix* transpose(const Matrix *a) {
    MATRIX_CHECK(a != NULL, "Matrix cannot be NULL");
    
    Matrix *result = create_matrix(a->cols, a->rows);
    
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            result->data[j][i] = a->data[i][j];
        }
    }
    
    return result;
}

// Apply function to each element
Matrix* apply_function(const Matrix *a, double (*func)(double)) {
    MATRIX_CHECK(a != NULL, "Matrix cannot be NULL");
    MATRIX_CHECK(func != NULL, "Function pointer cannot be NULL");
    
    Matrix *result = create_matrix(a->rows, a->cols);
    
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            result->data[i][j] = func(a->data[i][j]);
        }
    }
    
    return result;
}

// In-place operations (more efficient)
void add_inplace(Matrix *a, const Matrix *b) {
    MATRIX_CHECK(a != NULL && b != NULL, "Matrices cannot be NULL");
    MATRIX_CHECK(a->rows == b->rows && a->cols == b->cols, 
                "Matrix dimensions don't match for in-place addition");
    
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            a->data[i][j] += b->data[i][j];
        }
    }
}

void subtract_inplace(Matrix *a, const Matrix *b) {
    MATRIX_CHECK(a != NULL && b != NULL, "Matrices cannot be NULL");
    MATRIX_CHECK(a->rows == b->rows && a->cols == b->cols, 
                "Matrix dimensions don't match for in-place subtraction");
    
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            a->data[i][j] -= b->data[i][j];
        }
    }
}

void scale_inplace(Matrix *a, double scalar) {
    MATRIX_CHECK(a != NULL, "Matrix cannot be NULL");
    
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            a->data[i][j] *= scalar;
        }
    }
}

// Activation functions
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double relu(double x) {
    return x > 0 ? x : 0;
}

double tanh_func(double x) {
    return tanh(x);
}

// Helper function to check matrix for NaN values
int matrix_has_nan(const Matrix* m) {
    if (!m) return 0;
    
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            if (isnan(m->data[i][j])) {
                return 1;
            }
        }
    }
    return 0;
}


// Extract features from dataset (exclude label column)
Matrix* get_features(const Matrix* data_with_labels, int label_column) {
    MATRIX_CHECK(data_with_labels != NULL, "Data matrix cannot be NULL");
    MATRIX_CHECK(label_column >= 0 && label_column < data_with_labels->cols, 
                "Label column out of bounds");
    
    int num_samples = data_with_labels->rows;
    int num_features = data_with_labels->cols - 1;
    
    Matrix* features = create_matrix(num_samples, num_features);
    
    for (int i = 0; i < num_samples; i++) {
        int feature_idx = 0;
        for (int j = 0; j < data_with_labels->cols; j++) {
            if (j != label_column) {
                features->data[i][feature_idx] = data_with_labels->data[i][j];
                feature_idx++;
            }
        }
    }
    
    return features;
}

// Extract labels from dataset
Matrix* get_labels(const Matrix* data_with_labels, int label_column) {
    MATRIX_CHECK(data_with_labels != NULL, "Data matrix cannot be NULL");
    MATRIX_CHECK(label_column >= 0 && label_column < data_with_labels->cols, 
                "Label column out of bounds");
    
    int num_samples = data_with_labels->rows;
    Matrix* labels = create_matrix(num_samples, 1);
    
    for (int i = 0; i < num_samples; i++) {
        labels->data[i][0] = data_with_labels->data[i][label_column];
    }
    
    return labels;
}

// Print class distribution
void print_class_distribution(const Matrix* labels) {
    MATRIX_CHECK(labels != NULL, "Labels matrix cannot be NULL");
    MATRIX_CHECK(labels->cols == 1, "Labels must be a single column");
    
    int num_samples = labels->rows;
    
    // Count classes (assuming classes are 0,1,2,...)
    int max_class = 0;
    for (int i = 0; i < num_samples; i++) {
        int current_class = (int)labels->data[i][0];
        if (current_class > max_class) {
            max_class = current_class;
        }
    }
    
    int num_classes = max_class + 1;
    int* class_counts = (int*)calloc(num_classes, sizeof(int));
    
    // Count each class
    for (int i = 0; i < num_samples; i++) {
        int class_label = (int)labels->data[i][0];
        if (class_label >= 0 && class_label < num_classes) {
            class_counts[class_label]++;
        }
    }
    
    printf("Class Distribution:\n");
    printf("Class\tCount\tPercentage\n");
    printf("----------------------------\n");
    
    for (int i = 0; i < num_classes; i++) {
        double percentage = (double)class_counts[i] / num_samples * 100.0;
        printf("%d\t%d\t%.1f%%\n", i, class_counts[i], percentage);
    }
    
    free(class_counts);
}
