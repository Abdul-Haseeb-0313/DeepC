#include "deepc/data_processing.h"

// Error handling macros
#define CSV_ERROR(msg) do { \
    fprintf(stderr, "\n*** CSV LOADER ERROR ***\n"); \
    fprintf(stderr, "Message: %s\n", msg); \
    fprintf(stderr, "File: %s\n", __FILE__); \
    fprintf(stderr, "Line: %d\n", __LINE__); \
    fprintf(stderr, "Function: %s\n", __func__); \
    exit(EXIT_FAILURE); \
} while(0)

#define CSV_CHECK(condition, msg) do { \
    if (!(condition)) { \
        CSV_ERROR(msg); \
    } \
} while(0)

// Helper function to count columns in a CSV line
int count_columns(const char *line) {

    CSV_CHECK(line != NULL, "Line cannot be NULL");
    
    int count = 0;
    int in_quotes = 0;

    const char *p = line;
    
    while (*p) {
        if (*p == '"') {
            in_quotes = !in_quotes;
        } else if (*p == ',' && !in_quotes) {
            count++;
        }
        p++;
    }
    
    return count + 1; // +1 because n commas means n+1 columns
}

// Helper function to trim whitespace
char* trim_whitespace(char *str) {
    if (!str) return NULL;
    
    char *end;
    
    // Trim leading space
    while (isspace((unsigned char)*str)) str++;
    
    if (*str == 0) return str;
    
    // Trim trailing space
    end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) end--;
    
    end[1] = '\0';
    return str;
}

// Simple CSV loader - reads entire CSV into a matrix
Matrix* load_csv(const char *filename, int has_header) {
    CSV_CHECK(filename != NULL, "Filename cannot be NULL");
    
    FILE *file = fopen(filename, "r");
    CSV_CHECK(file != NULL, "Cannot open CSV file");
    
    // Count lines and determine dimensions
    char line[4096];
    int line_count = 0;
    int num_cols = 0;
    int num_rows = 0;
    
    // First pass: count rows and columns
    while (fgets(line, sizeof(line), file)) {
        if (line_count == 0) {
            num_cols = count_columns(line);
        }
        line_count++;
    }
    
    CSV_CHECK(num_cols > 0, "CSV file appears to be empty or invalid");
    
    // Calculate dataset dimensions
    num_rows = has_header ? line_count - 1 : line_count;
    CSV_CHECK(num_rows > 0, "No data rows found");
    
    // Rewind file for second pass
    rewind(file);
    
    // Create matrix
    Matrix *matrix = create_matrix(num_rows, num_cols);
    
    // Skip header if present
    if (has_header) {
        CSV_CHECK(fgets(line, sizeof(line), file) != NULL, "Cannot read header line");
    }
    
    // Read data rows
    int row = 0;
    while (fgets(line, sizeof(line), file)) {
        if (strlen(trim_whitespace(line)) == 0) continue; // Skip empty lines
        
        char *token = strtok(line, ",\n");
        int col = 0;
        
        while (token != NULL && col < num_cols) {
            char *clean_token = trim_whitespace(token);
            double value;
            
            // Handle missing values (empty strings or "NA", "NULL", etc.)
            if (strlen(clean_token) == 0 || 
                strcasecmp(clean_token, "NA") == 0 ||
                strcasecmp(clean_token, "NULL") == 0 ||
                strcasecmp(clean_token, "N/A") == 0 ||
                strcasecmp(clean_token, "?") == 0) {
                value = NAN; // Use NaN to represent missing values
            } else {
                char *endptr;
                value = strtod(clean_token, &endptr);
                if (endptr == clean_token) {
                    // Conversion failed, treat as missing
                    value = NAN;
                }
            }
            
            matrix->data[row][col] = value;
            token = strtok(NULL, ",\n");
            col++;
        }
        
        // Fill remaining columns with NaN if line had fewer columns
        while (col < num_cols) {
            matrix->data[row][col] = NAN;
            col++;
        }
        
        row++;
    }
    
    fclose(file);
    return matrix;
}

// Count missing values in matrix (NaN values)
int count_missing_values(const Matrix *m) {
    CSV_CHECK(m != NULL, "Matrix cannot be NULL");
    
    int count = 0;
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            if (isnan(m->data[i][j])) {
                count++;
            }
        }
    }
    return count;
}

// Fill missing values with column mean
void fill_missing_with_mean(Matrix *m) {
    CSV_CHECK(m != NULL, "Matrix cannot be NULL");
    
    for (int j = 0; j < m->cols; j++) {
        // Calculate mean for column j (ignoring NaN)
        double sum = 0.0;
        int count = 0;
        
        for (int i = 0; i < m->rows; i++) {
            if (!isnan(m->data[i][j])) {
                sum += m->data[i][j];
                count++;
            }
        }
        
        if (count > 0) {
            double mean = sum / count;
            // Fill missing values with mean
            for (int i = 0; i < m->rows; i++) {
                if (isnan(m->data[i][j])) {
                    m->data[i][j] = mean;
                }
            }
        } else {
            // If all values are missing, fill with 0
            for (int i = 0; i < m->rows; i++) {
                if (isnan(m->data[i][j])) {
                    m->data[i][j] = 0.0;
                }
            }
        }
    }
}

// fills the missing values with zero
void fill_missing_with_zeros(Matrix *m) {
    CSV_CHECK(m != NULL, "Matrix cannot be NULL");
    
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            if (isnan(m->data[i][j])) {
                m->data[i][j] = 0.0;
            }
        }
    }
}

// Print matrix statistics
void print_matrix_stats(const Matrix *m) {
    CSV_CHECK(m != NULL, "Matrix cannot be NULL");
    
    printf("=== MATRIX STATISTICS ===\n");
    printf("Dimensions: %d x %d\n", m->rows, m->cols);
    printf("Missing values: %d\n", count_missing_values(m));
    
    for (int j = 0; j < m->cols; j++) {
        double sum = 0.0;
        double min_val = INFINITY;
        double max_val = -INFINITY;
        int count = 0;
        
        for (int i = 0; i < m->rows; i++) {
            if (!isnan(m->data[i][j])) {
                sum += m->data[i][j];
                if (m->data[i][j] < min_val) min_val = m->data[i][j];
                if (m->data[i][j] > max_val) max_val = m->data[i][j];
                count++;
            }
        }
        
        if (count > 0) {
            double mean = sum / count;
            printf("Col %d: mean=%.4f, min=%.4f, max=%.4f, valid=%d\n",
                   j, mean, min_val, max_val, count);
        } else {
            printf("Col %d: All values missing\n", j);
        }
    }
    printf("========================\n\n");
}

// Train-test split function
void train_test_split(const Matrix *X, const Matrix *y, double test_size, 
                     Matrix **X_train, Matrix **X_test, 
                     Matrix **y_train, Matrix **y_test) {
    CSV_CHECK(X != NULL, "Feature matrix X cannot be NULL");
    CSV_CHECK(y != NULL, "Target matrix y cannot be NULL");
    CSV_CHECK(X->rows == y->rows, "X and y must have same number of rows");
    CSV_CHECK(test_size > 0.0 && test_size < 1.0, "test_size must be between 0 and 1");
    
    int total_samples = X->rows;
    int test_samples = (int)(total_samples * test_size);
    int train_samples = total_samples - test_samples;
    
    CSV_CHECK(train_samples > 0, "Training set would be empty");
    CSV_CHECK(test_samples > 0, "Test set would be empty");
    
    // Create result matrices
    *X_train = create_matrix(train_samples, X->cols);
    *X_test = create_matrix(test_samples, X->cols);
    *y_train = create_matrix(train_samples, y->cols);
    *y_test = create_matrix(test_samples, y->cols);
    
    // Create shuffled indices
    int *indices = (int*)malloc(total_samples * sizeof(int));
    CSV_CHECK(indices != NULL, "Memory allocation failed for indices");
    
    for (int i = 0; i < total_samples; i++) {
        indices[i] = i;
    }
    
    // Shuffle indices (Fisher-Yates shuffle)
    srand(time(NULL));
    for (int i = total_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
    
    // Split data using shuffled indices
    for (int i = 0; i < train_samples; i++) {
        int orig_idx = indices[i];
        // Copy features
        for (int j = 0; j < X->cols; j++) {
            (*X_train)->data[i][j] = X->data[orig_idx][j];
        }
        // Copy targets
        for (int j = 0; j < y->cols; j++) {
            (*y_train)->data[i][j] = y->data[orig_idx][j];
        }
    }
    
    for (int i = 0; i < test_samples; i++) {
        int orig_idx = indices[train_samples + i];
        // Copy features
        for (int j = 0; j < X->cols; j++) {
            (*X_test)->data[i][j] = X->data[orig_idx][j];
        }
        // Copy targets
        for (int j = 0; j < y->cols; j++) {
            (*y_test)->data[i][j] = y->data[orig_idx][j];
        }
    }
    
    free(indices);
}

Matrix* split_features_labels(const Matrix *data, int label_column) {
    CSV_CHECK(data != NULL, "Data matrix cannot be NULL");
    CSV_CHECK(label_column >= 0 && label_column < data->cols, "Label column out of bounds");
    
    int num_samples = data->rows;
    int num_features = data->cols - 1;
    
    // Create result matrix: [X, y] combined
    Matrix* result = create_matrix(num_samples, num_features + 1);
    
    for (int i = 0; i < num_samples; i++) {
        int feature_idx = 0;
        for (int j = 0; j < data->cols; j++) {
            if (j == label_column) {
                // Put label in the last column
                result->data[i][num_features] = data->data[i][j];
            } else {
                // Put feature in current position
                result->data[i][feature_idx] = data->data[i][j];
                feature_idx++;
            }
        }
    }
    
    return result;
}

// One-hot encode labels
Matrix* one_hot_encode_labels(const Matrix *labels, int num_classes) {
    CSV_CHECK(labels != NULL, "Labels matrix cannot be NULL");
    CSV_CHECK(num_classes > 0, "Number of classes must be positive");
    CSV_CHECK(labels->cols == 1, "Labels must be a single column");
    
    int num_samples = labels->rows;
    Matrix* one_hot = create_matrix(num_samples, num_classes);
    
    for (int i = 0; i < num_samples; i++) {
        int label = (int)labels->data[i][0];
        
        // Validate label
        if (label < 0 || label >= num_classes) {
            printf("WARNING: Label %d out of range [0, %d] at sample %d\n", 
                   label, num_classes - 1, i);
            // Set to uniform distribution for invalid labels
            for (int j = 0; j < num_classes; j++) {
                one_hot->data[i][j] = 1.0 / num_classes;
            }
        } else {
            // Set one-hot encoding
            for (int j = 0; j < num_classes; j++) {
                one_hot->data[i][j] = (j == label) ? 1.0 : 0.0;
            }
        }
    }
    
    return one_hot;
}

// Convert one-hot encoded labels back to class indices
Matrix* one_hot_decode_labels(const Matrix *one_hot) {
    CSV_CHECK(one_hot != NULL, "One-hot matrix cannot be NULL");
    
    int num_samples = one_hot->rows;
    int num_classes = one_hot->cols;
    Matrix* labels = create_matrix(num_samples, 1);
    
    for (int i = 0; i < num_samples; i++) {
        int max_index = 0;
        double max_value = one_hot->data[i][0];
        
        for (int j = 1; j < num_classes; j++) {
            if (one_hot->data[i][j] > max_value) {
                max_value = one_hot->data[i][j];
                max_index = j;
            }
        }
        
        labels->data[i][0] = (double)max_index;
    }
    
    return labels;
}

// Normalize matrix to [0, 1] range
Matrix* normalize_matrix(Matrix *X) {
    CSV_CHECK(X != NULL, "Matrix cannot be NULL");
    
    Matrix* normalized = copy_matrix(X);
    
    for (int j = 0; j < normalized->cols; j++) {
        // Find min and max for column j
        double min_val = INFINITY;
        double max_val = -INFINITY;
        int valid_count = 0;
        
        for (int i = 0; i < normalized->rows; i++) {
            if (!isnan(normalized->data[i][j])) {
                if (normalized->data[i][j] < min_val) min_val = normalized->data[i][j];
                if (normalized->data[i][j] > max_val) max_val = normalized->data[i][j];
                valid_count++;
            }
        }
        
        if (valid_count > 0 && max_val > min_val) {
            // Normalize: (x - min) / (max - min)
            for (int i = 0; i < normalized->rows; i++) {
                if (!isnan(normalized->data[i][j])) {
                    normalized->data[i][j] = (normalized->data[i][j] - min_val) / (max_val - min_val);
                }
            }
        }
    }
    
    return normalized;
}

// Standardize matrix to mean=0, std=1
Matrix* standardize_matrix(Matrix *X) {
    CSV_CHECK(X != NULL, "Matrix cannot be NULL");
    
    Matrix* standardized = copy_matrix(X);
    
    for (int j = 0; j < standardized->cols; j++) {
        // Calculate mean for column j
        double sum = 0.0;
        int valid_count = 0;
        
        for (int i = 0; i < standardized->rows; i++) {
            if (!isnan(standardized->data[i][j])) {
                sum += standardized->data[i][j];
                valid_count++;
            }
        }
        
        if (valid_count > 0) {
            double mean = sum / valid_count;
            
            // Calculate standard deviation
            double variance = 0.0;
            for (int i = 0; i < standardized->rows; i++) {
                if (!isnan(standardized->data[i][j])) {
                    double diff = standardized->data[i][j] - mean;
                    variance += diff * diff;
                }
            }
            double std_dev = sqrt(variance / valid_count);
            
            if (std_dev > 1e-10) {  // Avoid division by zero
                // Standardize: (x - mean) / std_dev
                for (int i = 0; i < standardized->rows; i++) {
                    if (!isnan(standardized->data[i][j])) {
                        standardized->data[i][j] = (standardized->data[i][j] - mean) / std_dev;
                    }
                }
            }
        }
    }
    
    return standardized;
}

// Shuffle dataset (X and y together)
void shuffle_dataset(Matrix *X, Matrix *y) {
    CSV_CHECK(X != NULL, "X matrix cannot be NULL");
    CSV_CHECK(y != NULL, "y matrix cannot be NULL");
    CSV_CHECK(X->rows == y->rows, "X and y must have same number of samples");
    
    int num_samples = X->rows;
    
    // Create temporary storage for one sample
    double* temp_X = (double*)malloc(X->cols * sizeof(double));
    double temp_y = 0.0;
    
    // Fisher-Yates shuffle
    srand(time(NULL));
    for (int i = num_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        
        // Swap X samples
        memcpy(temp_X, X->data[i], X->cols * sizeof(double));
        memcpy(X->data[i], X->data[j], X->cols * sizeof(double));
        memcpy(X->data[j], temp_X, X->cols * sizeof(double));
        
        // Swap y samples
        temp_y = y->data[i][0];
        y->data[i][0] = y->data[j][0];
        y->data[j][0] = temp_y;
    }
    
    free(temp_X);
}
