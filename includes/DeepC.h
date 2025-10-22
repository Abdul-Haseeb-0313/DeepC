#ifndef DEEPC_H
#define DEEPC_H

/*
 * DeepC: A Deep Learning Library in Pure C
 * Version: 1.0.0
 * GitHub: https://github.com/yourusername/DeepC
 * 
 * Usage: #include <DeepC.h>
 * 
 * Features:
 * - Complete neural network implementation
 * - Matrix operations optimized for performance
 * - Model serialization in .dc format
 * - No external dependencies
 */

// Standard library includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>

// Configuration macros
#define DEEPC_VERSION "1.0.0"
#define DEEPC_MAX_LAYERS 50
#define DEEPC_MAX_MATRIX_DIM 10000

// Error handling macros
#define DEEPC_CHECK(condition, msg) do { \
    if (!(condition)) { \
        fprintf(stderr, "DeepC Error: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Include all module headers
#include "matrix.h"
#include "layers.h"
#include "models.h"
#include "losses.h"
#include "optimizers.h"
#include "data_processing.h"

// Utility functions
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Get library version information
 * @return Version string
 */
const char* deepc_version();

/**
 * @brief Initialize DeepC library
 * @note Call this before using any DeepC functions
 */
void deepc_init();

/**
 * @brief Cleanup DeepC library resources
 * @note Call this when done using DeepC
 */
void deepc_cleanup();

/**
 * @brief Set random seed for reproducible results
 * @param seed Random seed value
 */
void deepc_seed(unsigned int seed);

#ifdef __cplusplus
}
#endif

#endif // DEEPC_H