#ifndef DEEPC_H
#define DEEPC_H

// Standard library includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>


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

#endif // DEEPC_H
