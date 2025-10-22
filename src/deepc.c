// deepc.c
#include "DeepC.h"

const char* deepc_version() {
    return DEEPC_VERSION;
}

void deepc_init() {
    // Initialize random seed
    srand(time(NULL));
    printf("DeepC %s initialized\n", DEEPC_VERSION);
}

void deepc_cleanup() {
    // Cleanup any global resources
    printf("DeepC cleanup completed\n");
}

void deepc_seed(unsigned int seed) {
    srand(seed);
}