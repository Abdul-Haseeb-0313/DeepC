# DeepC

A lightweight deep learning library implemented in pure C.

## Quick Install

```bash
git clone https://github.com/yourusername/DeepC.git
```
## Usage
```c
#include "deepc/DeepC.h"

int main() {
    SequentialModel* model = create_model("MyModel");
    add_layer(model, Dense(64, RELU, 10));
    add_layer(model, Dense(1, SIGMOID, 64));
    compile(model, ADAM, BINARY_CROSSENTROPY, 0.001);
    
    // Train with your data
    // fit(model, X, y, 100, 32, 1);
    
    free_model(model);
    return 0;
}
```
## Compile test with:
```bash
mkdir build 
cd build 
cmake ..
cmake --build .
cd ..
bin/test
```

## Features
- Neural networks with multiple layer types

- Matrix operations optimized for performance

- Model saving/loading in .dc format

- Data preprocessing and CSV loading

- No external dependencies
