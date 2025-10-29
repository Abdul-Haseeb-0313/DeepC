#include "deepc/DeepC.h"

int main() {
    SequentialModel* model = create_model("MyModel");
    add_layer(model, Dense(64, RELU, 10));
    add_layer(model, Dense(1, SIGMOID, 64));
    compile(model, ADAM, BINARY_CROSSENTROPY, 0.001);

    free_model(model);
}
