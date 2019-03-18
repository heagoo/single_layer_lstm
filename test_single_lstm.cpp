#include "single_lstm_layer.h"
#include "timer.h"
#include <stdio.h>

void init_data(float *weights, int size, const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        printf("Cannot open %s\n", filename);
        exit(-1);
    }
    fread(weights, 4, size, fp);
    fclose(fp);
}

void init_ones(float *p, int size) {
    for (int i = 0; i < size; ++i) {
        p[i] = 1;
    }
}

int main() {
    int input_size = 1024;
    int output_size = 1024;
    SingleLSTMLayer layer(std::string(""), input_size, output_size, 1.0, 1);
    layer.Init();

    float *weights_data = new float[4 * (input_size + output_size) * output_size];
    float *bias_data = new float[4 * output_size];
    init_data(weights_data, 4 * (input_size + output_size) * output_size, "weights.bin");
    memset(bias_data, 0, 4 * output_size * sizeof(float));
    layer.SetKernel(weights_data, bias_data);

    float *input_data = new float[4 * input_size];
    //init_data(input_data, input_size, "input.bin");
    init_ones(input_data, 4 * input_size);
    layer.SetInput(input_data);

    // Warm up and init again
    layer.Forward();
    layer.Init();
    layer.SetInput(input_data);

    float *output = NULL;
    {
        int times = 1;
        printf("%d times ", times);
        Timer t("forward");
        for (int i = 0; i < times; ++i) {
            layer.Forward();
            output = layer.GetOutput();
        }
    }

    printf("First 4 outputs: ");
    for (int i = 0; i < 4; ++i) {
        printf("%f ", output[i]);
    }
    printf("\nLast  4 outputs: ");
    for (int i = 1020; i < 1024; ++i) {
        printf("%f ", output[i]);
    }
    printf("\n");
    return 0;
}
