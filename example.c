#include <stdio.h>
#include "uFFT.h"

int main() {
    size_t N = 1 << 3;

    fft_complex vector[N];

    for (size_t n = 0; n < N; n++) {
        vector[n].real = n;
        vector[n].imag = 0;
    }
    printf("in time domain:\n");

    for (size_t n = 0; n < N; n++) {
        printf("%f %+f\n", vector[n].real, vector[n].imag);
    }

    fft(vector, N);

    printf("in frequency domain:\n");

    for (size_t n = 0; n < N; n++) {
        printf("%f %+f\n", vector[n].real, vector[n].imag);
    }

    ifft(vector, N);

    printf("in time domain:\n");

    for (size_t n = 0; n < N; n++) {
        printf("%f %+f\n", vector[n].real, vector[n].imag);
    }

    return 0;
}
