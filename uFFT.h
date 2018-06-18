#ifndef UFFT_H
#define UFFT_H

#include <math.h>
#include <stddef.h>

typedef struct {
    float real, imag;
} fft_complex;

//#define  USE_DIF

/**
 * @brief FFT algorithm (forward transform)
 *
 * This function computes forward radix-2 fast Fourier transform (FFT).
 * The output is written in-place over the input.
 *
 * @param vector An array of @p N complex values in single-precision floating-point format.
 * @param N The size of the transform must be a power of two.
 *
 * @return Zero for success.
 */
int fft(fft_complex *vector, size_t N);

/**
 * @brief FFT algorithm (inverse transform)
 *
 * This function computes inverse radix-2 fast Fourier transform (FFT).
 * The output is written in-place over the input.
 *
 * @param vector An array of @p N complex values in single-precision floating-point format.
 * @param N The size of the transform must be a power of two.
 *
 * @return Zero for success.
 */
int ifft(fft_complex *vector, size_t N);

#endif
