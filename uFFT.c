#include "uFFT.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifndef TWO_PI
#define TWO_PI        (2.0f*3.14159265358979323846f)
#endif


#define  FAST_MATH
#ifdef FAST_MATH

static float fastAbs(float f) {
    int i = ((*(int *) &f) & 0x7fffffff);
    return (*(float *) &i);
}

#endif

static float fastSin(float x) {

#ifdef FAST_MATH
    const float pi = 3.14159265358979323846f;
    const float B = 4 / pi;
    const float C = -4 / (pi * pi);
    const float P = 0.224008178776f;
    const float Q = 0.775991821224f;
    float y = (B + C * fastAbs(x)) * x;
    return (Q + P * fastAbs(y)) * y;
#else
    return sinf(x);
#endif
}


static float fastCos(float x) {
#ifdef FAST_MATH
    const float pi = 3.14159265358979323846f;
    const float P = 0.224008178776f;
    const float Q = 0.775991821224f;
    const float pi_2 = pi * 0.5f;
    const float twopi = 2 * pi;
    const float B = 4 / pi;
    const float C = -4 / (pi * pi);
    x += pi_2;
    if (x > pi) {
        x -= twopi;
    }
    float y = (B + C * fastAbs(x)) * x;
    return (Q + P * fastAbs(y)) * y;
#else
    return cosf(x);
#endif
}

static int ctz(size_t N) {
    int ctz1 = 0;
    while (N) {
        ctz1++;
        N >>= 1;
    }
    return ctz1 - 1;
}

static size_t revbits(size_t v, int J) {
    size_t r = 0;
    for (int j = 0; j < J; j++) {
        r |= ((v >> j) & 1) << (J - 1 - j);
    }
    return r;
}

#ifndef USE_DIF

static void nop_split(const fft_complex *x, fft_complex *X, size_t N) {
    size_t halfOfN = N >> 1;
    const fft_complex *px = x;
    const fft_complex *halfOfx = x + halfOfN;
    fft_complex *pX = X;
    for (size_t n = 0; n < halfOfN; n++) {
        pX[0] = px[0];
        pX[1] = halfOfx[0];
        pX += 2;
        halfOfx++;
        px++;
    }
}

static void fft_split(const fft_complex *x, fft_complex *X, size_t N, float phi) {
    float t = (-TWO_PI) * phi;
    fft_complex cexp;
    cexp.real = fastCos(t);
    cexp.imag = fastSin(t);
    fft_complex val;
    size_t halfOfN = N >> 1;
    const fft_complex *px = x;
    const fft_complex *halfOfx = x + halfOfN;
    fft_complex *pX = X;
    for (size_t n = 0; n < halfOfN; n++) {
        pX[0].real = (px[0].real + halfOfx[0].real);
        pX[0].imag = (px[0].imag + halfOfx[0].imag);
        val.real = ((px[0].real - halfOfx[0].real));
        val.imag = ((px[0].imag - halfOfx[0].imag));
        pX[1].real = val.real * cexp.real - val.imag * cexp.imag;
        pX[1].imag = val.real * cexp.imag + val.imag * cexp.real;
        pX += 2;
        px++;
        halfOfx++;
    }
}

static void ifft_split(const fft_complex *x, fft_complex *X, size_t N, float phi) {
    float t = TWO_PI * phi;
    fft_complex cexp;
    cexp.real = fastCos(t);
    cexp.imag = fastSin(t);
    fft_complex val;
    size_t halfOfN = N >> 1;
    const fft_complex *px = x;
    const fft_complex *halfOfx = x + halfOfN;
    fft_complex *pX = X;
    for (size_t n = 0; n < N / 2; n++) {
        pX[0].real = (px[0].real + halfOfx[0].real) * 0.5f;
        pX[0].imag = (px[0].imag + halfOfx[0].imag) * 0.5f;
        val.real = ((0.5f * (px[0].real - halfOfx[0].real)));
        val.imag = ((0.5f * (px[0].imag - halfOfx[0].imag)));
        pX[1].real = val.real * cexp.real - val.imag * cexp.imag;
        pX[1].imag = val.real * cexp.imag + val.imag * cexp.real;
        pX += 2;
        px++;
        halfOfx++;
    }
}

static int nop_reverse(int b, fft_complex *buffers[2], size_t N) {
    int J = ctz(N);
    for (int j = J - 2; j >= 0; j--, b++) {
        size_t delta = N >> j;
        for (size_t n = 0; n < N; n += delta) {
            nop_split(buffers[b & 1] + n, buffers[~b & 1] + n, delta);
        }
    }
    return b;
}

static int fft_reverse(int b, fft_complex *buffers[2], size_t N) {
    int J = ctz(N);
    for (int j = J - 1; j >= 0; j--, b++) {
        size_t delta = N >> j;
        for (size_t n = 0; n < N; n += delta) {
            float phi = (float) revbits(n / delta, j) / (float) (2 << j);

            fft_split(buffers[b & 1] + n, buffers[~b & 1] + n, delta, phi);
        }
    }
    return b;
}

static int ifft_reverse(int b, fft_complex *buffers[2], size_t N) {
    int J = ctz(N);
    for (int j = J - 1; j >= 0; j--, b++) {
        size_t delta = N >> j;
        for (size_t n = 0; n < N; n += delta) {
            float phi = (float) revbits(n / delta, j) / (float) (2 << j);
            ifft_split(buffers[b & 1] + n, buffers[~b & 1] + n, delta, phi);
        }
    }
    return b;
}

#else

static void nop_split(const fft_complex *x, fft_complex *X, size_t N) {
    size_t halfOfN = N >> 1;
    const fft_complex *px = x;
    fft_complex *pX = X;
    fft_complex *halfOfX = X + halfOfN;
    for (size_t n = 0; n < halfOfN; n++) {
        pX[0] = px[0];
        halfOfX[0] = px[1];
        px += 2;
        pX++;
        halfOfX++;
    }
}

static void fft_split(const fft_complex *x, fft_complex *X, size_t N, float phi) {
    float t = (-TWO_PI) * phi;
    fft_complex cexp;
    cexp.real = fastCos(t);
    cexp.imag = fastSin(t);
    fft_complex val;
    size_t halfOfN = N >> 1;
    const fft_complex *px = x;
    fft_complex *pX = X;
    fft_complex *halfOfX = X + halfOfN;
    for (size_t n = 0; n < halfOfN; n++) {
        val.real = px[1].real * cexp.real - px[1].imag * cexp.imag;
        val.imag = px[1].real * cexp.imag + px[1].imag * cexp.real;
        pX[0].real = (px[0].real + val.real);
        pX[0].imag = (px[0].imag + val.imag);
        halfOfX[0].real = (px[0].real - val.real);
        halfOfX[0].imag = (px[0].imag - val.imag);
        px += 2;
        pX++;
        halfOfX++;
    }
}

static void ifft_split(const fft_complex *x, fft_complex *X, size_t N, float phi) {
    float t = TWO_PI * phi;
    fft_complex cexp;
    cexp.real = fastCos(t);
    cexp.imag = fastSin(t);
    fft_complex val;
    size_t halfOfN = N >> 1;
    const fft_complex *px = x;
    fft_complex *pX = X;
    fft_complex *halfOfX = X + halfOfN;
    for (size_t n = 0; n < halfOfN; n++) {
        val.real = px[1].real * cexp.real - px[1].imag * cexp.imag;
        val.imag = px[1].real * cexp.imag + px[1].imag * cexp.real;
        pX[0].real = (px[0].real + val.real) * 0.5f;
        pX[0].imag = (px[0].imag + val.imag) * 0.5f;
        halfOfX[0].real = (px[0].real - val.real) * 0.5f;
        halfOfX[0].imag = (px[0].imag - val.imag) * 0.5f;
        px += 2;
        pX++;
        halfOfX++;
    }
}

static int nop_reverse(int b, fft_complex *buffers[2], size_t N) {
    int J = ctz(N);
    for (int j = 0; j < J - 1; j++, b++) {
        size_t delta = N >> j;
        for (size_t n = 0; n < N; n += delta) {
            nop_split(buffers[b & 1] + n, buffers[~b & 1] + n, delta);
        }
    }
    return b;
}

static int fft_reverse(int b, fft_complex *buffers[2], size_t N) {
    int J = ctz(N);
    for (int j = 0; j < J; j++, b++) {
        size_t delta = N >> j;
        for (size_t n = 0; n < N; n += delta) {
            float phi = (float) revbits(n / delta, j) / (float) (2 << j);
            fft_split(buffers[b & 1] + n, buffers[~b & 1] + n, delta, phi);
        }
    }
    return b;
}

static int ifft_reverse(int b, fft_complex *buffers[2], size_t N) {
    int J = ctz(N);
    for (int j = 0; j < J; j++, b++) {
        size_t delta = N >> j;
        for (size_t n = 0; n < N; n += delta) {
            float phi = ((float) revbits(n / delta, j) / (float) (2 << j));
            ifft_split(buffers[b & 1] + n, buffers[~b & 1] + n, delta, phi);
        }
    }
    return b;
}

#endif

int fft(fft_complex *vector, size_t N) {
    if (!N) return 0;

    if (N & (N - 1)) return 1;

    fft_complex *buffers[2] = {vector, malloc(N * sizeof(fft_complex))};

    if (!buffers[1]) return -1;

    int b = 0;

    b = nop_reverse(b, buffers, N);
    b = fft_reverse(b, buffers, N);
    b = nop_reverse(b, buffers, N);

    memmove(vector, buffers[b & 1], N * sizeof(fft_complex));

    free(buffers[1]);

    return 0;
}

int ifft(fft_complex *vector, size_t N) {
    if (!N) return 0;

    if (N & (N - 1)) return 1;

    fft_complex *buffers[2] = {vector, malloc(N * sizeof(fft_complex))};

    if (!buffers[1]) return -1;

    int b = 0;

    b = nop_reverse(b, buffers, N);
    b = ifft_reverse(b, buffers, N);
    b = nop_reverse(b, buffers, N);

    memmove(vector, buffers[b & 1], N * sizeof(fft_complex));

    free(buffers[1]);

    return 0;
}
