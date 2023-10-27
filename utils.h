#pragma once

#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define SWAP_LVALUE(type, temp, a, b) do {  \
    type temp = (a);                        \
    (a) = (b);                              \
    (b) = (temp);                           \
} while (0)

/**
 * Generate an uniformly distributed random float in [0, to]
 */
inline float frand(float to)
{
    return ((float)rand() / (float)(RAND_MAX)) * to;
}

/**
 * Calculate the mean of an arbitrary double array
 */
inline double fitness_mean(double *arr, int size)
{
    assert(size);

    double sum = 0;
    for (int i = 0; i < size; i++)
        sum += arr[i];
    return sum / (double)size;
}

/**
 * Calculate the standard deviation of an arbitrary double array.
 * Does not use Bessel's correction.
 */
inline double fitness_stdev(double *arr, int size)
{
    assert(size);

    double mean = fitness_mean(arr, size);
    double ssum = 0;
    for (int i = 0; i < size; i++)
    {
        double res = arr[i] - mean;
        ssum += res * res;
    }
    return sqrt(ssum / (double)size);
}

/**
 * Find the min of an arbitrary double array.
 */
inline double fitness_min(double *arr, int size)
{
    assert(size >= 1);

    double min = arr[0];
    for (int i = 1; i < size; i++)
    {
        if (arr[i] < min)
            min = arr[i];
    }
    return min;
}

/**
 * Find the max of an arbitrary double array.
 */
inline double fitness_max(double *arr, int size)
{
    assert(size >= 1);

    double max = arr[0];
    for (int i = 1; i < size; i++)
    {
        if(arr[i] > max)
            max = arr[i];
    }
    return max;
}


/*
 * Checks whether value is contained in array of given size.
 * Returns 1 if value is contained, 0 if not.
 */
inline int contains(int *array, int size, int value)
{
    if(!size) return 0;
    for (int i=0; i<size; i++)
        if (array[i] == value) return 1;
    return 0;
}
