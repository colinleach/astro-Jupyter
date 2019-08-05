#include <stdio.h>
#include <math.h>
//#include <stdbool.h>
#include "isprime.h"

bool is_prime(long n) {
    if (n % 2 == 0) return false;

    long sqrt_n = (long) floor(sqrt(n));
    for (int i=3; i < sqrt_n+1; i += 2) {
        if (n % i == 0) return false;
    }
    return true;
}