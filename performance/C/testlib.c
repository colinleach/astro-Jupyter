#include <stdio.h>
#include "isprime.h"

int main() {
    int nPrimes = 6;
    long PRIMES[6] = {
    112272535095293,
    112582705942171,
    112272535095293,
    115280095190773,
    115797848077099,
    1099726899285419};

    for (int i=0; i<nPrimes; i++) {
        bool result = is_prime(PRIMES[i]);
        printf("%ld is prime: %s\n", PRIMES[i], result ? "true" : "false");
    }
}