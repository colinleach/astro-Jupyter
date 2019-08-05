#include <iostream>
#include <string>
#include "isprime.h"

using namespace std;

#include <chrono> 
using namespace std::chrono; 

//int main ( int argc, char* argv[] ) {
int main() {
    int nPrimes = 6;
    long PRIMES[6] = {
    112272535095293,
    112582705942171,
    112272535095293,
    115280095190773,
    115797848077099,
    1099726899285419};

    string bools[2] = {"True", "False"};

    for (int i=0; i<nPrimes; i++) {
        bool result = is_prime(PRIMES[i]);
        cout << PRIMES[i] << " is prime: " << bools[result] << endl;
    }

    int nLoops = 10;
    auto start = high_resolution_clock::now();
    for (int reps=0; reps<nLoops; reps++) {
        for (int i=0; i<nPrimes; i++) {
            is_prime(PRIMES[i]);
        }
    }
    auto stop = high_resolution_clock::now(); 
  
    auto duration = duration_cast<milliseconds>(stop - start); 
  
    cout << nLoops << " loops, average time taken: "
         << duration.count()/nLoops << " milliseconds" << endl; 

}