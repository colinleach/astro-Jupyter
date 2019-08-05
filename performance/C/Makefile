CC = gcc
CPP = g++

# settings for debugging:
# DEBUG = -g
# CFLAGS = -Wall $(DEBUG) -I.
# LFLAGS = -Wall $(DEBUG) 

# settings for compiler optimization:
CFLAGS = -Wall -O3 -I.
LFLAGS = -Wall  

LIBS = -lm
DEPS = isprime.h
OBJ_CPP = isprime_cpp.o testprime_cpp.o
OBJ_C = isprime_c.o testprime_c.o

all: testprime shared testlib

isprime_cpp.o: isprime.c $(DEPS)
	$(CPP) -c -o $@ $< $(CFLAGS)

testprime_cpp.o: testprime.cpp $(DEPS)
	$(CPP) -c -o $@ $< $(CFLAGS)

testprime : $(OBJ_CPP) $(DEPS)
	$(CPP) -o $@ $^ $(CFLAGS) $(LIBS)

isprime_c.o: isprime.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

shared: isprime_c.o
	$(CC) -shared -fPIC isprime_c.o -o libisprime.so

# Getting the executable to find a .so file in the same directory was a pain
# Don't asume that the hack below is any sort of best practice!
testlib: testlib.c libisprime.so
	$(CC) -o $@ $< -Wl,-rpath,. -L`pwd` -lisprime $(CFLAGS) $(LIBS)

clean:
	rm -f *.o testprime testlib *.so
