#include <cstdlib>
#include "immintrin.h"
#include <chrono>
#include <cstdio>
#include <iostream>
#include <random>
#include <unistd.h>
#include <omp.h>
//#define N 33'554'432 // 2^25
//#define N 33554432 // 2^25
#define N 8388608 // 2^25


using namespace std;

void swapVals(int *xp, int *yp) {
	int temp = *xp;
	*xp = *yp;
	*yp = temp;
}

void bitonicSort( __m512i &A1i, __m512i &A2i, __m512i &B1i, __m512i &B2i, __m512i &C1i, __m512i &C2i, 
			 __m512i &D1i, __m512i &D2i, __m512i &A1o, __m512i &A2o, __m512i &B1o, __m512i &B2o, __m512i &C1o, 
			__m512i &C2o, __m512i &D1o, __m512i &D2o) {
	// A1i and A2i are my 'a' and 'b'
	// A1o and a2o are going to be my L and H that I want to write to memory
	// L1 to L2
	A2i = _mm512_permutexvar_epi32(_mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), A2i);

	A1o = _mm512_min_epi32(A1i, A2i);
	A2o = _mm512_max_epi32(A1i, A2i);

	A1o = _mm512_permutex2var_epi32(A1o, _mm512_set_epi32(23, 22, 21, 20, 19, 18, 17, 16, 7, 6, 5, 4, 3, 2, 1, 0), A2o);
	A2o = _mm512_permutex2var_epi32(A1o, _mm512_set_epi32(31, 30, 29, 28, 27, 26, 25, 24, 15, 14, 13, 12, 11, 10, 9, 8), A2o);


	// Storing them as the inputs now so we can do the second round of sorting. L2 to L3
	A1i = _mm512_min_epi32(A1o, A2o);
	A2i = _mm512_max_epi32(A1o, A2o);

	A2i = _mm512_permutexvar_epi32(_mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), A2i);

	A1o = _mm512_min_epi32(A1i, A2i);
	A2o = _mm512_max_epi32(A1i, A2i);

	A1o = _mm512_permutex2var_epi32(A1o, _mm512_set_epi32(27, 26, 25, 24, 11, 10, 9, 8, 19, 18, 17, 16, 3, 2, 1, 0), A2o);
	A2o = _mm512_permutex2var_epi32(A1o, _mm512_set_epi32(31, 30, 29, 28, 15, 14, 13, 12, 23, 22, 21, 20, 7, 6, 5, 4), A2o);


	// L3 to L4
	A1i = _mm512_min_epi32(A1o, A2o);
	A2i = _mm512_max_epi32(A1o, A2o);

	A2i = _mm512_permutexvar_epi32(_mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), A2i);

	A1o = _mm512_min_epi32(A1i, A2i);
	A2o = _mm512_max_epi32(A1i, A2i);

	A1o = _mm512_permutex2var_epi32(A1o, _mm512_set_epi32(29, 28, 13, 12, 25, 24, 9, 8, 21, 20, 5, 4, 17, 16, 1, 0), A2o);
	A2o = _mm512_permutex2var_epi32(A1o, _mm512_set_epi32(31, 30, 15, 14, 27, 26, 11, 10, 23, 22, 7, 6, 19, 18, 3, 2), A2o);


	// L4 to L5
	A1i = _mm512_min_epi32(A1o, A2o);
	A2i = _mm512_max_epi32(A1o, A2o);

	A2i = _mm512_permutexvar_epi32(_mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), A2i);

	A1o = _mm512_min_epi32(A1i, A2i);
	A2o = _mm512_max_epi32(A1i, A2i);

	A1o = _mm512_permutex2var_epi32(A1o, _mm512_set_epi32(30, 14,28, 12, 26, 10, 24, 8, 22, 6, 20, 4, 18, 2, 16, 0), A2o);
	A2o = _mm512_permutex2var_epi32(A1o, _mm512_set_epi32(31, 15, 29, 13, 27, 11, 25, 9, 23, 7, 21, 5, 19, 3, 17, 1), A2o);


	// L5 final sort
	A1i = _mm512_min_epi32(A1o, A2o);
	A2i = _mm512_max_epi32(A1o, A2o);

	A2i = _mm512_permutexvar_epi32(_mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), A2i);

	A1o = _mm512_min_epi32(A1i, A2i);
	A2o = _mm512_max_epi32(A1i, A2i);

	A1o = _mm512_permutex2var_epi32(A1o, _mm512_set_epi32(23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0), A2o);
	A2o = _mm512_permutex2var_epi32(A1o, _mm512_set_epi32(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8), A2o);

	//===========================================================================================================================
		// L1 to L2
	B2i = _mm512_permutexvar_epi32(_mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), B2i);

	B1o = _mm512_min_epi32(B1i, B2i);
	B2o = _mm512_max_epi32(B1i, B2i);

	B1o = _mm512_permutex2var_epi32(B1o, _mm512_set_epi32(23, 22, 21, 20, 19, 18, 17, 16, 7, 6, 5, 4, 3, 2, 1, 0), B2o);
	B2o = _mm512_permutex2var_epi32(B1o, _mm512_set_epi32(31, 30, 29, 28, 27, 26, 25, 24, 15, 14, 13, 12, 11, 10, 9, 8), B2o);


	// Storing them as the inputs now so we can do the second round of sorting. L2 to L3
	B1i = _mm512_min_epi32(B1o, B2o);
	B2i = _mm512_max_epi32(B1o, B2o);

	B2i = _mm512_permutexvar_epi32(_mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), B2i);

	B1o = _mm512_min_epi32(B1i, B2i);
	B2o = _mm512_max_epi32(B1i, B2i);

	B1o = _mm512_permutex2var_epi32(B1o, _mm512_set_epi32(27, 26, 25, 24, 11, 10, 9, 8, 19, 18, 17, 16, 3, 2, 1, 0), B2o);
	B2o = _mm512_permutex2var_epi32(B1o, _mm512_set_epi32(31, 30, 29, 28, 15, 14, 13, 12, 23, 22, 21, 20, 7, 6, 5, 4), B2o);


	// L3 to L4
	B1i = _mm512_min_epi32(B1o, B2o);
	B2i = _mm512_max_epi32(B1o, B2o);

	B2i = _mm512_permutexvar_epi32(_mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), B2i);

	B1o = _mm512_min_epi32(B1i, B2i);
	B2o = _mm512_max_epi32(B1i, B2i);

	B1o = _mm512_permutex2var_epi32(B1o, _mm512_set_epi32(29, 28, 13, 12, 25, 24, 9, 8, 21, 20, 5, 4, 17, 16, 1, 0), B2o);
	B2o = _mm512_permutex2var_epi32(B1o, _mm512_set_epi32(31, 30, 15, 14, 27, 26, 11, 10, 23, 22, 7, 6, 19, 18, 3, 2), B2o);


	// L4 to L5
	B1i = _mm512_min_epi32(B1o, B2o);
	B2i = _mm512_max_epi32(B1o, B2o);

	B2i = _mm512_permutexvar_epi32(_mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), B2i);

	B1o = _mm512_min_epi32(B1i, B2i);
	B2o = _mm512_max_epi32(B1i, B2i);

	B1o = _mm512_permutex2var_epi32(B1o, _mm512_set_epi32(30, 14, 28, 12, 26, 10, 24, 8, 22, 6, 20, 4, 18, 2, 16, 0), B2o);
	B2o = _mm512_permutex2var_epi32(B1o, _mm512_set_epi32(31, 15, 29, 13, 27, 11, 25, 9, 23, 7, 21, 5, 19, 3, 17, 1), B2o);


	// L5 final sort
	B1i = _mm512_min_epi32(B1o, B2o);
	B2i = _mm512_max_epi32(B1o, B2o);

	B2i = _mm512_permutexvar_epi32(_mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), B2i);

	B1o = _mm512_min_epi32(B1i, B2i);
	B2o = _mm512_max_epi32(B1i, B2i);

	B1o = _mm512_permutex2var_epi32(B1o, _mm512_set_epi32(23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0), B2o);
	B2o = _mm512_permutex2var_epi32(B1o, _mm512_set_epi32(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8), B2o);

	//============================================================================================================================
		// L1 to L2
	C2i = _mm512_permutexvar_epi32(_mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), C2i);

	C1o = _mm512_min_epi32(C1i, C2i);
	C2o = _mm512_max_epi32(C1i, C2i);

	C1o = _mm512_permutex2var_epi32(C1o, _mm512_set_epi32(23, 22, 21, 20, 19, 18, 17, 16, 7, 6, 5, 4, 3, 2, 1, 0), C2o);
	C2o = _mm512_permutex2var_epi32(C1o, _mm512_set_epi32(31, 30, 29, 28, 27, 26, 25, 24, 15, 14, 13, 12, 11, 10, 9, 8), C2o);


	// Storing them as the inputs now so we can do the second round of sorting. L2 to L3
	C1i = _mm512_min_epi32(C1o, C2o);
	C2i = _mm512_max_epi32(C1o, C2o);

	C2i = _mm512_permutexvar_epi32(_mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), C2i);

	C1o = _mm512_min_epi32(C1i, C2i);
	C2o = _mm512_max_epi32(C1i, C2i);

	C1o = _mm512_permutex2var_epi32(C1o, _mm512_set_epi32(27, 26, 25, 24, 11, 10, 9, 8, 19, 18, 17, 16, 3, 2, 1, 0), C2o);
	C2o = _mm512_permutex2var_epi32(C1o, _mm512_set_epi32(31, 30, 29, 28, 15, 14, 13, 12, 23, 22, 21, 20, 7, 6, 5, 4), C2o);


	// L3 to L4
	C1i = _mm512_min_epi32(C1o, C2o);
	C2i = _mm512_max_epi32(C1o, C2o);

	C2i = _mm512_permutexvar_epi32(_mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), C2i);

	C1o = _mm512_min_epi32(C1i, C2i);
	C2o = _mm512_max_epi32(C1i, C2i);

	C1o = _mm512_permutex2var_epi32(C1o, _mm512_set_epi32(29, 28, 13, 12, 25, 24, 9, 8, 21, 20, 5, 4, 17, 16, 1, 0), C2o);
	C2o = _mm512_permutex2var_epi32(C1o, _mm512_set_epi32(31, 30, 15, 14, 27, 26, 11, 10, 23, 22, 7, 6, 19, 18, 3, 2), C2o);


	// L4 to L5
	C1i = _mm512_min_epi32(C1o, C2o);
	C2i = _mm512_max_epi32(C1o, C2o);

	C2i = _mm512_permutexvar_epi32(_mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), C2i);

	C1o = _mm512_min_epi32(C1i, C2i);
	C2o = _mm512_max_epi32(C1i, C2i);

	C1o = _mm512_permutex2var_epi32(A1o, _mm512_set_epi32(30, 14, 28, 12, 26, 10, 24, 8, 22, 6, 20, 4, 18, 2, 16, 0), C2o);
	C2o = _mm512_permutex2var_epi32(A1o, _mm512_set_epi32(31, 15, 29, 13, 27, 11, 25, 9, 23, 7, 21, 5, 19, 3, 17, 1), C2o);


	// L5 final sort
	C1i = _mm512_min_epi32(C1o, C2o);
	C2i = _mm512_max_epi32(C1o, C2o);

	C2i = _mm512_permutexvar_epi32(_mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), C2i);

	C1o = _mm512_min_epi32(C1i, C2i);
	C2o = _mm512_max_epi32(C1i, C2i);

	C1o = _mm512_permutex2var_epi32(C1o, _mm512_set_epi32(23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0), C2o);
	C2o = _mm512_permutex2var_epi32(C1o, _mm512_set_epi32(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8), C2o);

	//==========================================================================================================================
		// L1 to L2
	D2i = _mm512_permutexvar_epi32(_mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), D2i);

	D1o = _mm512_min_epi32(D1i, D2i);
	D2o = _mm512_max_epi32(D1i, D2i);

	D1o = _mm512_permutex2var_epi32(D1o, _mm512_set_epi32(23, 22, 21, 20, 19, 18, 17, 16, 7, 6, 5, 4, 3, 2, 1, 0), D2o);
	D2o = _mm512_permutex2var_epi32(D1o, _mm512_set_epi32(31, 30, 29, 28, 27, 26, 25, 24, 15, 14, 13, 12, 11, 10, 9, 8), D2o);


	// Storing them as the inputs now so we can do the second round of sorting. L2 to L3
	D1i = _mm512_min_epi32(D1o, D2o);
	D2i = _mm512_max_epi32(D1o, D2o);

	D2i = _mm512_permutexvar_epi32(_mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), D2i);

	D1o = _mm512_min_epi32(D1i, D2i);
	D2o = _mm512_max_epi32(D1i, D2i);

	D1o = _mm512_permutex2var_epi32(D1o, _mm512_set_epi32(27, 26, 25, 24, 11, 10, 9, 8, 19, 18, 17, 16, 3, 2, 1, 0), D2o);
	D2o = _mm512_permutex2var_epi32(D1o, _mm512_set_epi32(31, 30, 29, 28, 15, 14, 13, 12, 23, 22, 21, 20, 7, 6, 5, 4), D2o);


	// L3 to L4
	D1i = _mm512_min_epi32(D1o, D2o);
	D2i = _mm512_max_epi32(D1o, D2o);

	D2i = _mm512_permutexvar_epi32(_mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), D2i);

	D1o = _mm512_min_epi32(D1i, D2i);
	D2o = _mm512_max_epi32(D1i, D2i);

	D1o = _mm512_permutex2var_epi32(D1o, _mm512_set_epi32(29, 28, 13, 12, 25, 24, 9, 8, 21, 20, 5, 4, 17, 16, 1, 0), D2o);
	D2o = _mm512_permutex2var_epi32(D1o, _mm512_set_epi32(31, 30, 15, 14, 27, 26, 11, 10, 23, 22, 7, 6, 19, 18, 3, 2), D2o);


	// L4 to L5
	D1i = _mm512_min_epi32(D1o, D2o);
	D2i = _mm512_max_epi32(D1o, D2o);

	D2i = _mm512_permutexvar_epi32(_mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), D2i);

	D1o = _mm512_min_epi32(D1i, D2i);
	D2o = _mm512_max_epi32(D1i, D2i);

	D1o = _mm512_permutex2var_epi32(D1o, _mm512_set_epi32(30, 14, 28, 12, 26, 10, 24, 8, 22, 6, 20, 4, 18, 2, 16, 0), D2o);
	D2o = _mm512_permutex2var_epi32(D1o, _mm512_set_epi32(31, 15, 29, 13, 27, 11, 25, 9, 23, 7, 21, 5, 19, 3, 17, 1), D2o);


	// L5 final sort
	D1i = _mm512_min_epi32(D1o, D2o);
	D2i = _mm512_max_epi32(D1o, D2o);

	D2i = _mm512_permutexvar_epi32(_mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), D2i);

	D1o = _mm512_min_epi32(D1i, D2i);
	D2o = _mm512_max_epi32(D1i, D2i);

	D1o = _mm512_permutex2var_epi32(D1o, _mm512_set_epi32(23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0), D2o);
	D2o = _mm512_permutex2var_epi32(D1o, _mm512_set_epi32(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8), D2o);


	//_mm512_store_si512(&output[indexVal], A2o); // cannot call output without passing it into the function.


	// The algorithm has run, the values have been shuffled within the function, now I just need to store them in
	// the output array.
	return; // Will be stored back into the array later
}



void selectionSort(int arr[], int startIndex, int endIndex) {
	int i;
	int j;
	int minIndex;

	for (i = startIndex; i < endIndex; i++) {
		minIndex = i;
		for (j = i + 1; j < endIndex + 1; j++) {
			if (arr[j] < arr[minIndex]) {
				minIndex = j;
			}
		}
		swapVals(&arr[minIndex], &arr[i]);
	}
}

void printArray(int arr[], int size) {
	int i;
	for (i = 0; i < size; i++) {
		cout << arr[i] << " ";
	}
	cout << endl;
}


int main() {
	//int endIndex = 65536;
	//int endingSortedBlockSize = 16384;
	//int arrSize = 65536;
	int arrSize = 8388608;
	int j; // random number to fill array

	cout << "before making the arrays" << endl;
	int *a = (int*)aligned_alloc(64, sizeof(int) * arrSize);
	int *output = (int*)aligned_alloc(64, sizeof(int) * arrSize);
	//int *mainArray = (int*)aligned_alloc(64, sizeof(int) * arrSize);
	//int *outputMain = (int*)aligned_alloc(64, sizeof(int) * arrSize);
	//int* mainArray = new int[arrSize];
	//int* outputMain = new int[arrSize];
	cout << "broke regular arrays" << endl;

	std::random_device rd; // obtains a seed for the random number engine
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> distrib(0, 2147483646);
	//cout << "before array pointers are created" << endl;

	//int startIndex = i;
	//int endIndex = i + 65536;
	//int sortedBlockSize = 16;
	//int endingSortedBlockSize = 16384;

	for (int i = 0; i < arrSize; i++) {
		j = distrib(gen);
		a[i] = j; // the array is filled with 65536 random values from 0 to 32 bits
	}
	//cout << "array successfully filled with random values" << endl;


	#pragma omp parallel for schedule(dynamic) num_threads(64)
	for (unsigned int i = 0; i < N; i += 65536) {
		sleep(1);
		printf("Thread %d is ready to work within range [%d,%d).\n", omp_get_thread_num(), i, (i + 65536));
	
	
	//int forLoopSize = 1;
	//cout << "before thread loop" << endl;

	//for (int i = 0; i < forLoopSize; i++) {
		//int indexTracker;
		//int startIndex = 0; // needs to be adjusted for each thread.
		//int endIndex = 65536; // needs to be adjusted for each thread
		//cout << "before startIndex" << endl;
		int startIndex = i;
		int endIndex = i + 65536;
		int sortedBlockSize = 16;
		int endingSortedBlockSize = 16384;
		//int arrSize = 33554432;
		//cout << "startIndex" << endl;
		//cout << startIndex << endl;
		//cout << "endIndex" << endl;
		//cout << endIndex << endl;

		//cout << "about to create registers" << endl;
		__m512i A1in; 
		__m512i A2in;
		__m512i A1out; // Become inputs of next call
		__m512i A2out; // Become inputs of next call

		__m512i B1in; 
		__m512i B2in; 
		__m512i B1out;
		__m512i B2out;

		__m512i C1in;
		__m512i C2in;
		__m512i C1out;
		__m512i C2out;

		__m512i D1in;
		__m512i D2in; 
		__m512i D1out;
		__m512i D2out;
		//cout << "registers created" << endl;

		//-------------------------
		//int *tempIn;// = a;
		//int *tempOut;// = output;
		//tempIn = a;
		//tempOut = output;
		//tempIn += startIndex;
		//tempOut += endIndex;
		
		//tempIn = *a[i];// = new int[65536];
		//tempOut = *output[i + 65536];// = new int[i + 65536];
		int *temp = new int[65536];
		//cout << "innerArr declared" << endl;
		//int k = 0;
		//for (int i = startIndex; i < endIndex; i++) {
		//	//int j = 0;
		//	innerArr[k] = a[i];
		//	k++;
		//}
		////*tempIn = innerArr[0];
		//cout << "innerArr created" << endl;
		//for (i = 0; i < 65536; i++) {
			//cout << "innerArr " << innerArr[65536] << endl;
		//}


		//for (int i = 0; i < 65536; i++) {
		//	innerArr[i] = *tempIn;
		//	tempIn++;
		//	//cout << tempIn << endl;
		//	if (innerArr[i] % 1000 == 0) {
		//		cout << innerArr[i] << endl;

		//	}
		//}
		//cout << "innerArr: " << innerArr[65535] << endl;
		//int *innerArr = mainArray;
		//a = innerArr;
		//-----------------------

		//int *a = (int*)aligned_alloc(64, sizeof(int) * arrSize);
		//int *output = (int*)aligned_alloc(64, sizeof(int) * arrSize);
		//a = mainArray;
		//output = outputMain;


			while (sortedBlockSize < endingSortedBlockSize) {
				//sortedBlockSize = sortedBlockSize * 2;

				for (int arrIndex = startIndex; arrIndex < endIndex; arrIndex += sortedBlockSize * 8) {
					if (sortedBlockSize > 16) {
						int temp = *a;
						*a = *output;
						*output = temp;
					}

					// We want to run this on each chunk of 128 ints, then move on. With ILP, we sort 128 ints a pop.

					// we just want to pull from the master array like normal, then do the selection sort.
					// need to computer the 

					//cout << "before startA1" << endl;
					int startA1 = arrIndex;
					int startA2 = arrIndex + sortedBlockSize;
					int startB1 = arrIndex + (sortedBlockSize * 2);
					int startB2 = arrIndex + (sortedBlockSize * 3);
					int startC1 = arrIndex + (sortedBlockSize * 4);
					int startC2 = arrIndex + (sortedBlockSize * 5);
					int startD1 = arrIndex + (sortedBlockSize * 6);
					int startD2 = arrIndex + (sortedBlockSize * 7);

					int endA1 = arrIndex + (sortedBlockSize - 1);
					int endA2 = arrIndex + (sortedBlockSize * 2) - 1;
					int endB1 = arrIndex + (sortedBlockSize * 3) - 1;
					int endB2 = arrIndex + (sortedBlockSize * 4) - 1;
					int endC1 = arrIndex + (sortedBlockSize * 5) - 1;
					int endC2 = arrIndex + (sortedBlockSize * 6) - 1;
					int endD1 = arrIndex + (sortedBlockSize * 7) - 1;
					int endD2 = arrIndex + (sortedBlockSize * 8) - 1;
					//cout << "arrIndex " << arrIndex << endl;


					// Try this conditional...
					if (arrIndex < 128) {
						selectionSort(a, startA1, endA1);
						selectionSort(a, startA2, endA2);
						selectionSort(a, startB1, endB1);
						selectionSort(a, startB2, endB2);
						selectionSort(a, startC1, endC1);
						selectionSort(a, startC2, endC2);
						selectionSort(a, startD1, endD1);
						selectionSort(a, startD2, endD2);
						//cout << "Selection Sort successful" << endl;
					}


					//cout << "writeA = startA1" << endl;
					//cout << startA1 << endl;
					//cout << startB1 << endl;
					//cout << startC1 << endl;
					//cout << startD1 << endl;

					int writeA = startA1; // tracks where to write back out to the array.
					int writeB = startB1;
					int writeC = startC1;
					int writeD = startD1;
					//cout << "finished setting writes equal to starts" << endl;

					//int regTest[16] = { 5,4,3,12,13,45,658,18,10,21,168,10,2,1,78,98 };

					// This loads the individual array parts into the XMM registers
					//cout << "Xmm registers" << endl;
					A1in = _mm512_load_si512(&a[startA1]); // first 16 integers <-- loading in vectors
					//cout << "A1in" << endl;
					A2in = _mm512_load_si512(&a[startA2]); // second 16 integers <-- loading in vectors
					//cout << "A2in" << endl;

					A1out; // Become inputs of next call
					A2out; // Become inputs of next call

					B1in = _mm512_load_si512(&a[startB1]);
					//cout << "B1in" << endl;
					B2in = _mm512_load_si512(&a[startB2]);
					//cout << "B2in" << endl;
					B1out;
					B2out;

					C1in = _mm512_load_si512(&a[startC1]);
					//cout << "C1in" << endl;
					C2in = _mm512_load_si512(&a[startC2]);
					//cout << "C2in" << endl;
					C1out;
					C2out;

					D1in = _mm512_load_si512(&a[startD1]);
					//cout << "D1in" << endl;
					D2in = _mm512_load_si512(&a[startD2]);
					//cout << "D2in" << endl;
					D1out;
					D2out;
					//^^^This is what I'm passing into the function
					//increment the start indexes
					//cout << "xmm registers loaded" << endl;

					
					startA1 += 16;
					startA2 += 16;
					startB1 += 16;
					startB2 += 16;
					startC1 += 16;
					startC2 += 16;
					startD1 += 16;
					startD2 += 16;
				

					//cout << "start indexes incremented" << endl;
					


					for (int j = 0; j < (sortedBlockSize / 8) - 1; j++) { 
						//cout << "before bitonicSort" << endl;
						bitonicSort(A1in, A2in, B1in, B2in, C1in, C2in, D1in, D2in, A1out, A2out, B1out, B2out, C1out, C2out, D1out, D2out); 
						//cout << "bitonicSort called" << endl;
						//cout << sortedBlockSize << endl;
						//cout << writeA << endl;
						//cout << writeB << endl;
						//cout << writeC << endl;
						//cout << writeD << endl;
						_mm512_store_si512(&output[writeA], A1out);
						//cout << "A1out" << endl;
						_mm512_store_si512(&output[writeB], B1out);
						//cout << "B1out" << endl;
						_mm512_store_si512(&output[writeC], C1out);
						//cout << "C1out" << endl;
						_mm512_store_si512(&output[writeD], D1out);
						//cout << "D1out" << endl;
						//cout << "sorted values stored to output array" << endl;
						// need to figure out how to write these sorted values back out to the output array
						// increment the 4 write indexes
						//if (write D >= arrSize)
						
						writeA += 16;
						writeB += 16;
						writeC += 16;
						writeD += 16;
						

						//cout << "write indexes incremented" << endl;
						//cout << "writeA: " << endl;
						//cout << writeA << endl;
						// Determine the other input.  For example, whatever value is smaller at indexes start
						// startA1 + 16 or startA2 + 16 dtermines the next input
						if (j == (sortedBlockSize / 8) - 2) {
							// This j loop is on its last iteration.
							//Write to the output array A2out, B2out, C2out, D2out
							//Increment the 4 write indexes by 16 each
							_mm512_store_si512(&output[writeA], A2out);
							_mm512_store_si512(&output[writeB], B2out);
							_mm512_store_si512(&output[writeC], C2out);
							_mm512_store_si512(&output[writeD], D2out);

							writeA += 16;
							writeB += 16;
							writeC += 16;
							writeD += 16;

							//cout << "if statement 'if' fulfilled" << endl;
						}
						else {
							//A1in = A2out;
							//B1in = B2out;
							//C1in = C2out;
							//D1in = D2out;
							//cout << "else portion initialized" << endl;

							if (startA1 == endA1) {
								// Load the A2in vector from input array index startA2
								//A1in = A2out;
								//cout << "breaks in A" << endl;
								A2in = _mm512_load_si512(&a[startA2]);
								startA2 += 16;
							}
							else if (startA2 == endA2) {
								// do something else
								A2in = _mm512_load_si512(&a[startA1]);
								startA1 += 16;
							}
							else if (startA1 < startA2) {
								A2in = _mm512_load_si512(&a[startA1]);
								startA1 += 16;
							}
							else {
								A2in = _mm512_load_si512(&a[startA2]);
								startA2 += 16;

							}
							//==============================================
							if (startB1 == endB1) {
								// Load the A2in vector from input array index startA2
								//A1in = A2out;
								//cout << "breaks in B" << endl;
								B2in = _mm512_load_si512(&a[startB2]);
								startB2 += 16;
							}
							else if (startB2 == endB2) {
								// do something else
								B2in = _mm512_load_si512(&a[startB1]);
								startB1 += 16;
							}
							else if (startB1 < startB2) {
								B2in = _mm512_load_si512(&a[startB1]);
								startB1 += 16;
							}
							else {
								B2in = _mm512_load_si512(&a[startB2]);
								startB2 += 16;

							}
							//=================================================
							if (startC1 == endC1) {
								// Load the A2in vector from input array index startA2
								//A1in = A2out;
								//cout << "breaks in C" << endl;
								C2in = _mm512_load_si512(&a[startC2]);
								startC2 += 16;
							}
							else if (startC2 == endC2) {
								// do something else
								C2in = _mm512_load_si512(&a[startC1]);
								startC1 += 16;
							}
							else if (startC1 < startC2) {
								C2in = _mm512_load_si512(&a[startC1]);
								startC1 += 16;
							}
							else {
								C2in = _mm512_load_si512(&a[startC2]);
								startC2 += 16;

							}
							//=====================================================
							if (startD1 == endD1) {
								// Load the A2in vector from input array index startA2
								//A1in = A2out;
								//cout << "breaks in D" << endl;
								D2in = _mm512_load_si512(&a[startD2]);
								startD2 += 16;
							}
							else if (startD2 == endD2) {
								// do something else
								D2in = _mm512_load_si512(&a[startD1]);
								startD1 += 16;
							}
							else if (startD1 < startD2) {
								D2in = _mm512_load_si512(&a[startD1]);
								startD1 += 16;
							}
							else {
								D2in = _mm512_load_si512(&a[startD2]);
								startD2 += 16;

							}

							//cout << "end of elses" << endl;
								
						}
						//cout << "end of conditionals" << endl;
						int temp = *a;
						*a = *output;
						*output = temp;

					 }
					 //int temp = *a;
					 //*a = *output;
					 //*output = temp;
					//cout << "bitonic loop finished" << endl;
				}
				//cout << "doubling sortedBlock Size" << endl;
				sortedBlockSize = sortedBlockSize * 2;
				//cout << "Sorted Block " << sortedBlockSize << endl;
				//cout << "Ending Block " << endingSortedBlockSize << endl;

				//cout << "sortedBlockSize successfully doubled" << endl;
				//cout << "sorted block size: " << endl;  
				//cout << sortedBlockSize << endl;
				// // exchange input and output pointers, so that your output becomes the new input and the input is now
				// // the space for your new output.

				//int temp = *a;
				//*a = *output;
				//*output = temp;

				//cout << "swapVal initiated" << endl;
				//swapVals(*a, *output);
				//cout << "swapVal completed" << endl;
			}
	/*		int temp = *a;
			*a = *output;
			*output = temp;*/

			//Deallocate the output array
			//cout << "attempting to deallocate output" << endl;
			//delete[] output;
			//cout << "output dallocated" << endl;



		//}

	} // End OpenMP for loop

	for (int i = 0; i < 8500; i++) {
		//if (i % 100 == 0) {
		cout << a[i] << endl;
	}

	return 0;
} // End main

//Compile with: g++ vector_avx_512_max.cc -o vector_avx_512_max.x -mavx512f