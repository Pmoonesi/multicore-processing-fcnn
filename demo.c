#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include "genann.h"
#include <omp.h>

#ifndef _OPENMP
	printf("openMP is not activated!\n");
	return 0;
#endif // !_OPENMP

const char *save_name = "semeion.data";

void print_pic(double* pic) {
	int i, j;
	for (i = 0; i < 16; i++) {
		for (j = 0; j < 16; j++) {
			printf("%5.2lf", pic[i * 16 + j]);
		}
		printf("\n");
	}
}

void print_num(double* num) {
	int i;
	for (i = 0; i < 10; i++) {
		if (num[i] == 1) {
			printf("number is: %d\n", i);
			return;
		}
	}
	printf("wrong num array\n");
}

int get_num(double* num) {
	int i;
	for (i = 0; i < 10; i++) {
		if (num[i] == 1) return i;
	}
	return -1;
}

int main() {
	double start = omp_get_wtime();
	srand(time(0));

	// open the dataset file
	FILE *ptr = fopen(save_name, "a+");
	if (!ptr) {
		printf("Couldn't open file: %s\n", save_name);
		exit(1);
	}

	// find the size of dataset
	int samples = 0;
	char line[2048];
	while (!feof(ptr) && fgets(line, 2048, ptr)) {
		++samples;
	}
	fseek(ptr, 0, SEEK_SET);

	// allocate and populate pics and nums arrays of data and their tags
	double** pics = (double**)malloc(samples * sizeof(double*));
	double** nums = (double**)malloc(samples * sizeof(double*));
	int i, j;
	for (i = 0; i < samples; i++) {
		pics[i] = (double*)malloc(256 * sizeof(double));
		nums[i] = (double*)malloc(10 * sizeof(double));

		if (fgets(line, 2048, ptr) == NULL) {
			perror("fgets");
			exit(1);
		}

		char *split = strtok(line, " ");
		for (j = 0; j < 256; ++j) {
			pics[i][j] = atof(split);
			split = strtok(0, " ");
		}
		for (j = 0; j < 10; j++) {
			nums[i][j] = atof(split);
			split = strtok(0, " ");
		}
	}


	// close the dataset file
	fclose(ptr);
	
	// initialize the FCNN
	genann *ann = genann_init(256, 1, 28, 10);

	// train the FCNN with training data
	int epochs = 100;
	double learning_rate = 0.05;
	float train_percent = 0.7;
	int train_data = train_percent * samples;

	for (i = 0; i < epochs; i++) {
		for (j = 0; j < train_data; j++) {
			genann_train(ann, pics[j], nums[j], learning_rate);
		}
	}

	// test the FCNN with test data
	int corrects = 0;
	int ind = 0;
	for (i = train_data; i < samples; i++) {
		const double *guess = genann_run(ann, pics[i]);
		ind = 0;
		printf("answer: %d\t %6.2lf", get_num(nums[i]), guess[0]);
		for (j = 1; j < 10; j++) {
			printf("%6.2lf", guess[j]);
			if (guess[ind] < guess[j]) ind = j;
		}
		if (get_num(nums[i]) == ind) corrects++;
		printf("\tpicked: %d\n", ind);
	}

	// deallocate the data space
	for (i = 0; i < samples; i++) {
		free(pics[i]);
	}
	free(pics);

	// print accuracy
	printf("accuracy: %lf%%\n", (corrects * 100.0) / (samples - train_data));

	// measure the execution time and print it
	double finish = omp_get_wtime();
	printf("execution time: %lfs\n", finish - start);

	return 0;
}