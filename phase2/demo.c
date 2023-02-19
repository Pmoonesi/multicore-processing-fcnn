#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "genann.h"
#include <omp.h>

const char *save_name = "semeion.data";

void print_arr(double* arr, int len) {
	int i;
	for (i = 0; i < len; i++) {
		printf("%6.2lf", *(arr + i));
	}
	printf("\n");
}

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

int main(int argc, char *argv[]) {
	int num_inputs = 256, num_hidden_layers = 1, num_hidden_layers_neurons = 28, num_outputs = 10, epochs = 100;
	char *save_name = "semeion.data", *token = " ";
	if (argc > 1 && argc <= 5) {
		printf("not enough arguments!\n");
		exit(-1);
	}
	if (argc > 5) {
		num_inputs = atoi(argv[1]);
		num_hidden_layers = atoi(argv[2]);
		num_hidden_layers_neurons = atoi(argv[3]);
		num_outputs = atoi(argv[4]);
		save_name = argv[5];
	}
	if (argc > 6) {
		epochs = atoi(argv[6]);
	}
	if (argc > 7) {
		token = argv[7];
	}
	printf("%d\t%d\t%d\t%d\t%d\t%s\t%s\n", num_inputs, num_hidden_layers, num_hidden_layers_neurons, num_outputs, epochs, save_name, token);

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
	char line[4096];
	while (!feof(ptr) && fgets(line, 4096, ptr)) {
		++samples;
	}
	fseek(ptr, 0, SEEK_SET);

	// allocate and populate pics and nums arrays of data and their tags
	double** pics = (double**)malloc(samples * sizeof(double*));
	double** nums = (double**)malloc(samples * sizeof(double*));
	int i, j;
	for (i = 0; i < samples; i++) {
		pics[i] = (double*)malloc(num_inputs * sizeof(double));
		nums[i] = (double*)malloc(num_outputs * sizeof(double));

		if (fgets(line, 4096, ptr) == NULL) {
			perror("fgets");
			exit(1);
		}

		char *split = strtok(line, token);
		for (j = 0; j < num_inputs; ++j) {
			pics[i][j] = atof(split);
			split = strtok(0, token);
		}
		for (j = 0; j < num_outputs; j++) {
			nums[i][j] = atof(split);
			split = strtok(0, token);
		}
	}

	// close the dataset file
	fclose(ptr);

	// initialize the FCNN
	genann *ann = genann_init(num_inputs, num_hidden_layers, num_hidden_layers_neurons, num_outputs);

	// train the FCNN with training data
	float train_percent = 0.7;
	int train_data = train_percent * samples;
	genann_hillclimb(ann, pics, nums, train_data);

	// test the FCNN with test data
	int corrects = 0;
	int ind = 0;
	for (i = train_data; i < samples; i++) {
		const double *guess = genann_run(ann, pics[i]);
		ind = 0;
		for (j = 1; j < num_outputs; j++) {
			if (guess[ind] < guess[j]) ind = j;
		}
		if (get_num(nums[i]) == ind) corrects++;
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