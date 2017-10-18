#pragma once

int read_file(const char*, gsl_matrix**, gsl_vector**);
static int readline(FILE*, char**);
int compute_mean_vector(gsl_matrix*, gsl_matrix**);
int compute_scatter_matrix(gsl_matrix*, gsl_matrix*, gsl_matrix**);
int compute_eigen(gsl_matrix*, gsl_vector**, gsl_matrix**);
int transform_dataset(gsl_matrix*, gsl_matrix*, int, gsl_matrix**);
int pca(gsl_matrix*, int, gsl_matrix**);
int print_matrix(FILE*, const gsl_matrix*);