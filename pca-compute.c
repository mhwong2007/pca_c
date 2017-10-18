#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include "pca.h"

static int max_line_len;

/*************************************************************
    read the line in a file
    parameters: (in)  FILE *input
                (out) char **line
    return    : SUCCESS     rc = 0
                END_OF_FILE rc = 1
**************************************************************/
static int readline(FILE *input, char **line)
{
    int rc = 0;
    int len;
    if (fgets(*line, max_line_len, input) == NULL) {
        rc = 1;
        goto EXIT;
    }

    while (strrchr(*line, '\n') == NULL) {
        /* reallocate line */
        max_line_len *= 2;
        char *new_line = (char*)malloc(max_line_len * sizeof(char));
        strcpy(new_line, *line);
        free(*line);
        *line = new_line;
        len = (int)strlen(*line);

        if (fgets(*line + len, max_line_len - len, input) == NULL) {
            rc = 0;
            goto EXIT;
        }
    }

EXIT:
    return rc;
}

/*************************************************************
    read the input file
    parameters: (in)  const char *filename
                (out) gsl_matrix **x,
                      gsl_vector **y
    return    : SUCCESS				rc = 0
                Can't open file		rc = -1
                Failed to read line	rc = -2
**************************************************************/
int read_file(const char *filename, gsl_matrix **x, gsl_vector **y)
{
    int inst_max_index, i, rc, num_features, num_samples, min_index, max_index, index;
    double value;
    size_t elements, j;
    FILE *fp = fopen(filename, "r");
    char *endptr;
    char *idx, *val, *label;

    if (fp == NULL) {
        fprintf(stderr, "can't open input file\n");
        rc = -1;
        goto EXIT;
    }

    num_samples = 0;
    num_features = 0;
    max_index = INT_MIN;
    min_index = INT_MAX;


    max_line_len = 1024;
    char *line = (char*)malloc(max_line_len * sizeof(char));

    /* first pass: determine number of samples and features */
    while ((rc = readline(fp, &line)) == 0) {
        /* label */
        label = strtok(line, " \t\n");
        if (label == NULL) {	/* empty line */
            rc = -2;
            goto EXIT;
        }

        /* features */
        while (1) {
            idx = strtok(NULL, ":");
            val = strtok(NULL, " \t");

            if (val == NULL) {
                break;
            }

            errno = 0;
            index = (int)strtol(idx, &endptr, 10);
            /* if idx is not convertible */
            if (endptr == idx || errno != 0 || *endptr != '\0') {
                rc = -2;
                goto EXIT;
            }
            if (index < min_index) {
                min_index = index;
            }
            if (index > max_index) {
                max_index = index;
            }
        }
        ++num_samples;
    }
    rewind(fp);

    /* initialise x and y */
    num_features = max_index - min_index + 1;
    gsl_matrix *out_x = gsl_matrix_calloc(num_samples, num_features);
    gsl_vector *out_y = gsl_vector_alloc(num_samples);

    /* second pass: store values */
    j = 0;
    for (i = 0; i < num_samples; ++i) {
        inst_max_index = -1;
        rc = readline(fp, &line);

        /* label */
        label = strtok(line, " \t\n");
        if (label == NULL) {	/* empty line */
            rc = -2;
            goto EXIT;
        }

        gsl_vector_set(out_y, i, strtod(label, &endptr));
        /* if not able to convert or something left after the last valid character */
        if (endptr == label || *endptr != '\0') {
            rc = -2;
            goto EXIT;
        }

        while (1) {
            idx = strtok(NULL, ":");
            val = strtok(NULL, " \t");

            if (val == NULL) {
                break;
            }

            errno = 0;
            index = (int)strtol(idx, &endptr, 10);
            /* similar condition, index is incremental */
            if (endptr == idx || errno != 0 || *endptr != '\0' || index <= inst_max_index) {
                rc = -2;
                goto EXIT;
            }

            inst_max_index = index;

            errno = 0;
            value = strtod(val, &endptr);
            /* similar condition,  if *entprt isn't a '\0', it should be a whitespace */
            if (endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr))) {
                rc = -2;
                goto EXIT;
            }
            gsl_matrix_set(out_x, i, index - min_index, value);
        }
    }

    *x = out_x;
    *y = out_y;
    rc = 0;
    fclose(fp);

EXIT:
    return rc;
}


/*************************************************************
    Compute the mean vector of each features
    parameters: (in)  gsl_matrix *x
                (out) gsl_matrix **mean_vector
    return:		SUCCESS		rc = 0
**************************************************************/
int compute_mean_vector(gsl_matrix *x, gsl_matrix **mean_vector)
{
    gsl_matrix *out_mean_vector = gsl_matrix_alloc(1, x->size2);

    int rc, j;
    for (j = 0; j < x->size2; ++j) {
        /* get column vector */
        gsl_vector_view col_vector = gsl_matrix_column(x, j);

        /* calculate mean value */
        double mean = gsl_stats_mean(col_vector.vector.data, col_vector.vector.stride, col_vector.vector.size);
        gsl_matrix_set(out_mean_vector, 0, j, mean);
    }

    *mean_vector = out_mean_vector;
    rc = 0;
EXIT:
    return rc;
}


/*****************************************************************
    Compute the scatter matrix
    parameters: (in)  gsl_matrix *x,
                      gsl_matrix *mean_vector
                (out) gsl_matrix **scatter_matrix
    return    : SUCCESS		rc = 0
******************************************************************/
int compute_scatter_matrix(gsl_matrix *x, gsl_matrix *mean_vector, gsl_matrix **scatter_matrix)
{
    int rc;
    int i, j;

    /* initialise scatter matrix */
    gsl_matrix *out_scatter_matrix = gsl_matrix_calloc(x->size2, x->size2);

    for (i = 0; i < x->size1; ++i) {
        /* pull out row vector */
        gsl_matrix_view row_vector_view = gsl_matrix_submatrix(x, i, 0, 1, x->size2);
        gsl_matrix *row_vector = gsl_matrix_alloc(row_vector_view.matrix.size1, row_vector_view.matrix.size2);
        gsl_matrix_memcpy(row_vector, &(row_vector_view.matrix));
        /* subtract row vector by mean vector */
        gsl_matrix_sub(row_vector, mean_vector);
        /* compute the dot product between row vector and its transpose */
        gsl_matrix *dot_product = gsl_matrix_calloc(x->size2, x->size2);
        gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, row_vector, row_vector, 0.0, dot_product);
        /* add the result to scatter matrix */
        gsl_matrix_add(out_scatter_matrix, dot_product);
        /* free temp data */
        gsl_matrix_free(dot_product);
        gsl_matrix_free(row_vector);
    }

    *(scatter_matrix) = out_scatter_matrix;

    rc = 0;

EXIT:
    return rc;

}

/*****************************************************************
    Compute the eigenvalue and eigenvector
    parameters: (in)  gsl_matrix *scatter_matrix
                (out) gsl_vector **eigenvalues
                      gsl_matrix **eigenvectors
    return    : SUCCESS		rc = 0
******************************************************************/
int compute_eigen(gsl_matrix *scatter_matrix, gsl_vector **eigenvalues, gsl_matrix **eigenvectors)
{
    int rc;
    gsl_vector *out_eigenvalues = gsl_vector_alloc(scatter_matrix->size1);
    gsl_matrix *out_eigenvectors = gsl_matrix_alloc(scatter_matrix->size1, scatter_matrix->size2);

    gsl_eigen_symmv_workspace *workspace = gsl_eigen_symmv_alloc(scatter_matrix->size1);

    gsl_eigen_symmv(scatter_matrix, out_eigenvalues, out_eigenvectors, workspace);

    gsl_eigen_symmv_free(workspace);

    gsl_eigen_symmv_sort(out_eigenvalues, out_eigenvectors, GSL_EIGEN_SORT_VAL_DESC);

    *eigenvalues = out_eigenvalues;
    *eigenvectors = out_eigenvectors;
    rc = 0;

EXIT:
    return rc;
}



/*****************************************************************
    Transform the original dataset using top k eigenvectors
    parameters: (in)  gsl_matrix *data
                      gsl_matrix *eigenvectors
                      int k
                (out) gsl_matrix **transformed_data
    return    : SUCCESS					rc = 0
                K-dimension not match	rc = -1
******************************************************************/
int transform_dataset(gsl_matrix *data, gsl_matrix *eigenvectors, int k, gsl_matrix **transformed_data)
{
    int rc;

    /* dimensionality check */
    if (k <= 0 || k > eigenvectors->size2) {
        rc = -1;
        goto EXIT;
    }
    /* get top K vectors as submatrix */
    gsl_matrix_view w = gsl_matrix_submatrix(eigenvectors, 0, 0, eigenvectors->size1, k);

    /* transform data */
    gsl_matrix *out_transformed_data = gsl_matrix_calloc(data->size1, w.matrix.size2);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, data, &w, 0.0, out_transformed_data);

    *transformed_data = out_transformed_data;
    rc = 0;

EXIT:
    return rc;
}


/*****************************************************************
    Print the matrix
    parameters: (in)  FILE *f
                      const gsl_matrix *m
    return    : SUCCESS			rc = 0
                Fail to print	rc = -1
******************************************************************/
int print_matrix(FILE *f, const gsl_matrix *m)
{
    int rc;
    for (size_t i = 0; i < m->size1; i++) {
        for (size_t j = 0; j < m->size2; j++) {
            if ((rc = fprintf(f, "%g ", gsl_matrix_get(m, i, j))) < 0) {
                rc = -1;
                goto EXIT;
            }
        }
        if ((rc = fprintf(f, "\n")) < 0) {
            rc = -1;
            goto EXIT;
        }
    }
    rc = 0;
EXIT:
    return rc;
}

/*****************************************************************
    Transform the original dataset using top k eigenvectors
    parameters: (in)  gsl_matrix *x
                      int k
                (out) gsl_matrix **transformed_data
    return    : SUCCESS	rc = 0
                Fail	rc = -1
******************************************************************/
int pca(gsl_matrix *x, int k, gsl_matrix **transformed_data)
{
    int rc;
    
    gsl_matrix *mean_vector;
    rc = compute_mean_vector(x, &mean_vector);
    if (rc != 0) {
        rc = -1;
        goto EXIT;
    }

    gsl_matrix *scatter_matrix;
    rc = compute_scatter_matrix(x, mean_vector, &scatter_matrix);
    if (rc != 0) {
        rc = -1;
        goto EXIT;
    }

    gsl_vector *eigenvalues;
    gsl_matrix *eigenvectors;
    rc = compute_eigen(scatter_matrix, &eigenvalues, &eigenvectors);
    if (rc != 0) {
        rc = -1;
        goto EXIT;
    }

    rc = transform_dataset(x, eigenvectors, k, transformed_data);
    if (rc != 0) {
        rc = -1;
        goto EXIT;
    }

    rc = 0;

    /* free data */
    gsl_matrix_free(mean_vector);
    gsl_matrix_free(scatter_matrix);
    gsl_vector_free(eigenvalues);
    gsl_matrix_free(eigenvectors);

EXIT:
    return rc;
}

int main(int argc, char**argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: pca-compute filename k\n");
        exit(1);
    }

    int rc;
    char *filename = argv[1];
    int k = atoi(argv[2]);
    gsl_matrix *x;
    gsl_vector *y;
    rc = read_file(filename, &x, &y);
    if (rc != 0) {
        fprintf(stderr, "Read File Error\n");
        exit(1);
    }

    gsl_matrix *transformed_data;
    rc = pca(x, k, &transformed_data);
    if (rc != 0) {
        fprintf(stderr, "PCA Failed\n");
        exit(1);
    }

    print_matrix(stdout, transformed_data);
    
    /* free data */
    gsl_matrix_free(x);
    gsl_vector_free(y);
    gsl_matrix_free(transformed_data);
}