#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <ViennaRNA/datastructures/dp_matrices.h>
#include <ViennaRNA/fold_compound.h>
#include <ViennaRNA/model.h>
#include <ViennaRNA/partfunc/adjoint.h>
#include <ViennaRNA/partfunc/global.h>


static int
check_sequence(const char *sequence,
               double     *global_max_error,
               size_t     *global_cells)
{
  FLT_OR_DBL            *adjoint, *reference;
  vrna_fold_compound_t  *fc;
  vrna_md_t             md;
  unsigned int          i, j, n;
  size_t                matrix_size;
  double                max_error = 0.;
  int                   result = 0;

  vrna_md_set_default(&md);
  md.compute_bpp = 1;
  fc             = vrna_fold_compound(sequence, &md, VRNA_OPTION_PF);
  if (!fc)
    return 0;

  (void)vrna_pf(fc, NULL);
  n           = fc->length;
  matrix_size = ((size_t)n + 1) * ((size_t)n + 2) / 2;
  reference   = (FLT_OR_DBL *)malloc(sizeof(FLT_OR_DBL) * matrix_size);
  if (!reference)
    goto cleanup;

  for (size_t cell = 0; cell < matrix_size; cell++) {
    reference[cell]                = fc->exp_matrices->probs[cell];
    fc->exp_matrices->probs[cell]  = (FLT_OR_DBL)NAN;
  }

  /* Poisoning probs above ensures that the oracle is independent of the
   * existing outside implementation. */
  adjoint = vrna_pf_adjoint_oracle(fc);
  if (!adjoint) {
    fprintf(stderr, "adjoint oracle rejected supported sequence %s\n", sequence);
    goto cleanup;
  }

  for (i = 1; i < n; i++) {
    for (j = i + 1; j <= n; j++) {
      const int     ij       = fc->iindx[i] - j;
      const double  expected = reference[ij];
      const double  observed = fc->exp_matrices->qb[ij] * adjoint[ij];
      const double  error    = fabs(expected - observed);

      if (error > max_error)
        max_error = error;

      (*global_cells)++;
      if (error > 5.e-10) {
        fprintf(stderr,
                "p=B*dB mismatch for %s at (%u,%u): expected %.17g, "
                "observed %.17g, error %.3g\n",
                sequence,
                i,
                j,
                expected,
                observed,
                error);
        free(adjoint);
        goto cleanup;
      }
    }
  }

  if (max_error > *global_max_error)
    *global_max_error = max_error;

  free(adjoint);
  result = 1;

cleanup:
  free(reference);
  vrna_fold_compound_free(fc);
  return result;
}


int
main(void)
{
  const char *sequences[] = {
    "GCGCUUCGCCGAAAGGCGAAGCGC",
    "GGGAAACCCUUUGGGAAACCC",
    "AUGCUAGCUAGCUACGUAUGCUAGCUAGC",
    "GCAUCGGAUCCGAUCGGAUCCGAUCGGAUC"
  };
  const size_t  count = sizeof(sequences) / sizeof(sequences[0]);
  double        max_error = 0.;
  size_t        cells = 0;

  for (size_t s = 0; s < count; s++)
    if (!check_sequence(sequences[s], &max_error, &cells))
      return EXIT_FAILURE;

  {
    vrna_md_t             md;
    vrna_fold_compound_t  *unsupported;

    vrna_md_set_default(&md);
    md.dangles  = 0;
    unsupported = vrna_fold_compound("GCGCUUCGCCGAAAGGCGAAGCGC",
                                     &md,
                                     VRNA_OPTION_PF);
    if ((!unsupported) ||
        (vrna_pf(unsupported, NULL) >= INF / 100.) ||
        (vrna_pf_adjoint_oracle(unsupported) != NULL)) {
      fprintf(stderr, "adjoint oracle accepted unsupported dangles model\n");
      vrna_fold_compound_free(unsupported);
      return EXIT_FAILURE;
    }
    vrna_fold_compound_free(unsupported);
  }

  printf("CPU PF adjoint identity: %zu sequences, %zu cells, max error %.3g\n",
         count,
         cells,
         max_error);
  return EXIT_SUCCESS;
}
