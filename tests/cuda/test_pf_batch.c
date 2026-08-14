#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <ViennaRNA/datastructures/dp_matrices.h>
#include <ViennaRNA/fold_compound.h>
#include <ViennaRNA/model.h>
#include <ViennaRNA/partfunc/global.h>


int
main(void)
{
  const char *sequences[] = {
    "GCGCUUCGCCGAAAGGCGAAGCGC",
    "GGGAAACCCUUUGGGAAACCC",
    "AUGCUAGCUAGCUACGUAUGCUAGCUAGC",
    "GCAUCGGAUCCGAUCGGAUCCGAUCGGAUC"
  };
  const size_t count = sizeof(sequences) / sizeof(sequences[0]);
  vrna_fold_compound_t *cpu[count];
  vrna_fold_compound_t *gpu[count];
  FLT_OR_DBL *reference[count];
  float energies[count];
  double max_probability_error = 0.;
  double max_energy_error = 0.;
  size_t cells = 0;
  int result = EXIT_FAILURE;

  memset(cpu, 0, sizeof(cpu));
  memset(gpu, 0, sizeof(gpu));
  memset(reference, 0, sizeof(reference));

  for (size_t s = 0; s < count; s++) {
    vrna_md_t md;
    vrna_md_set_default(&md);
    md.compute_bpp = 1;
    cpu[s] = vrna_fold_compound(sequences[s], &md, VRNA_OPTION_PF);
    gpu[s] = vrna_fold_compound(sequences[s], &md, VRNA_OPTION_PF);
    if ((!cpu[s]) || (!gpu[s]))
      goto cleanup;

    const unsigned int n = cpu[s]->length;
    const size_t matrix_size = ((size_t)n + 1) * ((size_t)n + 2) / 2;
    reference[s] = (FLT_OR_DBL *)malloc(sizeof(FLT_OR_DBL) * matrix_size);
    if (!reference[s])
      goto cleanup;

    const double cpu_energy = vrna_pf(cpu[s], NULL);
    memcpy(reference[s],
           cpu[s]->exp_matrices->probs,
           sizeof(FLT_OR_DBL) * matrix_size);
    energies[s] = (float)cpu_energy;
  }

  if (!vrna_pf_batch(gpu, count, VRNA_PF_BATCH_BPP_DENSE, energies)) {
    fprintf(stderr, "vrna_pf_batch returned failure\n");
    goto cleanup;
  }

  for (size_t s = 0; s < count; s++) {
    const double cpu_energy = vrna_pf(cpu[s], NULL);
    const double energy_error = fabs(cpu_energy - energies[s]);
    if (energy_error > max_energy_error)
      max_energy_error = energy_error;
    if (energy_error > 2.e-5) {
      fprintf(stderr,
              "ensemble energy mismatch for %s: CPU %.17g GPU %.17g error %.3g\n",
              sequences[s],
              cpu_energy,
              (double)energies[s],
              energy_error);
      goto cleanup;
    }

    const unsigned int n = gpu[s]->length;
    for (unsigned int i = 1; i < n; i++)
      for (unsigned int j = i + 1; j <= n; j++) {
        const int ij = gpu[s]->iindx[i] - j;
        const double expected = reference[s][ij];
        const double observed = gpu[s]->exp_matrices->probs[ij];
        const double error = fabs(expected - observed);
        if (error > max_probability_error)
          max_probability_error = error;
        cells++;
        if (error > 5.e-10) {
          fprintf(stderr,
                  "BPP mismatch for %s at (%u,%u): CPU %.17g GPU %.17g error %.3g\n",
                  sequences[s],
                  i,
                  j,
                  expected,
                  observed,
                  error);
          goto cleanup;
        }
      }
  }

  {
    vrna_md_t md;
    vrna_fold_compound_t *fallback;
    vrna_fold_compound_t *inputs[1];
    float fallback_energy;
    vrna_md_set_default(&md);
    md.compute_bpp = 1;
    md.dangles = 0;
    fallback = vrna_fold_compound(sequences[0], &md, VRNA_OPTION_PF);
    if (!fallback)
      goto cleanup;
    inputs[0] = fallback;
    if ((!vrna_pf_batch(inputs, 1, VRNA_PF_BATCH_BPP_DENSE, &fallback_energy)) ||
        (!isfinite(fallback_energy))) {
      fprintf(stderr, "unsupported model did not fall back to CPU\n");
      vrna_fold_compound_free(fallback);
      goto cleanup;
    }
    vrna_fold_compound_free(fallback);
  }

  printf("CUDA PF/BPP match: %zu sequences, %zu cells, max energy error %.3g, "
         "max probability error %.3g\n",
         count,
         cells,
         max_energy_error,
         max_probability_error);
  result = EXIT_SUCCESS;

cleanup:
  for (size_t s = 0; s < count; s++) {
    free(reference[s]);
    vrna_fold_compound_free(cpu[s]);
    vrna_fold_compound_free(gpu[s]);
  }
  return result;
}
