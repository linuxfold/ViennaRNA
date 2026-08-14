#define _POSIX_C_SOURCE 200809L

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <ViennaRNA/fold_compound.h>
#include <ViennaRNA/mfe/global.h>
#include <ViennaRNA/model.h>


static unsigned int random_state = 0x243f6a88U;


static unsigned int
random_u32(void)
{
  random_state ^= random_state << 13;
  random_state ^= random_state >> 17;
  random_state ^= random_state << 5;
  return random_state;
}


static double
now_seconds(void)
{
  struct timespec value;
  clock_gettime(CLOCK_MONOTONIC, &value);
  return (double)value.tv_sec + (double)value.tv_nsec * 1.e-9;
}


int
main(int argc,
     char **argv)
{
  static const char alphabet[] = "ACGU";
  const char *mode = (argc > 1) ? argv[1] : "cuda";
  const size_t count = (argc > 2) ? strtoul(argv[2], NULL, 10) : 256;
  const unsigned int length = (argc > 3) ? strtoul(argv[3], NULL, 10) : 500;
  const unsigned int iterations = (argc > 4) ? strtoul(argv[4], NULL, 10) : 1;
  vrna_fold_compound_t **fc = (vrna_fold_compound_t **)calloc(count, sizeof(*fc));
  char **sequences = (char **)calloc(count, sizeof(*sequences));
  char **structures = (char **)calloc(count, sizeof(*structures));
  float *energies = (float *)calloc(count, sizeof(*energies));
  int result = EXIT_FAILURE;
  double start, elapsed;
  double checksum = 0.;
  const int with_backtrack = strstr(mode, "energy") == NULL;
  const int no_lp = strstr(mode, "nolp") != NULL;
  const int cpu_mode = (strcmp(mode, "cpu") == 0) ||
                       (strcmp(mode, "cpu-energy") == 0) ||
                       (strcmp(mode, "cpu-nolp") == 0) ||
                       (strcmp(mode, "cpu-energy-nolp") == 0);
  const int cuda_mode = (strcmp(mode, "cuda") == 0) ||
                        (strcmp(mode, "cuda-energy") == 0) ||
                        (strcmp(mode, "cuda-nolp") == 0) ||
                        (strcmp(mode, "cuda-energy-nolp") == 0);

  if ((!fc) || (!sequences) || (!structures) || (!energies) ||
      (count == 0) || (length == 0) || (iterations == 0))
    goto cleanup;

  if ((!cpu_mode) && (!cuda_mode)) {
    fprintf(stderr,
            "mode must be cpu, cpu-energy, cuda, or cuda-energy, optionally suffixed with -nolp\n");
    goto cleanup;
  }

  for (size_t b = 0; b < count; b++) {
    vrna_md_t md;

    sequences[b] = (char *)calloc(length + 1, sizeof(char));
    structures[b] = (char *)calloc(length + 1, sizeof(char));
    if ((!sequences[b]) || (!structures[b]))
      goto cleanup;

    for (unsigned int i = 0; i < length; i++)
      sequences[b][i] = alphabet[random_u32() & 3U];

    vrna_md_set_default(&md);
    md.noLP = no_lp;
    fc[b] = vrna_fold_compound(sequences[b], &md, VRNA_OPTION_MFE);
    if (!fc[b])
      goto cleanup;
  }

  /* Exclude one-time runtime, plugin, and memory-pool initialization from
   * both backends. */
  if (cpu_mode) {
#pragma omp parallel for schedule(dynamic)
    for (size_t b = 0; b < count; b++)
      energies[b] = vrna_mfe(fc[b], with_backtrack ? structures[b] : NULL);
  } else if (!vrna_mfe_batch(fc, count, with_backtrack ? structures : NULL, energies)) {
    fprintf(stderr, "vrna_mfe_batch warm-up returned failure\n");
    goto cleanup;
  }

  start = now_seconds();

  if (cpu_mode) {
    for (unsigned int iteration = 0; iteration < iterations; iteration++) {
#pragma omp parallel for schedule(dynamic)
      for (size_t b = 0; b < count; b++)
        energies[b] = vrna_mfe(fc[b], with_backtrack ? structures[b] : NULL);
    }
  } else {
    for (unsigned int iteration = 0; iteration < iterations; iteration++) {
      if (!vrna_mfe_batch(fc, count, with_backtrack ? structures : NULL, energies)) {
        fprintf(stderr, "vrna_mfe_batch returned failure\n");
        goto cleanup;
      }
    }
  }

  elapsed = now_seconds() - start;

  for (size_t b = 0; b < count; b++)
    checksum += energies[b] + (with_backtrack ? structures[b][length / 2] : 0);

  printf("mode=%s count=%zu length=%u iterations=%u seconds=%.6f total_seconds=%.6f "
         "seq_per_s=%.3f cpu_threads=%d checksum=%.3f\n",
         mode,
         count,
         length,
         iterations,
         elapsed / iterations,
         elapsed,
         (double)(count * iterations) / elapsed,
         omp_get_max_threads(),
         checksum);
  result = EXIT_SUCCESS;

cleanup:
  for (size_t b = 0; b < count; b++) {
    vrna_fold_compound_free(fc ? fc[b] : NULL);
    free(sequences ? sequences[b] : NULL);
    free(structures ? structures[b] : NULL);
  }

  free(fc);
  free(sequences);
  free(structures);
  free(energies);

  return result;
}
