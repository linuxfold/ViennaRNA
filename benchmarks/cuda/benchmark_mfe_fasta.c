#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <ViennaRNA/fold_compound.h>
#include <ViennaRNA/io/file_formats.h>
#include <ViennaRNA/mfe/global.h>
#include <ViennaRNA/utils/basic.h>


static double
now_seconds(void)
{
  struct timespec value;
  clock_gettime(CLOCK_MONOTONIC, &value);
  return (double)value.tv_sec + (double)value.tv_nsec * 1.e-9;
}


static void
free_rest(char **rest)
{
  if (!rest)
    return;

  for (size_t i = 0; rest[i]; i++)
    free(rest[i]);
  free(rest);
}


static int
normalize_sequence(char *sequence)
{
  if (!sequence)
    return 0;

  for (char *base = sequence; *base; base++) {
    *base = (char)toupper((unsigned char)*base);
    if (*base == 'T')
      *base = 'U';
    if (!strchr("ACGUN", *base))
      return 0;
  }

  return 1;
}


int
main(int argc,
     char **argv)
{
  const char *mode = (argc > 1) ? argv[1] : "cuda";
  const char *path = (argc > 2) ? argv[2] : NULL;
  const size_t requested = (argc > 3) ? strtoul(argv[3], NULL, 10) : 256;
  const unsigned int iterations = (argc > 4) ? strtoul(argv[4], NULL, 10) : 3;
  const unsigned int exact_length = (argc > 5) ? strtoul(argv[5], NULL, 10) : 0;
  const int with_backtrack = strstr(mode, "energy") == NULL;
  const int cpu_mode = (strcmp(mode, "cpu") == 0) || (strcmp(mode, "cpu-energy") == 0);
  const int cuda_mode = (strcmp(mode, "cuda") == 0) || (strcmp(mode, "cuda-energy") == 0);
  FILE *input = NULL;
  vrna_fold_compound_t **fc = NULL;
  char **sequences = NULL;
  char **structures = NULL;
  float *energies = NULL;
  size_t count = 0;
  unsigned int min_length = 0;
  unsigned int max_length = 0;
  int result = EXIT_FAILURE;

  if ((!path) || (requested == 0) || (iterations == 0)) {
    fprintf(stderr, "usage: %s MODE FASTA COUNT ITERATIONS [EXACT_LENGTH]\n", argv[0]);
    return EXIT_FAILURE;
  }

  if ((!cpu_mode) && (!cuda_mode)) {
    fprintf(stderr, "mode must be cpu, cpu-energy, cuda, or cuda-energy\n");
    return EXIT_FAILURE;
  }

  input = fopen(path, "r");
  fc = (vrna_fold_compound_t **)calloc(requested, sizeof(*fc));
  sequences = (char **)calloc(requested, sizeof(*sequences));
  structures = (char **)calloc(requested, sizeof(*structures));
  energies = (float *)calloc(requested, sizeof(*energies));
  if ((!input) || (!fc) || (!sequences) || (!structures) || (!energies))
    goto cleanup;

  while (count < requested) {
    char *header = NULL;
    char *sequence = NULL;
    char **rest = NULL;
    const unsigned int record = vrna_file_fasta_read_record(&header,
                                                             &sequence,
                                                             &rest,
                                                             input,
                                                             VRNA_INPUT_NO_REST);
    free(header);
    free_rest(rest);

    if (record & (VRNA_INPUT_ERROR | VRNA_INPUT_QUIT)) {
      free(sequence);
      break;
    }

    if ((!normalize_sequence(sequence)) || (sequence[0] == '\0')) {
      fprintf(stderr, "unsupported sequence in FASTA record %zu\n", count + 1);
      free(sequence);
      goto cleanup;
    }

    const unsigned int length = (unsigned int)strlen(sequence);
    if (exact_length && (length != exact_length)) {
      free(sequence);
      continue;
    }

    sequences[count] = sequence;
    structures[count] = (char *)calloc(length + 1, sizeof(char));
    fc[count] = vrna_fold_compound(sequence, NULL, VRNA_OPTION_MFE);
    if ((!structures[count]) || (!fc[count]))
      goto cleanup;

    if ((min_length == 0) || (length < min_length))
      min_length = length;
    if (length > max_length)
      max_length = length;
    count++;
  }

  if (count != requested) {
    fprintf(stderr, "requested %zu records but read %zu\n", requested, count);
    goto cleanup;
  }

  /* Keep one-time runtime, plugin, and memory-pool initialization out of the
   * measurements on both backends. */
  if (cpu_mode) {
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < count; i++)
      energies[i] = vrna_mfe(fc[i], with_backtrack ? structures[i] : NULL);
  } else if (!vrna_mfe_batch(fc, count, with_backtrack ? structures : NULL, energies)) {
    fprintf(stderr, "vrna_mfe_batch warm-up returned failure\n");
    goto cleanup;
  }

  const double start = now_seconds();
  if (cpu_mode) {
    for (unsigned int iteration = 0; iteration < iterations; iteration++) {
#pragma omp parallel for schedule(dynamic)
      for (size_t i = 0; i < count; i++)
        energies[i] = vrna_mfe(fc[i], with_backtrack ? structures[i] : NULL);
    }
  } else {
    for (unsigned int iteration = 0; iteration < iterations; iteration++) {
      if (!vrna_mfe_batch(fc, count, with_backtrack ? structures : NULL, energies)) {
        fprintf(stderr, "vrna_mfe_batch returned failure\n");
        goto cleanup;
      }
    }
  }

  const double elapsed = now_seconds() - start;
  double checksum = 0.;
  for (size_t i = 0; i < count; i++)
    checksum += energies[i] + (with_backtrack ? structures[i][strlen(sequences[i]) / 2] : 0);

  printf("mode=%s source=%s count=%zu min_length=%u max_length=%u iterations=%u "
         "seconds=%.6f total_seconds=%.6f seq_per_s=%.3f cpu_threads=%d checksum=%.3f\n",
         mode,
         path,
         count,
         min_length,
         max_length,
         iterations,
         elapsed / iterations,
         elapsed,
         (double)(count * iterations) / elapsed,
         omp_get_max_threads(),
         checksum);
  result = EXIT_SUCCESS;

cleanup:
  if (input)
    fclose(input);
  for (size_t i = 0; i < requested; i++) {
    vrna_fold_compound_free(fc ? fc[i] : NULL);
    free(sequences ? sequences[i] : NULL);
    free(structures ? structures[i] : NULL);
  }
  free(fc);
  free(sequences);
  free(structures);
  free(energies);
  return result;
}
