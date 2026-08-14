#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <dlfcn.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <ViennaRNA/datastructures/dp_matrices.h>
#include <ViennaRNA/fold_compound.h>
#include <ViennaRNA/io/file_formats.h>
#include <ViennaRNA/model.h>
#include <ViennaRNA/partfunc/global.h>
#include <ViennaRNA/utils/basic.h>


static double
now_seconds(void)
{
  struct timespec value;
  clock_gettime(CLOCK_MONOTONIC, &value);
  return (double)value.tv_sec + (double)value.tv_nsec * 1.e-9;
}


static int
selected_cuda_device(void)
{
  const char  *path = getenv("VRNA_CUDA_LIBRARY");
  void        *handle;
  int         (*selected_device)(void);
  int         device = -1;

  handle = dlopen((path && path[0]) ? path : "libRNA_cuda.so",
                  RTLD_NOW | RTLD_LOCAL);
  if (!handle)
    return -1;

  selected_device = (int (*)(void))dlsym(handle,
                                         "vrna_cuda_pf_selected_device");
  if (selected_device)
    device = selected_device();
  dlclose(handle);
  return device;
}


static size_t
last_cuda_fallback_count(void)
{
  const char  *path = getenv("VRNA_CUDA_LIBRARY");
  void        *handle;
  size_t      (*fallback_count)(void);
  size_t      count = 0;

  handle = dlopen((path && path[0]) ? path : "libRNA_cuda.so",
                  RTLD_NOW | RTLD_LOCAL);
  if (!handle)
    return 0;

  fallback_count = (size_t (*)(void))dlsym(
    handle,
    "vrna_cuda_pf_last_fallback_count");
  if (fallback_count)
    count = fallback_count();
  dlclose(handle);
  return count;
}


static int
require_fallback_free_cuda_run(void)
{
  const size_t count = last_cuda_fallback_count();
  if (count == 0)
    return 1;

  fprintf(stderr,
          "CUDA PF benchmark rejected a run with %zu fallback sequences\n",
          count);
  return 0;
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
  const char          *mode = (argc > 1) ? argv[1] : "cpu-bpp";
  const char          *path = (argc > 2) ? argv[2] : NULL;
  const size_t        requested = (argc > 3) ? strtoul(argv[3], NULL, 10) : 256;
  const unsigned int  iterations = (argc > 4) ? strtoul(argv[4], NULL, 10) : 3;
  const unsigned int  exact_length = (argc > 5) ? strtoul(argv[5], NULL, 10) : 0;
  const int           cuda_mode = strncmp(mode, "cuda-", 5) == 0;
  const int           with_bpp = (strcmp(mode, "cpu-bpp") == 0) ||
                                 (strcmp(mode, "cuda-bpp") == 0);
  FILE                *input = NULL;
  vrna_fold_compound_t **fc = NULL;
  char                **sequences = NULL;
  float               *energies = NULL;
  size_t              count = 0;
  unsigned int        min_length = 0;
  unsigned int        max_length = 0;
  int                 result = EXIT_FAILURE;

  if ((!path) || (requested == 0) || (iterations == 0)) {
    fprintf(stderr, "usage: %s MODE FASTA COUNT ITERATIONS [EXACT_LENGTH]\n", argv[0]);
    return EXIT_FAILURE;
  }

  if ((strcmp(mode, "cpu-pf") != 0) &&
      (strcmp(mode, "cpu-bpp") != 0) &&
      (strcmp(mode, "cuda-pf") != 0) &&
      (strcmp(mode, "cuda-bpp") != 0)) {
    fprintf(stderr, "mode must be cpu-pf, cpu-bpp, cuda-pf, or cuda-bpp\n");
    return EXIT_FAILURE;
  }

  input     = fopen(path, "r");
  fc        = (vrna_fold_compound_t **)calloc(requested, sizeof(*fc));
  sequences = (char **)calloc(requested, sizeof(*sequences));
  energies  = (float *)calloc(requested, sizeof(*energies));
  if ((!input) || (!fc) || (!sequences) || (!energies))
    goto cleanup;

  while (count < requested) {
    char                *header = NULL;
    char                *sequence = NULL;
    char                **rest = NULL;
    const unsigned int  record = vrna_file_fasta_read_record(&header,
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

    vrna_md_t md;
    vrna_md_set_default(&md);
    md.compute_bpp    = with_bpp;
    sequences[count] = sequence;
    fc[count]         = vrna_fold_compound(sequence, &md, VRNA_OPTION_PF);
    if (!fc[count])
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

  /* Fold-compound construction and the first allocation-heavy run stay
   * outside the timed region, matching the exact MFE FASTA harness. */
  if (cuda_mode) {
    if (!vrna_pf_batch(fc,
                       count,
                       with_bpp ? VRNA_PF_BATCH_BPP_DENSE : VRNA_PF_BATCH_ENERGY,
                       energies) ||
        !require_fallback_free_cuda_run())
      goto cleanup;
  } else {
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < count; i++)
      energies[i] = (float)vrna_pf(fc[i], NULL);
  }

  const double start = now_seconds();
  for (unsigned int iteration = 0; iteration < iterations; iteration++) {
    if (cuda_mode) {
      if (!vrna_pf_batch(fc,
                         count,
                         with_bpp ? VRNA_PF_BATCH_BPP_DENSE : VRNA_PF_BATCH_ENERGY,
                         energies) ||
          !require_fallback_free_cuda_run())
        goto cleanup;
    } else {
#pragma omp parallel for schedule(dynamic)
      for (size_t i = 0; i < count; i++)
        energies[i] = (float)vrna_pf(fc[i], NULL);
    }
  }
  const double elapsed = now_seconds() - start;
  const int selected_device = cuda_mode ? selected_cuda_device() : -1;
  const size_t fallback_count = cuda_mode ? last_cuda_fallback_count() : 0;

  double energy_checksum = 0.;
  double bpp_checksum = 0.;
  for (size_t b = 0; b < count; b++) {
    energy_checksum += energies[b];
    if (with_bpp) {
      const unsigned int n = fc[b]->length;
      for (unsigned int i = 1; i < n; i++)
        for (unsigned int j = i + 1; j <= n; j++)
          bpp_checksum += fc[b]->exp_matrices->probs[fc[b]->iindx[i] - j];
    }
  }

  printf("mode=%s source=%s count=%zu min_length=%u max_length=%u iterations=%u "
         "selected_device=%d fallback_count=%zu "
         "seconds=%.6f total_seconds=%.6f seq_per_s=%.3f "
         "energy_checksum=%.9g bpp_checksum=%.9g\n",
         mode,
         path,
         count,
         min_length,
         max_length,
         iterations,
         selected_device,
         fallback_count,
         elapsed / iterations,
         elapsed,
         (double)(count * iterations) / elapsed,
         energy_checksum,
         bpp_checksum);
  result = EXIT_SUCCESS;

cleanup:
  if (input)
    fclose(input);
  for (size_t i = 0; i < requested; i++) {
    vrna_fold_compound_free(fc ? fc[i] : NULL);
    free(sequences ? sequences[i] : NULL);
  }
  free(fc);
  free(sequences);
  free(energies);
  return result;
}
