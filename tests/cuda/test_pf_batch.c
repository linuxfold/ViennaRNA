#define _POSIX_C_SOURCE 200809L

#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <ViennaRNA/datastructures/dp_matrices.h>
#include <ViennaRNA/fold_compound.h>
#include <ViennaRNA/model.h>
#include <ViennaRNA/partfunc/global.h>

#define VRNA_CUDA_PF_BATCH_SYMBOL "vrna_cuda_pf_batch"

typedef int (*vrna_cuda_pf_batch_f)(vrna_fold_compound_t **fc,
                                    size_t               count,
                                    unsigned char        *handled,
                                    float                *energies,
                                    unsigned int         flags);


typedef struct {
  char                  *sequence;
  vrna_fold_compound_t  *cpu;
  FLT_OR_DBL            *probabilities;
  double                energy;
  size_t                matrix_size;
} test_case_t;


typedef struct {
  const char  *engine;
  const char  *gemm;
  const char  *precision;
  const char  *scale_adjustment;
  unsigned int flags;
  const char  *label;
} test_mode_t;


static char *
make_pattern(const char  *pattern,
             size_t      length)
{
  const size_t  pattern_length = strlen(pattern);
  char          *sequence = (char *)malloc(length + 1);

  if (!sequence)
    return NULL;

  for (size_t i = 0; i < length; i++)
    sequence[i] = pattern[i % pattern_length];
  sequence[length] = '\0';
  return sequence;
}


static char *
make_random(size_t   length,
            uint32_t *state)
{
  static const char alphabet[] = "ACGU";
  char              *sequence = (char *)malloc(length + 1);

  if (!sequence)
    return NULL;

  for (size_t i = 0; i < length; i++) {
    *state      = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    sequence[i] = alphabet[*state >> 30];
  }
  sequence[length] = '\0';
  return sequence;
}


static int
initialize_cases(test_case_t *cases,
                 size_t      count)
{
  static const char *fixed[] = {
    "GCGCUUCGCCGAAAGGCGAAGCGC",
    "GGGAAACCCUUUGGGAAACCC",
    "GCAAAAGC",
  };
  static const size_t random_lengths[] = {
    31, 64, 97, 128, 160, 200, 900, 900, 900
  };
  uint32_t            random_state = UINT32_C(0x5eed1234);
  size_t              next = 0;

  memset(cases, 0, sizeof(*cases) * count);
  for (size_t i = 0; i < sizeof(fixed) / sizeof(fixed[0]); i++)
    cases[next++].sequence = strdup(fixed[i]);
  cases[next++].sequence = make_pattern("GC", 48);
  cases[next++].sequence = make_pattern("AU", 57);
  cases[next++].sequence = make_pattern("GU", 62);
  cases[next++].sequence = make_pattern("A", 39);
  cases[next++].sequence = make_pattern("ACGUN", 71);
  cases[next++].sequence = make_pattern("GCAU", 83);
  for (size_t i = 0; i < sizeof(random_lengths) / sizeof(random_lengths[0]); i++)
    cases[next++].sequence = make_random(random_lengths[i], &random_state);

  if (next != count)
    return 0;

  for (size_t s = 0; s < count; s++) {
    vrna_md_t md;
    if (!cases[s].sequence)
      return 0;
    vrna_md_set_default(&md);
    md.compute_bpp = 1;
    cases[s].cpu = vrna_fold_compound(cases[s].sequence, &md, VRNA_OPTION_PF);
    if (!cases[s].cpu)
      return 0;

    cases[s].matrix_size = ((size_t)cases[s].cpu->length + 1) *
                           ((size_t)cases[s].cpu->length + 2) / 2;
    cases[s].probabilities = (FLT_OR_DBL *)malloc(sizeof(FLT_OR_DBL) *
                                                  cases[s].matrix_size);
    if (!cases[s].probabilities)
      return 0;

    cases[s].energy = vrna_pf(cases[s].cpu, NULL);
    memcpy(cases[s].probabilities,
           cases[s].cpu->exp_matrices->probs,
           sizeof(FLT_OR_DBL) * cases[s].matrix_size);
  }
  return 1;
}


static void
free_cases(test_case_t *cases,
           size_t      count)
{
  for (size_t s = 0; s < count; s++) {
    free(cases[s].sequence);
    free(cases[s].probabilities);
    vrna_fold_compound_free(cases[s].cpu);
  }
}


static int
run_mode(const test_mode_t         *mode,
         const test_case_t         *cases,
         size_t                    count,
         vrna_cuda_pf_batch_f      cuda_batch,
         double                    *global_energy_error,
         double                    *global_probability_error,
         size_t                    *global_cells)
{
  vrna_fold_compound_t **gpu = NULL;
  FLT_OR_DBL          **first_probabilities = NULL;
  float               *energies = NULL;
  float               *first_energies = NULL;
  unsigned char       *handled = NULL;
  const int           with_bpp =
    (mode->flags & VRNA_PF_BATCH_BPP_DENSE) != 0;
  int                 result = 0;

  if ((setenv("VRNA_CUDA_PF_ENGINE", mode->engine, 1) != 0) ||
      (setenv("VRNA_CUDA_PF_GEMM", mode->gemm, 1) != 0) ||
      (setenv("VRNA_CUDA_PF_PRECISION", mode->precision, 1) != 0) ||
      (setenv("VRNA_CUDA_PF_ACCURACY_FALLBACK", "0", 1) != 0) ||
      (mode->scale_adjustment ?
       (setenv("VRNA_CUDA_PF_SCALE_ADJUSTMENT",
               mode->scale_adjustment,
               1) != 0) :
       (unsetenv("VRNA_CUDA_PF_SCALE_ADJUSTMENT") != 0)))
    return 0;

  gpu                 = (vrna_fold_compound_t **)calloc(count, sizeof(*gpu));
  first_probabilities = (FLT_OR_DBL **)calloc(count,
                                               sizeof(*first_probabilities));
  energies            = (float *)calloc(count, sizeof(*energies));
  first_energies      = (float *)calloc(count, sizeof(*first_energies));
  handled             = (unsigned char *)calloc(count, sizeof(*handled));
  if ((!gpu) || (!first_probabilities) || (!energies) || (!first_energies) ||
      (!handled))
    goto cleanup;

  for (size_t s = 0; s < count; s++) {
    vrna_md_t md;
    vrna_md_set_default(&md);
    md.compute_bpp = 1;
    gpu[s] = vrna_fold_compound(cases[s].sequence, &md, VRNA_OPTION_PF);
    if ((!gpu[s]) ||
        (!vrna_fold_compound_prepare(gpu[s], VRNA_OPTION_PF)))
      goto cleanup;
    first_probabilities[s] = (FLT_OR_DBL *)malloc(sizeof(FLT_OR_DBL) *
                                                   cases[s].matrix_size);
    if (!first_probabilities[s])
      goto cleanup;
  }

  if (!cuda_batch(gpu,
                  count,
                  handled,
                  energies,
                  mode->flags)) {
    fprintf(stderr, "%s backend call failed\n", mode->label);
    goto cleanup;
  }

  for (size_t s = 0; s < count; s++) {
    const unsigned int n = gpu[s]->length;
    double paired_sum_max = 0.;
    const double energy_error = fabs(cases[s].energy - energies[s]);

    if (!handled[s]) {
      fprintf(stderr, "%s did not handle eligible sequence %zu\n", mode->label, s);
      goto cleanup;
    }
    if ((!isfinite(energies[s])) || (energy_error > 2.e-5)) {
      fprintf(stderr,
              "%s energy mismatch at sequence %zu: CPU %.17g GPU %.17g "
              "error %.3g\n",
              mode->label,
              s,
              cases[s].energy,
              (double)energies[s],
              energy_error);
      goto cleanup;
    }
    if (energy_error > *global_energy_error)
      *global_energy_error = energy_error;

    if (with_bpp) {
      for (unsigned int i = 1; i <= n; i++) {
        double paired_sum = 0.;
        for (unsigned int j = 1; j <= n; j++) {
          unsigned int left = i;
          unsigned int right = j;
          if (left == right)
            continue;
          if (left > right) {
            left  = j;
            right = i;
          }
          const int ij = gpu[s]->iindx[left] - right;
          const double observed = gpu[s]->exp_matrices->probs[ij];
          if (j > i) {
            const double expected = cases[s].probabilities[ij];
            const double error = fabs(expected - observed);
            if (error > *global_probability_error)
              *global_probability_error = error;
            (*global_cells)++;
            if (error > 5.e-10) {
              fprintf(stderr,
                      "%s BPP mismatch for sequence %zu at (%u,%u): CPU %.17g "
                      "GPU %.17g error %.3g\n",
                      mode->label,
                      s,
                      i,
                      j,
                      expected,
                      observed,
                      error);
              goto cleanup;
            }
          }
          if ((!isfinite(observed)) || (observed < -5.e-12) ||
              (observed > 1. + 5.e-10)) {
            fprintf(stderr,
                    "%s invalid probability for sequence %zu at (%u,%u): %.17g\n",
                    mode->label,
                    s,
                    left,
                    right,
                    observed);
            goto cleanup;
          }
          paired_sum += observed;
        }
        if (paired_sum > paired_sum_max)
          paired_sum_max = paired_sum;
      }
      if (paired_sum_max > 1. + 5.e-10) {
        fprintf(stderr,
                "%s paired sum exceeds one for sequence %zu: %.17g\n",
                mode->label,
                s,
                paired_sum_max);
        goto cleanup;
      }
    }

    first_energies[s] = energies[s];
    if (with_bpp)
      memcpy(first_probabilities[s],
             gpu[s]->exp_matrices->probs,
             sizeof(FLT_OR_DBL) * cases[s].matrix_size);
  }

  memset(handled, 0, count);
  if (!cuda_batch(gpu,
                  count,
                  handled,
                  energies,
                  mode->flags)) {
    fprintf(stderr, "%s repeat backend call failed\n", mode->label);
    goto cleanup;
  }
  for (size_t s = 0; s < count; s++) {
    if ((!handled[s]) || (energies[s] != first_energies[s]) ||
        (with_bpp &&
         (memcmp(first_probabilities[s],
                 gpu[s]->exp_matrices->probs,
                 sizeof(FLT_OR_DBL) * cases[s].matrix_size) != 0))) {
      fprintf(stderr, "%s repeat was not bitwise stable at sequence %zu\n",
              mode->label,
              s);
      goto cleanup;
    }
  }

  printf("%s CUDA %s match: %zu sequences through length 900\n",
         mode->label,
         with_bpp ? "PF/BPP" : "PF",
         count);
  result = 1;

cleanup:
  for (size_t s = 0; s < count; s++) {
    free(first_probabilities ? first_probabilities[s] : NULL);
    vrna_fold_compound_free(gpu ? gpu[s] : NULL);
  }
  free(gpu);
  free(first_probabilities);
  free(energies);
  free(first_energies);
  free(handled);
  return result;
}


int
main(void)
{
  enum { case_count = 18 };
  static const test_mode_t modes[] = {
    { "blocked",   "native",   "fp64",   NULL,
      VRNA_PF_BATCH_BPP_DENSE, "blocked-native" },
    { "blocked",   "emulated", "fp64",   NULL,
      VRNA_PF_BATCH_BPP_DENSE, "blocked-emulated" },
    { "reference", "native",   "fp64",   NULL,
      VRNA_PF_BATCH_BPP_DENSE, "reference-DAG" },
    { "blocked",   "native",   "ffloat", ".815",
      0U, "blocked-float-float" },
  };
  const char                  *library = getenv("VRNA_CUDA_LIBRARY");
  test_case_t                 cases[case_count];
  void                        *handle = NULL;
  vrna_cuda_pf_batch_f        cuda_batch = NULL;
  int                         (*selected_device)(void) = NULL;
  int                         initial_device;
  double                      max_energy_error = 0.;
  double                      max_probability_error = 0.;
  size_t                      cells = 0;
  int                         result = EXIT_FAILURE;

  memset(cases, 0, sizeof(cases));
  handle = dlopen((library && library[0]) ? library : "libRNA_cuda.so",
                  RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    fprintf(stderr, "could not load CUDA backend: %s\n", dlerror());
    goto cleanup;
  }
  cuda_batch = (vrna_cuda_pf_batch_f)dlsym(handle, VRNA_CUDA_PF_BATCH_SYMBOL);
  selected_device = (int (*)(void))dlsym(handle,
                                         "vrna_cuda_pf_selected_device");
  if ((!cuda_batch) || (!selected_device)) {
    fprintf(stderr, "CUDA PF backend is missing required symbols\n");
    goto cleanup;
  }

  initial_device = selected_device();
  if (initial_device < 0) {
    fprintf(stderr, "CUDA PF backend has no selected device\n");
    goto cleanup;
  }
  if (!initialize_cases(cases, case_count))
    goto cleanup;

  for (size_t mode = 0; mode < sizeof(modes) / sizeof(modes[0]); mode++)
    if (!run_mode(modes + mode,
                  cases,
                  case_count,
                  cuda_batch,
                  &max_energy_error,
                  &max_probability_error,
                  &cells))
      goto cleanup;

  if (selected_device() != initial_device) {
    fprintf(stderr, "CUDA PF backend changed the selected device\n");
    goto cleanup;
  }

  {
    vrna_md_t             md;
    vrna_fold_compound_t  *fallback;
    vrna_fold_compound_t  *inputs[1];
    float                 fallback_energy;

    vrna_md_set_default(&md);
    md.compute_bpp = 1;
    md.dangles     = 0;
    fallback       = vrna_fold_compound(cases[0].sequence, &md, VRNA_OPTION_PF);
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

  printf("CUDA PF/BPP validation: device=%d modes=%zu sequences=%d cells=%zu "
         "max energy error %.3g max probability error %.3g\n",
         initial_device,
         sizeof(modes) / sizeof(modes[0]),
         case_count,
         cells,
         max_energy_error,
         max_probability_error);
  result = EXIT_SUCCESS;

cleanup:
  free_cases(cases, case_count);
  if (handle)
    dlclose(handle);
  return result;
}
