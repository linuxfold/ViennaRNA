#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <ViennaRNA/backtrack/global.h>
#include <ViennaRNA/constraints/hard.h>
#include <ViennaRNA/datastructures/dp_matrices.h>
#include <ViennaRNA/fold_compound.h>
#include <ViennaRNA/gpu/backend.h>
#include <ViennaRNA/mfe/global.h>
#include <ViennaRNA/model.h>


static unsigned int random_state = 0x6d2b79f5U;


static unsigned int
random_u32(void)
{
  random_state ^= random_state << 13;
  random_state ^= random_state >> 17;
  random_state ^= random_state << 5;
  return random_state;
}


static char *
random_sequence(unsigned int length)
{
  static const char alphabet[] = "ACGU";
  char              *sequence = (char *)calloc(length + 1, sizeof(char));

  if (!sequence)
    return NULL;

  for (unsigned int i = 0; i < length; i++)
    sequence[i] = alphabet[random_u32() & 3U];

  return sequence;
}


static int
compare_matrices(vrna_fold_compound_t *cpu,
                 vrna_fold_compound_t *gpu)
{
  const unsigned int n = cpu->length;

  for (unsigned int j = 1; j <= n; j++) {
    if (cpu->matrices->f5[j] != gpu->matrices->f5[j]) {
      fprintf(stderr,
              "f5 mismatch at j=%u: cpu=%d gpu=%d\n",
              j,
              cpu->matrices->f5[j],
              gpu->matrices->f5[j]);
      return 0;
    }

    for (unsigned int i = 1; i <= j; i++) {
      const int ij = cpu->jindx[j] + i;
      if (cpu->matrices->c[ij] != gpu->matrices->c[ij]) {
        fprintf(stderr,
                "c mismatch at (%u,%u): cpu=%d gpu=%d\n",
                i,
                j,
                cpu->matrices->c[ij],
                gpu->matrices->c[ij]);
        return 0;
      }

      if (cpu->matrices->fML[ij] != gpu->matrices->fML[ij]) {
        fprintf(stderr,
                "fML mismatch at (%u,%u): cpu=%d gpu=%d\n",
                i,
                j,
                cpu->matrices->fML[ij],
                gpu->matrices->fML[ij]);
        return 0;
      }
    }
  }

  return 1;
}


int
main(int argc,
     char **argv)
{
  const size_t count = (argc > 1) ? strtoul(argv[1], NULL, 10) : 128;
  const unsigned int max_length = (argc > 2) ? strtoul(argv[2], NULL, 10) : 240;
  vrna_fold_compound_t **cpu = (vrna_fold_compound_t **)calloc(count, sizeof(*cpu));
  vrna_fold_compound_t **gpu = (vrna_fold_compound_t **)calloc(count, sizeof(*gpu));
  char                  **sequences = (char **)calloc(count, sizeof(*sequences));
  char                  **cpu_structures = (char **)calloc(count, sizeof(*cpu_structures));
  char                  **gpu_structures = (char **)calloc(count, sizeof(*gpu_structures));
  float                 *cpu_energies = (float *)calloc(count, sizeof(*cpu_energies));
  int                   *gpu_energies = (int *)calloc(count, sizeof(*gpu_energies));
  unsigned char         *handled = (unsigned char *)calloc(count, sizeof(*handled));
  unsigned char         *traced = (unsigned char *)calloc(count, sizeof(*traced));
  int                   result = EXIT_FAILURE;

  if ((!cpu) || (!gpu) || (!sequences) || (!cpu_structures) ||
      (!gpu_structures) || (!cpu_energies) || (!gpu_energies) || (!handled) || (!traced))
    goto cleanup;

  for (size_t b = 0; b < count; b++) {
    unsigned int length = 8 + random_u32() % (max_length - 7);
    vrna_md_t md;

    vrna_md_set_default(&md);

    switch (b % 7) {
      case 1:
        md.special_hp = 0;
        break;
      case 2:
        md.noGU = 1;
        break;
      case 3:
        md.noGUclosure = 1;
        break;
      case 4:
        md.min_loop_size = 0;
        break;
      case 5:
        md.max_bp_span = 48;
        break;
      case 6:
        md.temperature = 25.;
        break;
      default:
        break;
    }

    /* Repeated lengths exercise batch-fastest buckets; varied lengths exercise bucketing. */
    if (b < count / 2)
      length = max_length;

    sequences[b] = random_sequence(length);
    if (!sequences[b])
      goto cleanup;

    cpu[b] = vrna_fold_compound(sequences[b], &md, VRNA_OPTION_MFE);
    gpu[b] = vrna_fold_compound(sequences[b], &md, VRNA_OPTION_MFE);
    cpu_structures[b] = (char *)calloc(length + 1, sizeof(char));
    gpu_structures[b] = (char *)calloc(length + 1, sizeof(char));
    if ((!cpu[b]) || (!gpu[b]) || (!cpu_structures[b]) || (!gpu_structures[b]))
      goto cleanup;

    cpu_energies[b] = vrna_mfe(cpu[b], cpu_structures[b]);
    if (!vrna_fold_compound_prepare(gpu[b], VRNA_OPTION_MFE))
      goto cleanup;
  }

  memset(handled, 0, count * sizeof(*handled));
  memset(traced, 0, count * sizeof(*traced));
  for (size_t b = 0; b < count; b++)
    memset(gpu_structures[b], 0, gpu[b]->length + 1);

  if (!vrna_cuda_mfe_batch(gpu,
                           count,
                           handled,
                           traced,
                           gpu_energies,
                           gpu_structures,
                           VRNA_CUDA_BACKEND_TRACEBACK)) {
    fprintf(stderr, "CUDA traceback backend returned failure\n");
    goto cleanup;
  }

  for (size_t b = 0; b < count; b++) {
    if ((!handled[b]) || (!traced[b])) {
      fprintf(stderr,
              "input %zu was not traced on the GPU (length=%u)\n",
              b,
              gpu[b]->length);
      goto cleanup;
    }

    if ((gpu_energies[b] != cpu[b]->matrices->f5[cpu[b]->length]) ||
        (strcmp(cpu_structures[b], gpu_structures[b]) != 0)) {
      fprintf(stderr,
              "GPU traceback mismatch for input %zu:\ncpu %s\ngpu %s\n",
              b,
              cpu_structures[b],
              gpu_structures[b]);
      goto cleanup;
    }
  }

  if (!vrna_cuda_mfe_batch(gpu,
                           count,
                           handled,
                           NULL,
                           gpu_energies,
                           NULL,
                           VRNA_CUDA_BACKEND_COPY_MATRICES)) {
    fprintf(stderr, "CUDA backend returned failure\n");
    goto cleanup;
  }

  for (size_t b = 0; b < count; b++) {
    if (!handled[b]) {
      fprintf(stderr,
              "input %zu was not handled (length=%u param_file='%s')\n",
              b,
              gpu[b]->length,
              gpu[b]->params->param_file);
      goto cleanup;
    }

    if (gpu_energies[b] != cpu[b]->matrices->f5[cpu[b]->length]) {
      fprintf(stderr,
              "energy mismatch for input %zu: cpu=%d gpu=%d\n",
              b,
              cpu[b]->matrices->f5[cpu[b]->length],
              gpu_energies[b]);
      goto cleanup;
    }

    if (!compare_matrices(cpu[b], gpu[b])) {
      fprintf(stderr, "matrix comparison failed for input %zu (length=%u)\n", b, gpu[b]->length);
      goto cleanup;
    }

    (void)vrna_backtrack5(gpu[b], gpu[b]->length, gpu_structures[b]);
    if (strcmp(cpu_structures[b], gpu_structures[b]) != 0) {
      fprintf(stderr,
              "structure mismatch for input %zu:\ncpu %s\ngpu %s\n",
              b,
              cpu_structures[b],
              gpu_structures[b]);
      goto cleanup;
    }
  }

  {
    vrna_fold_compound_t *constrained = vrna_fold_compound("GGGAAACCCUUUGGGAAACCC",
                                                            NULL,
                                                            VRNA_OPTION_MFE);
    unsigned char constrained_handled = 0;
    int constrained_energy = 0;
    const unsigned char constraint = VRNA_CONSTRAINT_CONTEXT_ALL_LOOPS |
                                     VRNA_CONSTRAINT_CONTEXT_ENFORCE;

    if (constrained)
      vrna_hc_add_up(constrained, 5, constraint);

    if ((!constrained) ||
        (!vrna_fold_compound_prepare(constrained, VRNA_OPTION_MFE)) ||
        (!vrna_cuda_mfe_batch(&constrained,
                              1,
                              &constrained_handled,
                              NULL,
                              &constrained_energy,
                              NULL,
                              VRNA_CUDA_BACKEND_COPY_MATRICES)) ||
        constrained_handled) {
      fprintf(stderr, "hard-constrained input was not rejected by CUDA backend\n");
      vrna_fold_compound_free(constrained);
      goto cleanup;
    }

    vrna_fold_compound_free(constrained);
  }

  {
    vrna_md_t md;
    vrna_fold_compound_t *overflow;
    unsigned char overflow_handled = 0;
    int overflow_energy = 0;

    vrna_md_set_default(&md);
    md.special_hp = 0;
    overflow = vrna_fold_compound("GAAAC", &md, VRNA_OPTION_MFE);
    if (overflow)
      overflow->params->hairpin[3] = 50000;

    if ((!overflow) ||
        (!vrna_fold_compound_prepare(overflow, VRNA_OPTION_MFE)) ||
        (!vrna_cuda_mfe_batch(&overflow,
                              1,
                              &overflow_handled,
                              NULL,
                              &overflow_energy,
                              NULL,
                              VRNA_CUDA_BACKEND_COPY_MATRICES)) ||
        overflow_handled) {
      fprintf(stderr, "compact-energy overflow did not request CPU fallback\n");
      vrna_fold_compound_free(overflow);
      goto cleanup;
    }

    vrna_fold_compound_free(overflow);
  }

  printf("exact CUDA match: %zu sequences, lengths 8..%u\n", count, max_length);
  result = EXIT_SUCCESS;

cleanup:
  for (size_t b = 0; b < count; b++) {
    vrna_fold_compound_free(cpu ? cpu[b] : NULL);
    vrna_fold_compound_free(gpu ? gpu[b] : NULL);
    free(sequences ? sequences[b] : NULL);
    free(cpu_structures ? cpu_structures[b] : NULL);
    free(gpu_structures ? gpu_structures[b] : NULL);
  }

  free(cpu);
  free(gpu);
  free(sequences);
  free(cpu_structures);
  free(gpu_structures);
  free(cpu_energies);
  free(gpu_energies);
  free(handled);
  free(traced);

  return result;
}
