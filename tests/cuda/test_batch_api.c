#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <ViennaRNA/fold_compound.h>
#include <ViennaRNA/constraints/hard.h>
#include <ViennaRNA/mfe/global.h>
#include <ViennaRNA/model.h>


int
main(void)
{
  static const char *sequences[] = {
    "GCGCUUCGCC",
    "GGGAAACCCUUUGGGAAACCC",
    "AUGCUAGCUAGCUACGUAUGCUAGCUAGC",
    "GCGCGCGCAAAAUUUUGCGCGCGC",
    "ACGUACGUACGUACGUACGUACGUACGUACGU",
    "GGGGAAAACCCCUUUUGGGGAAAACCCC",
    "GCAUCGGAUCCGAUCGGAUCCGAUCGGAUC"
  };
  const size_t count = sizeof(sequences) / sizeof(sequences[0]);
  vrna_fold_compound_t *cpu[count];
  vrna_fold_compound_t *batch[count];
  char *cpu_structures[count];
  char *batch_structures[count];
  float cpu_energies[count];
  float batch_energies[count];
  int result = EXIT_FAILURE;

  memset(cpu, 0, sizeof(cpu));
  memset(batch, 0, sizeof(batch));
  memset(cpu_structures, 0, sizeof(cpu_structures));
  memset(batch_structures, 0, sizeof(batch_structures));

  for (size_t i = 0; i < count; i++) {
    const size_t length = strlen(sequences[i]);
    vrna_md_t md;

    vrna_md_set_default(&md);
    if (i == count - 1)
      md.dangles = 0;  /* deliberately unsupported: must take the exact CPU fallback */

    cpu[i]               = vrna_fold_compound(sequences[i], &md, VRNA_OPTION_MFE);
    batch[i]             = vrna_fold_compound(sequences[i], &md, VRNA_OPTION_MFE);
    cpu_structures[i]    = (char *)calloc(length + 1, sizeof(char));
    batch_structures[i]  = (char *)calloc(length + 1, sizeof(char));
    if ((!cpu[i]) || (!batch[i]) || (!cpu_structures[i]) || (!batch_structures[i]))
      goto cleanup;

    if (i == count - 2) {
      const unsigned char constraint = VRNA_CONSTRAINT_CONTEXT_ALL_LOOPS |
                                       VRNA_CONSTRAINT_CONTEXT_ENFORCE;
      vrna_hc_add_up(cpu[i], 5, constraint);
      vrna_hc_add_up(batch[i], 5, constraint);
    }

    cpu_energies[i] = vrna_mfe(cpu[i], cpu_structures[i]);
  }

  if (!vrna_mfe_batch(batch, count, batch_structures, batch_energies)) {
    fprintf(stderr, "vrna_mfe_batch returned failure\n");
    goto cleanup;
  }

  for (size_t i = 0; i < count; i++) {
    if ((cpu_energies[i] != batch_energies[i]) ||
        (strcmp(cpu_structures[i], batch_structures[i]) != 0)) {
      fprintf(stderr,
              "batch API mismatch at %zu: cpu=%0.2f %s batch=%0.2f %s\n",
              i,
              cpu_energies[i],
              cpu_structures[i],
              batch_energies[i],
              batch_structures[i]);
      goto cleanup;
    }
  }

  printf("batch API exact match: %zu inputs (including model and constraint fallbacks)\n", count);
  result = EXIT_SUCCESS;

cleanup:
  for (size_t i = 0; i < count; i++) {
    vrna_fold_compound_free(cpu[i]);
    vrna_fold_compound_free(batch[i]);
    free(cpu_structures[i]);
    free(batch_structures[i]);
  }

  return result;
}
