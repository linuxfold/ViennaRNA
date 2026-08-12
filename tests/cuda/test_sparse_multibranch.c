#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <ViennaRNA/datastructures/dp_matrices.h>
#include <ViennaRNA/eval/multibranch.h>
#include <ViennaRNA/fold_compound.h>
#include <ViennaRNA/mfe/global.h>
#include <ViennaRNA/model.h>
#include <ViennaRNA/params/constants.h>


typedef struct {
  uint64_t cases;
  uint64_t cells;
  uint64_t candidates;
  uint64_t finite_branches;
  uint64_t columns;
  uint64_t column_candidates;
  unsigned int max_column_candidates;
} sparse_stats_t;


static uint32_t random_state = 0x6d2b79f5U;


static uint32_t
random_u32(void)
{
  random_state ^= random_state << 13;
  random_state ^= random_state >> 17;
  random_state ^= random_state << 5;
  return random_state;
}


static int
minimum(int a,
        int b)
{
  return (a < b) ? a : b;
}


static int
add_finite(int a,
           int b)
{
  if ((a >= INF) || (b >= INF))
    return INF;

  return a + b;
}


static size_t
square_index(unsigned int pitch,
             unsigned int i,
             unsigned int j)
{
  return (size_t)i * pitch + j;
}


static int
check_sparse_multibranch(const char        *sequence,
                         const vrna_md_t   *model,
                         sparse_stats_t    *stats)
{
  vrna_fold_compound_t *fc = NULL;
  char                 *structure = NULL;
  int                  *dense_m2 = NULL;
  int                  *sparse_m2 = NULL;
  int                  *sparse_m = NULL;
  unsigned int         *candidates = NULL;
  unsigned int         *candidate_count = NULL;
  int                  ok = 0;

  fc = vrna_fold_compound(sequence, model, VRNA_OPTION_MFE);
  if (!fc) {
    fprintf(stderr, "failed to create fold compound for '%s'\n", sequence);
    goto cleanup;
  }

  structure = (char *)calloc(fc->length + 1, sizeof(*structure));
  if (!structure)
    goto cleanup;

  (void)vrna_mfe(fc, structure);

  const unsigned int n = fc->length;
  const unsigned int pitch = n + 2;
  const size_t square = (size_t)pitch * pitch;

  dense_m2       = (int *)malloc(square * sizeof(*dense_m2));
  sparse_m2      = (int *)malloc(square * sizeof(*sparse_m2));
  sparse_m       = (int *)malloc(square * sizeof(*sparse_m));
  candidates     = (unsigned int *)calloc(square, sizeof(*candidates));
  candidate_count = (unsigned int *)calloc(pitch, sizeof(*candidate_count));
  if ((!dense_m2) || (!sparse_m2) || (!sparse_m) ||
      (!candidates) || (!candidate_count))
    goto cleanup;

  for (size_t cell = 0; cell < square; cell++) {
    dense_m2[cell]  = INF;
    sparse_m2[cell] = INF;
    sparse_m[cell]  = INF;
  }

  for (unsigned int span = 1; span < n; span++) {
    for (unsigned int i = 1; i + span <= n; i++) {
      const unsigned int j = i + span;
      const size_t ij = square_index(pitch, i, j);
      int dense_split = INF;
      int sparse_split = INF;

      /* Dense ViennaRNA M2 recurrence, reconstructed from its exact fML matrix. */
      for (unsigned int k = i + 1; k + 1 < j; k++) {
        const int left = fc->matrices->fML[fc->jindx[k] + i];
        const int right = fc->matrices->fML[fc->jindx[j] + k + 1];
        dense_split = minimum(dense_split, add_finite(left, right));
      }
      dense_m2[ij] = dense_split;

      /* A right-extension carries all non-candidate suffixes from column j - 1. */
      if (span > 1)
        sparse_split = add_finite(sparse_m2[square_index(pitch, i, j - 1)],
                                  fc->params->MLbase);

      /* Only strict candidates ending at j are explicit split points. */
      for (unsigned int entry = 0; entry < candidate_count[j]; entry++) {
        const unsigned int a = candidates[square_index(pitch, j, entry)];
        if (a < i + 2)
          continue;

        sparse_split = minimum(sparse_split,
                               add_finite(sparse_m[square_index(pitch, i, a - 1)],
                                          sparse_m[square_index(pitch, a, j)]));
      }
      sparse_m2[ij] = sparse_split;

      int branch = INF;
      const int paired = fc->matrices->c[fc->jindx[j] + i];
      const short *s = fc->sequence_encoding;
      const short *s2 = fc->sequence_encoding2;
      unsigned int type = fc->params->model_details.pair[s2[i]][s2[j]];
      if (fc->params->model_details.noGU && ((type == 3) || (type == 4)))
        type = 0;

      if ((type != 0) &&
          (span > (unsigned int)fc->params->model_details.min_loop_size) &&
          (span < (unsigned int)fc->params->model_details.max_bp_span) &&
          (paired < INF)) {
        const int stem = vrna_E_multibranch_stem(type,
                                                  (i == 1) ? s[n] : s[i - 1],
                                                  s[j + 1],
                                                  fc->params);
        branch = add_finite(paired, stem);
        stats->finite_branches++;
      }

      int nonclosed = sparse_split;
      if (span > 1) {
        nonclosed = minimum(nonclosed,
                            add_finite(sparse_m[square_index(pitch, i, j - 1)],
                                       fc->params->MLbase));
        nonclosed = minimum(nonclosed,
                            add_finite(sparse_m[square_index(pitch, i + 1, j)],
                                       fc->params->MLbase));
      }

      sparse_m[ij] = minimum(branch, nonclosed);
      const int cpu_m = fc->matrices->fML[fc->jindx[j] + i];

      if (sparse_split != dense_split) {
        fprintf(stderr,
                "M2 mismatch for '%s' at (%u,%u): dense=%d sparse=%d "
                "candidates_in_column=%u\n",
                sequence,
                i,
                j,
                dense_split,
                sparse_split,
                candidate_count[j]);
        goto cleanup;
      }

      if (sparse_m[ij] != cpu_m) {
        fprintf(stderr,
                "fML mismatch for '%s' at (%u,%u): cpu=%d sparse=%d "
                "branch=%d nonclosed=%d m2=%d\n",
                sequence,
                i,
                j,
                cpu_m,
                sparse_m[ij],
                branch,
                nonclosed,
                sparse_split);
        goto cleanup;
      }

      stats->cells++;
      if (branch < nonclosed) {
        candidates[square_index(pitch, j, candidate_count[j])] = i;
        candidate_count[j]++;
        stats->candidates++;
      }
    }
  }

  for (unsigned int j = 1; j <= n; j++) {
    stats->columns++;
    stats->column_candidates += candidate_count[j];
    if (candidate_count[j] > stats->max_column_candidates)
      stats->max_column_candidates = candidate_count[j];
  }

  stats->cases++;
  ok = 1;

cleanup:
  free(candidate_count);
  free(candidates);
  free(sparse_m);
  free(sparse_m2);
  free(dense_m2);
  free(structure);
  vrna_fold_compound_free(fc);
  return ok;
}


static void
set_model_variant(vrna_md_t *model,
                  uint64_t  variant,
                  unsigned int length)
{
  vrna_md_set_default(model);

  switch (variant % 7) {
    case 1:
      model->special_hp = 0;
      break;
    case 2:
      model->noGU = 1;
      break;
    case 3:
      model->noGUclosure = 1;
      break;
    case 4:
      model->min_loop_size = 0;
      break;
    case 5:
      model->max_bp_span = (length > 24) ? 24 : (int)length;
      break;
    case 6:
      model->temperature = 25.;
      break;
    default:
      break;
  }
}


static int
run_exhaustive(unsigned int  max_length,
               sparse_stats_t *stats)
{
  static const char alphabet[] = "ACGU";

  if (max_length > 10) {
    fprintf(stderr, "--exhaustive is limited to length 10\n");
    return 0;
  }

  /* ViennaRNA does not initialize the MFE DP matrices for very short inputs. */
  for (unsigned int length = 8; length <= max_length; length++) {
    const uint64_t count = UINT64_C(1) << (2 * length);
    char *sequence = (char *)calloc(length + 1, sizeof(*sequence));
    if (!sequence)
      return 0;

    for (uint64_t code = 0; code < count; code++) {
      uint64_t value = code;
      vrna_md_t model;

      for (unsigned int pos = 0; pos < length; pos++) {
        sequence[length - pos - 1] = alphabet[value & 3U];
        value >>= 2;
      }

      set_model_variant(&model, 0, length);
      if (!check_sparse_multibranch(sequence, &model, stats)) {
        free(sequence);
        return 0;
      }
    }

    free(sequence);
  }

  return 1;
}


static int
run_random(uint64_t       count,
           unsigned int   max_length,
           sparse_stats_t *stats)
{
  static const char alphabet[] = "ACGU";

  if (max_length < 8)
    max_length = 8;

  for (uint64_t test = 0; test < count; test++) {
    const unsigned int length = 8 + random_u32() % (max_length - 7);
    char *sequence = (char *)calloc(length + 1, sizeof(*sequence));
    vrna_md_t model;

    if (!sequence)
      return 0;

    for (unsigned int pos = 0; pos < length; pos++)
      sequence[pos] = alphabet[random_u32() & 3U];

    set_model_variant(&model, test, length);
    if (!check_sparse_multibranch(sequence, &model, stats)) {
      free(sequence);
      return 0;
    }

    free(sequence);
  }

  return 1;
}


static int
run_adversarial(sparse_stats_t *stats)
{
  static const char *sequences[] = {
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC",
    "GGGGGGGGGGGGGGGGCCCCCCCCCCCCCCCCGGGGGGGGGGGGGGGGCCCCCCCCCCCCCCCC",
    "AUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAU",
    "GGGAAACCCGGGAAACCCGGGAAACCCGGGAAACCCGGGAAACCCGGGAAACCCGGGAAACCC",
    "GUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGU"
  };

  for (size_t test = 0; test < sizeof(sequences) / sizeof(sequences[0]); test++) {
    vrna_md_t model;
    const unsigned int length = (unsigned int)strlen(sequences[test]);

    set_model_variant(&model, test, length);
    if (!check_sparse_multibranch(sequences[test], &model, stats))
      return 0;
  }

  return 1;
}


static int
parse_unsigned(const char  *option,
               const char  *value,
               uint64_t    *result)
{
  char *end = NULL;
  const unsigned long long parsed = strtoull(value, &end, 10);

  if ((!value[0]) || (!end) || (*end != '\0')) {
    fprintf(stderr, "invalid value for %s: '%s'\n", option, value);
    return 0;
  }

  *result = (uint64_t)parsed;
  return 1;
}


int
main(int argc,
     char **argv)
{
  uint64_t exhaustive_length = 8;
  uint64_t random_count = 256;
  uint64_t max_length = 256;
  sparse_stats_t stats = { 0 };

  for (int arg = 1; arg < argc; arg++) {
    uint64_t *target = NULL;

    if (strcmp(argv[arg], "--exhaustive") == 0)
      target = &exhaustive_length;
    else if (strcmp(argv[arg], "--random") == 0)
      target = &random_count;
    else if (strcmp(argv[arg], "--max-length") == 0)
      target = &max_length;
    else {
      fprintf(stderr,
              "usage: %s [--exhaustive N] [--random N] [--max-length N]\n",
              argv[0]);
      return EXIT_FAILURE;
    }

    if ((++arg >= argc) || (!parse_unsigned(argv[arg - 1], argv[arg], target)))
      return EXIT_FAILURE;
  }

  if ((exhaustive_length > 10) || (max_length > UINT32_MAX)) {
    fprintf(stderr, "test bounds are too large\n");
    return EXIT_FAILURE;
  }

  if ((!run_exhaustive((unsigned int)exhaustive_length, &stats)) ||
      (!run_random(random_count, (unsigned int)max_length, &stats)) ||
      (!run_adversarial(&stats)))
    return EXIT_FAILURE;

  const double cell_density = stats.cells ?
                              (100. * (double)stats.candidates / (double)stats.cells) : 0.;
  const double branch_density = stats.finite_branches ?
                                (100. * (double)stats.candidates /
                                 (double)stats.finite_branches) : 0.;
  const double candidates_per_column = stats.columns ?
                                       ((double)stats.column_candidates /
                                        (double)stats.columns) : 0.;

  printf("sparse multibranch reference passed: cases=%" PRIu64
         " cells=%" PRIu64 " candidates=%" PRIu64
         " candidate_cell_density=%.3f%% candidate_branch_density=%.3f%%"
         " mean_candidates_per_column=%.3f max_candidates_per_column=%u\n",
         stats.cases,
         stats.cells,
         stats.candidates,
         cell_density,
         branch_density,
         candidates_per_column,
         stats.max_column_candidates);

  return EXIT_SUCCESS;
}
