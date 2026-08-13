#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <ViennaRNA/datastructures/dp_matrices.h>
#include <ViennaRNA/eval/internal.h>
#include <ViennaRNA/fold_compound.h>
#include <ViennaRNA/mfe/global.h>
#include <ViennaRNA/model.h>
#include <ViennaRNA/params/basic.h>
#include <ViennaRNA/params/constants.h>
#include <ViennaRNA/sequences/alphabet.h>


typedef struct {
  unsigned long long outer_cells;
  unsigned long long bands;
  unsigned long long pruned_bands;
  unsigned long long dense_evaluations;
  unsigned long long bounded_evaluations;
  unsigned long long bound_cell_loads;
} statistics_t;


static unsigned int random_state = 0x91e10da5U;


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
  char *sequence = (char *)calloc(length + 1, sizeof(char));

  if (!sequence)
    return NULL;

  for (unsigned int i = 0; i < length; i++)
    sequence[i] = alphabet[random_u32() & 3U];

  return sequence;
}


static void
build_total_size_bounds(vrna_param_t *params,
                        int          bounds[NBPAIRS + 1][MAXLOOP + 1])
{
  for (unsigned int type = 0; type <= NBPAIRS; type++)
    for (unsigned int total = 0; total <= MAXLOOP; total++)
      bounds[type][total] = INF;

  for (unsigned int type = 1; type <= NBPAIRS; type++) {
    for (unsigned int total = 0; total <= MAXLOOP; total++) {
      int minimum = INF;

      for (unsigned int u1 = 0; u1 <= total; u1++) {
        const unsigned int u2 = total - u1;

        for (unsigned int type2 = 1; type2 <= NBPAIRS; type2++) {
          if (total == 0) {
            const int energy = vrna_E_internal(u1,
                                               u2,
                                               type,
                                               type2,
                                               0,
                                               0,
                                               0,
                                               0,
                                               params);
            if (energy < minimum)
              minimum = energy;
            continue;
          }

          for (int si1 = 0; si1 <= 4; si1++)
            for (int sj1 = 0; sj1 <= 4; sj1++)
              for (int sp1 = 0; sp1 <= 4; sp1++)
                for (int sq1 = 0; sq1 <= 4; sq1++) {
                  const int energy = vrna_E_internal(u1,
                                                     u2,
                                                     type,
                                                     type2,
                                                     si1,
                                                     sj1,
                                                     sp1,
                                                     sq1,
                                                     params);
                  if (energy < minimum)
                    minimum = energy;
                }
        }
      }

      bounds[type][total] = minimum;
    }
  }
}


static void
build_candidate_bounds(vrna_param_t *params,
                       int          bounds[(MAXLOOP + 1) * (MAXLOOP + 1)])
{
  int min_stack = INF;
  int min_bulge_stack = INF;
  int min_terminal = INF;
  int min_int11 = INF;
  int min_int21 = INF;
  int min_int22 = INF;
  int min_mismatch_i = INF;
  int min_mismatch_1n = INF;
  int min_mismatch_23 = INF;

  for (unsigned int type = 1; type <= NBPAIRS; type++) {
    for (unsigned int type2 = 1; type2 <= NBPAIRS; type2++) {
#define TAKE_MINIMUM(TARGET, VALUE) do { if ((VALUE) < (TARGET)) (TARGET) = (VALUE); } while (0)
      TAKE_MINIMUM(min_stack, params->stack[type][type2] + params->SaltStack);
      TAKE_MINIMUM(min_bulge_stack, params->stack[type][type2]);
      TAKE_MINIMUM(min_terminal,
                   ((type > 2) ? params->TerminalAU : 0) +
                   ((type2 > 2) ? params->TerminalAU : 0));

      for (unsigned int a = 0; a < 5; a++) {
        for (unsigned int b = 0; b < 5; b++) {
          TAKE_MINIMUM(min_int11, params->int11[type][type2][a][b]);
          for (unsigned int c = 0; c < 5; c++) {
            TAKE_MINIMUM(min_int21, params->int21[type][type2][a][b][c]);
            for (unsigned int d = 0; d < 5; d++)
              TAKE_MINIMUM(min_int22, params->int22[type][type2][a][b][c][d]);
          }
        }
      }
    }

    for (unsigned int a = 0; a < 5; a++) {
      for (unsigned int b = 0; b < 5; b++) {
        TAKE_MINIMUM(min_mismatch_i, params->mismatchI[type][a][b]);
        TAKE_MINIMUM(min_mismatch_1n, params->mismatch1nI[type][a][b]);
        TAKE_MINIMUM(min_mismatch_23, params->mismatch23I[type][a][b]);
      }
    }
  }
#undef TAKE_MINIMUM

  for (unsigned int n1 = 0; n1 <= MAXLOOP; n1++) {
    for (unsigned int n2 = 0; n2 <= MAXLOOP; n2++) {
      const unsigned int index = n1 * (MAXLOOP + 1) + n2;
      const unsigned int nl = (n1 > n2) ? n1 : n2;
      const unsigned int ns = (n1 > n2) ? n2 : n1;
      bounds[index] = INF;
      if (n1 + n2 > MAXLOOP)
        continue;

      if (nl == 0) {
        bounds[index] = min_stack;
      } else if (ns == 0) {
        bounds[index] = params->bulge[nl] +
                        ((nl == 1) ? min_bulge_stack : min_terminal);
      } else if ((ns == 1) && (nl == 1)) {
        bounds[index] = min_int11;
      } else if ((ns == 1) && (nl == 2)) {
        bounds[index] = min_int21;
      } else if ((ns == 2) && (nl == 2)) {
        bounds[index] = min_int22;
      } else if (ns == 1) {
        const int ninio = ((int)(nl - ns) * params->ninio[2] < 300) ?
                          (int)(nl - ns) * params->ninio[2] : 300;
        bounds[index] = params->internal_loop[nl + 1] + ninio + 2 * min_mismatch_1n;
      } else if ((ns == 2) && (nl == 3)) {
        bounds[index] = params->internal_loop[5] + params->ninio[2] +
                        2 * min_mismatch_23;
      } else {
        const int ninio = ((int)(nl - ns) * params->ninio[2] < 300) ?
                          (int)(nl - ns) * params->ninio[2] : 300;
        bounds[index] = params->internal_loop[nl + ns] + ninio + 2 * min_mismatch_i;
      }
    }
  }
}


static int
validate_candidate_bounds(vrna_param_t *params,
                          int          bounds[(MAXLOOP + 1) * (MAXLOOP + 1)])
{
  for (unsigned int n1 = 0; n1 <= MAXLOOP; n1++) {
    for (unsigned int n2 = 0; n2 + n1 <= MAXLOOP; n2++) {
      const int lower = bounds[n1 * (MAXLOOP + 1) + n2];
      for (unsigned int type = 1; type <= NBPAIRS; type++) {
        for (unsigned int type2 = 1; type2 <= NBPAIRS; type2++) {
          for (int si1 = 0; si1 <= 4; si1++) {
            for (int sj1 = 0; sj1 <= 4; sj1++) {
              for (int sp1 = 0; sp1 <= 4; sp1++) {
                for (int sq1 = 0; sq1 <= 4; sq1++) {
                  const int energy = vrna_E_internal(n1,
                                                     n2,
                                                     type,
                                                     type2,
                                                     si1,
                                                     sj1,
                                                     sp1,
                                                     sq1,
                                                     params);
                  if (lower > energy) {
                    fprintf(stderr,
                            "invalid candidate bound for (%u,%u): lower=%d energy=%d "
                            "types=%u,%u context=%d,%d,%d,%d\n",
                            n1,
                            n2,
                            lower,
                            energy,
                            type,
                            type2,
                            si1,
                            sj1,
                            sp1,
                            sq1);
                    return 0;
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  return 1;
}


static int
validate_fold(vrna_fold_compound_t *fc,
              int                  bounds[NBPAIRS + 1][MAXLOOP + 1],
              statistics_t         *statistics)
{
  const unsigned int n = fc->length;
  const short *sequence = fc->sequence_encoding;
  const unsigned int *rtype = fc->params->model_details.rtype;

  for (unsigned int span = 1; span < n; span++) {
    for (unsigned int i = 1; i + span <= n; i++) {
      const unsigned int j = i + span;
      const int ij = fc->jindx[j] + i;
      const unsigned int type = vrna_get_ptype(ij, fc->ptype);

      if (type == 0)
        continue;

      statistics->outer_cells++;
      int dense_best = INF;
      int bounded_best = INF;
      const unsigned int max_total = (span > 1) ?
                                     ((span - 2 < MAXLOOP) ? span - 2 : MAXLOOP) : 0;

      for (unsigned int total = 0; total <= max_total; total++) {
        const unsigned int inner_span = span - total - 2;
        int band_minimum = INF;
        int enclosed_minimum = INF;
        unsigned long long finite_in_band = 0;

        statistics->bands++;
        for (unsigned int u1 = 0; u1 <= total; u1++) {
          const unsigned int u2 = total - u1;
          const unsigned int p = i + u1 + 1;
          const unsigned int q = p + inner_span;

          if ((p >= q) || (q >= j))
            continue;

          statistics->bound_cell_loads++;
          const int pq = fc->jindx[q] + p;
          const int enclosed = fc->matrices->c[pq];
          const unsigned int inner_type = vrna_get_ptype(pq, fc->ptype);
          if ((enclosed >= INF) || (inner_type == 0))
            continue;

          if (enclosed < enclosed_minimum)
            enclosed_minimum = enclosed;

          const int loop = vrna_E_internal(u1,
                                           u2,
                                           type,
                                           rtype[inner_type],
                                           sequence[i + 1],
                                           sequence[j - 1],
                                           sequence[p - 1],
                                           sequence[q + 1],
                                           fc->params);
          if ((loop < INF) && (enclosed + loop < band_minimum))
            band_minimum = enclosed + loop;
          finite_in_band++;
        }

        statistics->dense_evaluations += finite_in_band;
        if (band_minimum < dense_best)
          dense_best = band_minimum;

        const int lower_bound = ((enclosed_minimum >= INF) ||
                                 (bounds[type][total] >= INF)) ?
                                INF : enclosed_minimum + bounds[type][total];
        if (lower_bound > band_minimum) {
          fprintf(stderr,
                  "invalid internal-loop bound at (%u,%u), total=%u: "
                  "lower=%d actual=%d\n",
                  i,
                  j,
                  total,
                  lower_bound,
                  band_minimum);
          return 0;
        }

        if (lower_bound >= bounded_best) {
          statistics->pruned_bands++;
        } else {
          statistics->bounded_evaluations += finite_in_band;
          if (band_minimum < bounded_best)
            bounded_best = band_minimum;
        }
      }

      if (bounded_best != dense_best) {
        fprintf(stderr,
                "bounded internal recurrence mismatch at (%u,%u): dense=%d bounded=%d\n",
                i,
                j,
                dense_best,
                bounded_best);
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
  const size_t cases = (argc > 1) ? strtoul(argv[1], NULL, 10) : 128;
  const unsigned int max_length = (argc > 2) ? strtoul(argv[2], NULL, 10) : 240;
  int bounds[7][NBPAIRS + 1][MAXLOOP + 1];
  int candidate_bounds[7][(MAXLOOP + 1) * (MAXLOOP + 1)];
  unsigned char bounds_ready[7] = { 0 };
  statistics_t statistics = { 0 };

  if ((cases == 0) || (max_length < 8))
    return EXIT_FAILURE;

  for (size_t test = 0; test < cases; test++) {
    const unsigned int variant = test % 7;
    const unsigned int length = 8 + random_u32() % (max_length - 7);
    vrna_md_t md;
    char *sequence;
    vrna_fold_compound_t *fc;

    vrna_md_set_default(&md);
    switch (variant) {
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

    sequence = random_sequence(length);
    fc = sequence ? vrna_fold_compound(sequence, &md, VRNA_OPTION_MFE) : NULL;
    if ((!fc) || (vrna_mfe(fc, NULL) >= INF / 100.)) {
      fprintf(stderr, "failed to fold internal-loop oracle case %zu\n", test);
      vrna_fold_compound_free(fc);
      free(sequence);
      return EXIT_FAILURE;
    }

    if (!bounds_ready[variant]) {
      build_total_size_bounds(fc->params, bounds[variant]);
      build_candidate_bounds(fc->params, candidate_bounds[variant]);
      if (!validate_candidate_bounds(fc->params, candidate_bounds[variant])) {
        fprintf(stderr, "candidate lower-bound proof failed for model variant %u\n", variant);
        vrna_fold_compound_free(fc);
        free(sequence);
        return EXIT_FAILURE;
      }
      bounds_ready[variant] = 1;
    }

    if (!validate_fold(fc, bounds[variant], &statistics)) {
      fprintf(stderr, "internal-loop oracle failed for case %zu, sequence=%s\n", test, sequence);
      vrna_fold_compound_free(fc);
      free(sequence);
      return EXIT_FAILURE;
    }

    vrna_fold_compound_free(fc);
    free(sequence);
  }

  const double band_reduction = statistics.bands ?
                                100. * (double)statistics.pruned_bands /
                                (double)statistics.bands : 0.;
  const double evaluation_reduction = statistics.dense_evaluations ?
                                      100. * (double)(statistics.dense_evaluations -
                                                      statistics.bounded_evaluations) /
                                      (double)statistics.dense_evaluations : 0.;
  printf("exact internal-loop bounds passed: cases=%zu candidate_shapes=%u outer=%llu bands=%llu "
         "pruned_bands=%llu band_reduction=%.3f%% dense_evaluations=%llu "
         "bounded_evaluations=%llu evaluation_reduction=%.3f%% bound_cell_loads=%llu\n",
         cases,
         (MAXLOOP + 1) * (MAXLOOP + 2) / 2,
         statistics.outer_cells,
         statistics.bands,
         statistics.pruned_bands,
         band_reduction,
         statistics.dense_evaluations,
         statistics.bounded_evaluations,
         evaluation_reduction,
         statistics.bound_cell_loads);
  return EXIT_SUCCESS;
}
