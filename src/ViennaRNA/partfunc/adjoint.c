#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <float.h>
#include <math.h>
#include <stdlib.h>

#include "ViennaRNA/constraints/basic.h"
#include "ViennaRNA/constraints/hard.h"
#include "ViennaRNA/constraints/soft.h"
#include "ViennaRNA/datastructures/dp_matrices.h"
#include "ViennaRNA/fold_compound.h"
#include "ViennaRNA/model.h"
#include "ViennaRNA/partfunc/adjoint.h"
#include "ViennaRNA/partfunc/global.h"
#include "ViennaRNA/utils/basic.h"

#define VRNA_PF_ADJOINT_ORACLE_MAX_LENGTH 60U
#define VRNA_PF_ADJOINT_PROBE_ENERGY      (-10.)


static int
can_stack_under_no_lp(const vrna_fold_compound_t *fc,
                      unsigned int               i,
                      unsigned int               j)
{
  const vrna_md_t *md;
  const short     *sequence2;

  md        = &(fc->exp_params->model_details);
  sequence2 = fc->sequence_encoding2;

  if ((i > 1U) &&
      (j < fc->length) &&
      ((j - i + 2U) < (unsigned int)md->max_bp_span) &&
      md->pair[sequence2[i - 1U]][sequence2[j + 1U]])
    return 1;

  return (i + 2U < j) &&
         ((j - i - 2U) > (unsigned int)md->min_loop_size) &&
         md->pair[sequence2[i + 1U]][sequence2[j - 1U]];
}


static int
default_hard_constraints(const vrna_fold_compound_t *fc)
{
  const vrna_hc_t *hc;
  const vrna_md_t *md;
  unsigned int    i, j, n;

  hc  = fc->hc;
  md  = &(fc->exp_params->model_details);
  n   = fc->length;

  if ((!hc) ||
      (hc->type != VRNA_HC_DEFAULT) ||
      (hc->f != NULL) ||
      (hc->data != NULL) ||
      (!hc->mx) ||
      (!hc->up_ext) ||
      (!hc->up_hp) ||
      (!hc->up_int) ||
      (!hc->up_ml))
    return 0;

  for (i = 1; i <= n; i++) {
    if ((hc->mx[(size_t)n * i + i] != VRNA_CONSTRAINT_CONTEXT_ALL_LOOPS) ||
        (hc->up_ext[i] != n) ||
        (hc->up_hp[i] != n) ||
        (hc->up_int[i] != n) ||
        (hc->up_ml[i] != n))
      return 0;

    for (j = i + 1; j <= n; j++) {
      unsigned char expected = VRNA_CONSTRAINT_CONTEXT_NONE;

      if (((j - i) < (unsigned int)md->max_bp_span) &&
          ((j - i) > (unsigned int)md->min_loop_size)) {
        const unsigned int type = md->pair[fc->sequence_encoding2[i]][fc->sequence_encoding2[j]];

        if ((type != 0) &&
            (!(((type == 3) || (type == 4)) && md->noGU))) {
          expected = VRNA_CONSTRAINT_CONTEXT_ALL_LOOPS;
          if (((type == 3) || (type == 4)) && md->noGUclosure)
            expected &= ~(VRNA_CONSTRAINT_CONTEXT_HP_LOOP |
                          VRNA_CONSTRAINT_CONTEXT_MB_LOOP);
          if (md->noLP && !can_stack_under_no_lp(fc, i, j))
            expected = VRNA_CONSTRAINT_CONTEXT_NONE;
        }
      }

      if (hc->mx[(size_t)n * i + j] != expected)
        return 0;
    }
  }

  return 1;
}


static int
eligible(const vrna_fold_compound_t *fc)
{
  const vrna_md_t *md;

  if ((!fc) ||
      (fc->type != VRNA_FC_TYPE_SINGLE) ||
      (fc->strands != 1) ||
      (fc->length == 0) ||
      (fc->length > VRNA_PF_ADJOINT_ORACLE_MAX_LENGTH) ||
      (!fc->sequence) ||
      (!fc->sequence_encoding2) ||
      (!fc->exp_params) ||
      (!fc->exp_matrices) ||
      (fc->exp_matrices->type != VRNA_MX_DEFAULT) ||
      (!fc->exp_matrices->q) ||
      (!fc->exp_matrices->qb) ||
      (fc->sc != NULL) ||
      (fc->domains_up != NULL) ||
      (fc->aux_grammar != NULL) ||
      (fc->stat_cb != NULL))
    return 0;

  md = &(fc->exp_params->model_details);
  if ((md->dangles != 2) ||
      md->logML ||
      md->circ ||
      md->gquad ||
      md->uniq_ML ||
      (md->backtrack_type != VRNA_MODEL_DEFAULT_BACKTRACK_TYPE) ||
      (md->salt != VRNA_MODEL_DEFAULT_SALT) ||
      (fc->exp_params->param_file[0] != '\0'))
    return 0;

  return default_hard_constraints(fc);
}


PUBLIC FLT_OR_DBL *
vrna_pf_adjoint_oracle(vrna_fold_compound_t *fc)
{
  FLT_OR_DBL   *adjoint, *qb, root;
  unsigned int i, j, n;
  size_t       matrix_size;
  const long double tolerance = 128.L *
                                ((sizeof(FLT_OR_DBL) == sizeof(float)) ?
                                 FLT_EPSILON : DBL_EPSILON);

  if (!eligible(fc))
    return NULL;

  n           = fc->length;
  matrix_size = ((size_t)n + 1) * ((size_t)n + 2) / 2;
  qb          = fc->exp_matrices->qb;
  root        = fc->exp_matrices->q[fc->iindx[1] - n];

  if ((!isfinite((double)root)) || (root <= 0.))
    return NULL;

  adjoint = (FLT_OR_DBL *)vrna_alloc(sizeof(FLT_OR_DBL) * matrix_size);

  for (i = 1; i < n; i++) {
    for (j = i + 1; j <= n; j++) {
      vrna_fold_compound_t *probe;
      vrna_md_t            probe_md;
      FLT_OR_DBL           paired_inside, probe_root, probe_weight;
      long double          probability;
      const int            ij = fc->iindx[i] - j;

      paired_inside = qb[ij];
      if (paired_inside <= 0.)
        continue;

      probe_md             = fc->exp_params->model_details;
      probe_md.compute_bpp = 0;
      probe                = vrna_fold_compound(fc->sequence,
                                                &probe_md,
                                                VRNA_OPTION_PF);
      if ((!probe) ||
          (!vrna_sc_add_bp(probe,
                           i,
                           j,
                           VRNA_PF_ADJOINT_PROBE_ENERGY,
                           VRNA_OPTION_PF))) {
        vrna_fold_compound_free(probe);
        free(adjoint);
        return NULL;
      }

      (void)vrna_pf(probe, NULL);
      probe_root   = probe->exp_matrices->q[probe->iindx[1] - n];
      probe_weight = probe->sc->exp_energy_bp[probe->jindx[j] + i];

      if ((!isfinite((double)probe_root)) ||
          (!isfinite((double)probe_weight)) ||
          (probe_root <= 0.) ||
          (probe_weight <= 1.)) {
        vrna_fold_compound_free(probe);
        free(adjoint);
        return NULL;
      }

      probability = (((long double)probe_root / (long double)root) - 1.L) /
                    ((long double)probe_weight - 1.L);

      /* Roundoff may erase probabilities far below machine precision. */
      if ((probability < 0.L) && (probability > -tolerance))
        probability = 0.L;

      if ((probability < 0.L) || (probability > 1.L + tolerance)) {
        vrna_fold_compound_free(probe);
        free(adjoint);
        return NULL;
      }

      if (probability > 1.L)
        probability = 1.L;

      adjoint[ij] = (FLT_OR_DBL)(probability / (long double)paired_inside);
      vrna_fold_compound_free(probe);
    }
  }

  return adjoint;
}
