#ifndef VIENNA_RNA_PACKAGE_GPU_BACKEND_H
#define VIENNA_RNA_PACKAGE_GPU_BACKEND_H

#include <stddef.h>
#include <ViennaRNA/fold_compound.h>

#define VRNA_CUDA_BACKEND_ABI_VERSION 3
#define VRNA_CUDA_BACKEND_ABI_SYMBOL "vrna_cuda_backend_abi_version"
#define VRNA_CUDA_BACKEND_BATCH_SYMBOL "vrna_cuda_mfe_batch"
#define VRNA_CUDA_PF_BATCH_SYMBOL "vrna_cuda_pf_batch"

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*vrna_cuda_backend_abi_f)(void);

enum {
  VRNA_CUDA_BACKEND_COPY_MATRICES = 1U,
  VRNA_CUDA_BACKEND_TRACEBACK     = 2U
};

typedef int (*vrna_cuda_backend_batch_f)(vrna_fold_compound_t **fc,
                                         size_t               count,
                                         unsigned char        *handled,
                                         unsigned char        *traced,
                                         int                  *energies,
                                         char                 **structures,
                                         unsigned int         flags);

typedef int (*vrna_cuda_pf_batch_f)(vrna_fold_compound_t **fc,
                                    size_t               count,
                                    unsigned char        *handled,
                                    float                *energies,
                                    unsigned int         flags);

int
vrna_cuda_backend_abi_version(void);

int
vrna_cuda_mfe_batch(vrna_fold_compound_t **fc,
                    size_t               count,
                    unsigned char        *handled,
                    unsigned char        *traced,
                    int                  *energies,
                    char                 **structures,
                    unsigned int         flags);

int
vrna_cuda_pf_batch(vrna_fold_compound_t **fc,
                   size_t               count,
                   unsigned char        *handled,
                   float                *energies,
                   unsigned int         flags);

int
vrna_cuda_pf_selected_device(void);

size_t
vrna_cuda_pf_last_fallback_count(void);

#ifdef __cplusplus
}
#endif

#endif
