// clang-format off
/*
 * BibTeX:
 * @inproceedings{ishai2013power,
 *   title={On the power of correlated randomness in secure computation},
 *   author={Ishai, Yuval and Kushilevitz, Eyal and Meldgaard, Sigurd and Orlandi, Claudio and Paskin-Cherniavsky, Anat},
 *   booktitle={Theory of cryptography conference},
 *   pages={600--620},
 *   year={2013},
 *   organization={Springer}
 * }
 * 
 * Cite: https://link.springer.com/chapter/10.1007/978-3-642-36594-2_34
 */
// clang-format on

#ifndef FAST_FSS_OTTT_H
#define FAST_FSS_OTTT_H

#include <FastFss/api.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

FAST_FSS_API int FastFss_otttGetKeyDataSize(size_t *keyDataSize, size_t bitWidthIn, size_t elementNum);

#ifdef __cplusplus
}
#endif

#endif
