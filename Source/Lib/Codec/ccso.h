#ifndef EbCcso_h
#define EbCcso_h

#include <string.h>
#include "definitions.h"
#include "utility.h"
#include "pcs.h"
#include "coding_unit.h"
#include "sequence_control_set.h"
#include "reference_object.h"
#include "common_utils.h"

// #define CCSO_BAND_NUM 128
// #define CCSO_NUM_COMPONENTS 3
// #define CONFIG_D143_CCSO_FM_FLAG 1



#define CCSO_PADDING_SIZE 5
#define CCSO_BLK_SIZE 7
#define CCSO_INPUT_INTERVAL 3
#define CONFIG_CCSO_SIGFIX 1
#define CCSO_MAX_ITERATIONS 15

// Only need this for fixed-size arrays, for structs just assign.
#define av1_copy(dest, src)                  \
    {                                        \
        assert(sizeof(dest) == sizeof(src)); \
        memcpy(dest, src, sizeof(src));      \
    }

static const int edge_clf_to_edge_interval[2] = {3, 2};

void svt_av1_setup_dst_planes(PictureControlSet *pcs, struct MacroblockdPlane *planes, BlockSize bsize,
                              //const Yv12BufferConfig *src,
                              const EbPictureBufferDesc *src, int32_t mi_row, int32_t mi_col, const int32_t plane_start,
                              const int32_t plane_end);

void extend_ccso_border(uint16_t *buf, const int d, MacroblockdPlane *pd);

void ccso_search(PictureControlSet *pcs, MacroblockdPlane *pd, int rdmult, const uint16_t *ext_rec_y,
                 uint16_t *rec_uv[3], uint16_t *org_uv[3]);

void ccso_frame(EbPictureBufferDesc *frame, PictureControlSet *pcs, MacroblockdPlane *pd, uint16_t *ext_rec_y);

/* Apply CCSO on one color component */
typedef void (*CCSO_FILTER_FUNC)(PictureControlSet *pcs, MacroblockdPlane *pd, const int plane, const uint16_t *src_y,
                                 uint8_t *dst_yuv, const int dst_stride, const uint8_t thr, const uint8_t filter_sup,
                                 const uint8_t max_band_log2, const int edge_clf);

#endif // EbCcso_h