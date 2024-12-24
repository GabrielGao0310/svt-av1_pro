#include <float.h>

#include "ccso.h"
#define RDCOST_DBL_WITH_NATIVE_BD_DIST1(RM, R, D, BD) RDCOST_DBL((RM), (R), (double)((D) >> (2 * (BD - 8))))

uint8_t final_band_log2;
int8_t  best_filter_offset[CCSO_NUM_COMPONENTS][CCSO_BAND_NUM * 16]  = {{0}};
int8_t  final_filter_offset[CCSO_NUM_COMPONENTS][CCSO_BAND_NUM * 16] = {{0}};
bool    best_filter_enabled[CCSO_NUM_COMPONENTS];
bool    final_filter_enabled[CCSO_NUM_COMPONENTS];
uint8_t final_ext_filter_support[CCSO_NUM_COMPONENTS];
uint8_t final_quant_idx[CCSO_NUM_COMPONENTS];
uint8_t final_ccso_bo_only[CCSO_NUM_COMPONENTS];

int  chroma_error[CCSO_BAND_NUM * 16] = {0};
int  chroma_count[CCSO_BAND_NUM * 16] = {0};
int *total_class_err[CCSO_INPUT_INTERVAL][CCSO_INPUT_INTERVAL][CCSO_BAND_NUM];
int *total_class_cnt[CCSO_INPUT_INTERVAL][CCSO_INPUT_INTERVAL][CCSO_BAND_NUM];
int *total_class_err_bo[CCSO_BAND_NUM];
int *total_class_cnt_bo[CCSO_BAND_NUM];

int      ccso_stride;
int      ccso_stride_ext;
uint64_t unfiltered_dist_frame = 0;
uint64_t filtered_dist_frame   = 0;

uint64_t *unfiltered_dist_block;
uint64_t *training_dist_block;
bool     *filter_control;
bool     *best_filter_control;
bool     *final_filter_control;

const int     ccso_offset[8] = {-10, -7, -3, -1, 0, 1, 3, 7};
const uint8_t quant_sz[4]    = {16, 8, 32, 64};

void svt_aom_get_recon_pic(PictureControlSet *pcs, EbPictureBufferDesc **recon_ptr, Bool is_highbd);


/* Pad the border of a frame */
void extend_ccso_border(uint16_t *buf, const int d, MacroblockdPlane *pd) {
    int       s = pd[0].dst.width + (CCSO_PADDING_SIZE << 1);
    uint16_t *p = &buf[d * s + d];
    int       h = pd[0].dst.height;
    int       w = pd[0].dst.width;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < d; x++) {
            *(p - d + x) = p[0];
            p[w + x]     = p[w - 1];
        }
        p += s;
    }
    p -= (s + d);
    for (int y = 0; y < d; y++) { memcpy(p + (y + 1) * s, p, sizeof(uint16_t) * (w + (d << 1))); }
    p -= ((h - 1) * s);
    for (int y = 0; y < d; y++) { memcpy(p - (y + 1) * s, p, sizeof(uint16_t) * (w + (d << 1))); }
}

/* Derive the quantized index, later it can be used for retriving offset values
* from the look-up table */
void cal_filter_support(int *rec_luma_idx, const uint16_t *rec_y, const uint8_t quant_step_size,
                       const int inv_quant_step, const int *rec_idx, const int edge_clf) {
   if (edge_clf == 0) {
       for (int i = 0; i < 2; i++) {
           int d = rec_y[rec_idx[i]] - rec_y[0];
           if (d > quant_step_size)
               rec_luma_idx[i] = 2;
           else if (d < inv_quant_step)
               rec_luma_idx[i] = 0;
           else
               rec_luma_idx[i] = 1;
       }
   } else { // if (edge_clf == 1)
       for (int i = 0; i < 2; i++) {
           int d = rec_y[rec_idx[i]] - rec_y[0];
           if (d < inv_quant_step)
               rec_luma_idx[i] = 0;
           else
               rec_luma_idx[i] = 1;
       }
   }
}

void ccso_derive_src_block(const uint16_t *src_y, uint8_t *const src_cls0, uint8_t *const src_cls1,
                          const int src_y_stride, const int src_cls_stride, const int x, const int y,
                          const int pic_width, const int pic_height, const int y_uv_hscale, const int y_uv_vscale,
                          const int qstep, const int neg_qstep, const int *src_loc, const int blk_size,
                          const int edge_clf) {
   int       src_cls[2];
   const int y_end = AOMMIN(pic_height - y, blk_size);
   const int x_end = AOMMIN(pic_width - x, blk_size);
   for (int y_start = 0; y_start < y_end; y_start++) {
       const int y_pos = y_start;
       for (int x_start = 0; x_start < x_end; x_start++) {
           const int x_pos = x + x_start;
           cal_filter_support(src_cls,
                              &src_y[(y_pos << y_uv_vscale) * src_y_stride + (x_pos << y_uv_hscale)],
                              qstep,
                              neg_qstep,
                              src_loc,
                              edge_clf);
           src_cls0[(y_pos << y_uv_vscale) * src_cls_stride + (x_pos << y_uv_hscale)] = src_cls[0];
           src_cls1[(y_pos << y_uv_vscale) * src_cls_stride + (x_pos << y_uv_hscale)] = src_cls[1];
       }
   }
}

uint64_t compute_distortion_block(const uint16_t *org, const int org_stride, const uint16_t *rec16,
                                 const int rec_stride, const int x, const int y, const int log2_filter_unit_size,
                                 const int height, const int width) {
   int      err;
   uint64_t ssd = 0;
   int      y_offset;
   int      x_offset;
   if (y + (1 << log2_filter_unit_size) >= height)
       y_offset = height - y;
   else
       y_offset = (1 << log2_filter_unit_size);

   if (x + (1 << log2_filter_unit_size) >= width)
       x_offset = width - x;
   else
       x_offset = (1 << log2_filter_unit_size);

   for (int y_off = 0; y_off < y_offset; y_off++) {
       for (int x_off = 0; x_off < x_offset; x_off++) {
           err = org[org_stride * y_off + x + x_off] - rec16[rec_stride * y_off + x + x_off];
           ssd += err * err;
       }
   }
   return ssd;
}

/* Compute SSE */
static void compute_distortion(const uint16_t *org, const int org_stride, const uint16_t *rec16, const int rec_stride,
                              const int log2_filter_unit_size, const int height, const int width,
                              uint64_t *distortion_buf, const int distortion_buf_stride, uint64_t *total_distortion) {
   for (int y = 0; y < height; y += (1 << log2_filter_unit_size)) {
       for (int x = 0; x < width; x += (1 << log2_filter_unit_size)) {
           const uint64_t ssd = compute_distortion_block(
               org, org_stride, rec16, rec_stride, x, y, log2_filter_unit_size, height, width);
           distortion_buf[(y >> log2_filter_unit_size) * distortion_buf_stride + (x >> log2_filter_unit_size)] = ssd;
           *total_distortion += ssd;
       }
       org += (org_stride << log2_filter_unit_size);
       rec16 += (rec_stride << log2_filter_unit_size);
   }
}

/* Derive block level on/off for CCSO */
void derive_blk_md(Av1Common *cm, MacroblockdPlane *pd, const int plane, const uint64_t *unfiltered_dist,
                  const uint64_t *training_dist, bool *m_filter_control, uint64_t *cur_total_dist, int *cur_total_rate,
                  bool *filter_enable) {
   AomCdfProb              ccso_cdf[CDF_SIZE(2)];
   static const AomCdfProb default_ccso_cdf[CDF_SIZE(2)] = {AOM_CDF2(11570)};
   av1_copy(ccso_cdf, default_ccso_cdf);
   const int log2_filter_unit_size = plane > 0 ? CCSO_BLK_SIZE : CCSO_BLK_SIZE + 1;
   // const CommonModeInfoParams *const mi_params = &cm->mi_params;
   const int ccso_nvfb = ((cm->mi_rows >> pd[plane].subsampling_y) + (1 << log2_filter_unit_size >> 2) - 1) /
       (1 << log2_filter_unit_size >> 2);
   const int ccso_nhfb = ((cm->mi_cols >> pd[plane].subsampling_x) + (1 << log2_filter_unit_size >> 2) - 1) /
       (1 << log2_filter_unit_size >> 2);
   bool cur_filter_enabled = false;
   int  sb_idx             = 0;
   for (int y_sb = 0; y_sb < ccso_nvfb; y_sb++) {
       for (int x_sb = 0; x_sb < ccso_nhfb; x_sb++) {
           uint64_t ssd;
           uint64_t best_ssd                = UINT64_MAX;
           int      best_rate               = INT_MAX;
           uint8_t  cur_best_filter_control = 0;
           int      cost_from_cdf[2];
           svt_aom_get_syntax_rate_from_cdf(cost_from_cdf, ccso_cdf, NULL);
           // svt_av1_cost_tokens_from_cdf(cost_from_cdf, ccso_cdf, NULL);
           for (int cur_filter_control = 0; cur_filter_control < 2; cur_filter_control++) {
               if (!(*filter_enable)) {
                   continue;
               }
               if (cur_filter_control == 0) {
                   ssd = unfiltered_dist[sb_idx];
               } else {
                   ssd = training_dist[sb_idx];
               }
               if (ssd < best_ssd) {
                   best_rate                = cost_from_cdf[cur_filter_control];
                   best_ssd                 = ssd;
                   cur_best_filter_control  = cur_filter_control;
                   m_filter_control[sb_idx] = cur_filter_control;
               }
           }
           update_cdf(ccso_cdf, cur_best_filter_control, 2);
           if (cur_best_filter_control != 0) {
               cur_filter_enabled = true;
           }
           *cur_total_rate += best_rate;
           *cur_total_dist += best_ssd;
           sb_idx++;
       }
   }
   *filter_enable = cur_filter_enabled;
}

/* Derive sample locations for CCSO */
void derive_ccso_sample_pos(int *rec_idx, const int ccsoStride, const uint8_t ext_filter_support) {
   // Input sample locations for CCSO
   //         2 1 4
   // 6 o 5 o 3 o 3 o 5 o 6
   //         4 1 2
   if (ext_filter_support == 0) {
       // Sample position 1
       rec_idx[0] = -1 * ccsoStride;
       rec_idx[1] = 1 * ccsoStride;
   } else if (ext_filter_support == 1) {
       // Sample position 2
       rec_idx[0] = -1 * ccsoStride - 1;
       rec_idx[1] = 1 * ccsoStride + 1;
   } else if (ext_filter_support == 2) {
       // Sample position 3
       rec_idx[0] = -1;
       rec_idx[1] = 1;
   } else if (ext_filter_support == 3) {
       // Sample position 4
       rec_idx[0] = 1 * ccsoStride - 1;
       rec_idx[1] = -1 * ccsoStride + 1;
   } else if (ext_filter_support == 4) {
       // Sample position 5
       rec_idx[0] = -3;
       rec_idx[1] = 3;
   } else { // if(ext_filter_support == 5) {
       // Sample position 6
       rec_idx[0] = -5;
       rec_idx[1] = 5;
   }
}

/* Derive CCSO filter support information */
void ccso_derive_src_info(MacroblockdPlane *pd, const int plane, const uint16_t *src_y, const uint8_t qstep,
                         const uint8_t filter_sup, uint8_t *src_cls0, uint8_t *src_cls1, int edge_clf) {
   const int pic_height  = pd[plane].dst.height;
   const int pic_width   = pd[plane].dst.width;
   const int y_uv_hscale = pd[plane].subsampling_x;
   const int y_uv_vscale = pd[plane].subsampling_y;
   const int neg_qstep   = qstep * -1;
   int       src_loc[2];
   derive_ccso_sample_pos(src_loc, ccso_stride_ext, filter_sup);
   const int blk_log2 = plane > 0 ? CCSO_BLK_SIZE : CCSO_BLK_SIZE + 1;
   const int blk_size = 1 << blk_log2;
   src_y += CCSO_PADDING_SIZE * ccso_stride_ext + CCSO_PADDING_SIZE;
   for (int y = 0; y < pic_height; y += blk_size) {
       for (int x = 0; x < pic_width; x += blk_size) {
           ccso_derive_src_block(src_y,
                                 src_cls0,
                                 src_cls1,
                                 ccso_stride_ext,
                                 ccso_stride,
                                 x,
                                 y,
                                 pic_width,
                                 pic_height,
                                 y_uv_hscale,
                                 y_uv_vscale,
                                 qstep,
                                 neg_qstep,
                                 src_loc,
                                 blk_size,
                                 edge_clf);
       }
       src_y += (ccso_stride_ext << (blk_log2 + y_uv_vscale));
       src_cls0 += (ccso_stride << (blk_log2 + y_uv_vscale));
       src_cls1 += (ccso_stride << (blk_log2 + y_uv_vscale));
   }
}

void ccso_pre_compute_class_err_bo(MacroblockdPlane *pd, const int plane, const uint16_t *src_y, const uint16_t *ref,
                                  const uint16_t *dst, const uint8_t shift_bits) {
   const int pic_height        = pd[plane].dst.height;
   const int pic_width         = pd[plane].dst.width;
   const int y_uv_hscale       = pd[plane].subsampling_x;
   const int y_uv_vscale       = pd[plane].subsampling_y;
   int       fb_idx            = 0;
   const int blk_log2          = plane > 0 ? CCSO_BLK_SIZE : CCSO_BLK_SIZE + 1;
   const int blk_size          = 1 << blk_log2;
   const int scaled_ext_stride = (ccso_stride_ext << y_uv_vscale);
   src_y += CCSO_PADDING_SIZE * ccso_stride_ext + CCSO_PADDING_SIZE;
   for (int y = 0; y < pic_height; y += blk_size) {
       for (int x = 0; x < pic_width; x += blk_size) {
           fb_idx++;
           const int y_end = AOMMIN(pic_height - y, blk_size);
           const int x_end = AOMMIN(pic_width - x, blk_size);
           for (int y_start = 0; y_start < y_end; y_start++) {
               for (int x_start = 0; x_start < x_end; x_start++) {
                   const int x_pos    = x + x_start;
                   const int band_num = src_y[x_pos << y_uv_hscale] >> shift_bits;
                   total_class_err_bo[band_num][fb_idx - 1] += ref[x_pos] - dst[x_pos];
                   total_class_cnt_bo[band_num][fb_idx - 1]++;
               }
               ref += ccso_stride;
               dst += ccso_stride;
               src_y += scaled_ext_stride;
           }
           ref -= ccso_stride * y_end;
           dst -= ccso_stride * y_end;
           src_y -= scaled_ext_stride * y_end;
       }
       ref += (ccso_stride << blk_log2);
       dst += (ccso_stride << blk_log2);
       src_y += (ccso_stride_ext << (blk_log2 + y_uv_vscale));
   }
}

/* Compute the residual for each entry of the LUT using CCSO enabled filter
* blocks
*/
void ccso_compute_class_err(Av1Common *cm, const int plane, MacroblockdPlane *pd, const int max_band_log2,
                           const int max_edge_interval, const uint8_t ccso_bo_only) {
   //   const CommonModeInfoParams *const mi_params = &cm->mi_params;
   const int blk_log2 = plane > 0 ? CCSO_BLK_SIZE : CCSO_BLK_SIZE + 1;
   const int nvfb     = ((cm->mi_rows >> pd[plane].subsampling_y) + (1 << blk_log2 >> MI_SIZE_LOG2) - 1) /
       (1 << blk_log2 >> MI_SIZE_LOG2);
   const int nhfb = ((cm->mi_cols >> pd[plane].subsampling_x) + (1 << blk_log2 >> MI_SIZE_LOG2) - 1) /
       (1 << blk_log2 >> MI_SIZE_LOG2);
   const int fb_count = nvfb * nhfb;
   for (int fb_idx = 0; fb_idx < fb_count; fb_idx++) {
       if (!filter_control[fb_idx])
           continue;
       if (ccso_bo_only) {
           int d0 = 0;
           int d1 = 0;
           for (int band_num = 0; band_num < (1 << max_band_log2); band_num++) {
               const int lut_idx_ext = (band_num << 4) + (d0 << 2) + d1;
               chroma_error[lut_idx_ext] += total_class_err_bo[band_num][fb_idx];
               chroma_count[lut_idx_ext] += total_class_cnt_bo[band_num][fb_idx];
           }
       } else {
           for (int d0 = 0; d0 < max_edge_interval; d0++) {
               for (int d1 = 0; d1 < max_edge_interval; d1++) {
                   for (int band_num = 0; band_num < (1 << max_band_log2); band_num++) {
                       const int lut_idx_ext = (band_num << 4) + (d0 << 2) + d1;
                       chroma_error[lut_idx_ext] += total_class_err[d0][d1][band_num][fb_idx];
                       chroma_count[lut_idx_ext] += total_class_cnt[d0][d1][band_num][fb_idx];
                   }
               }
           }
       }
   }
}

void ccso_filter_block_hbd_with_buf(const uint16_t *src_y, uint16_t *dst_yuv, const uint8_t *src_cls0,
                                   const uint8_t *src_cls1, const int src_y_stride, const int dst_stride,
                                   const int src_cls_stride, const int x, const int y, const int pic_width,
                                   const int pic_height, const int8_t *filter_offset, const int blk_size,
                                   const int y_uv_hscale, const int y_uv_vscale, const int max_val,
                                   const uint8_t shift_bits, const uint8_t ccso_bo_only) {
   if (ccso_bo_only) {
       (void)src_cls0;
       (void)src_cls1;
   }
   int       cur_src_cls0;
   int       cur_src_cls1;
   const int y_end = AOMMIN(pic_height - y, blk_size);
   const int x_end = AOMMIN(pic_width - x, blk_size);
   for (int y_start = 0; y_start < y_end; y_start++) {
       const int y_pos = y_start;
       for (int x_start = 0; x_start < x_end; x_start++) {
           const int x_pos = x + x_start;
           if (!ccso_bo_only) {
               cur_src_cls0 = src_cls0[(y_pos << y_uv_vscale) * src_cls_stride + (x_pos << y_uv_hscale)];
               cur_src_cls1 = src_cls1[(y_pos << y_uv_vscale) * src_cls_stride + (x_pos << y_uv_hscale)];
           } else {
               cur_src_cls0 = 0;
               cur_src_cls1 = 0;
           }
           const int band_num    = src_y[(y_pos << y_uv_vscale) * src_y_stride + (x_pos << y_uv_hscale)] >> shift_bits;
           const int lut_idx_ext = (band_num << 4) + (cur_src_cls0 << 2) + cur_src_cls1;
           const int offset_val  = filter_offset[lut_idx_ext];
           dst_yuv[y_pos * dst_stride + x_pos] = clamp(offset_val + dst_yuv[y_pos * dst_stride + x_pos], 0, max_val);
       }
   }
}

/* Apply CCSO on luma component at encoder (high bit-depth) */
void ccso_try_luma_filter(PictureControlSet *pcs, MacroblockdPlane *pd, const int plane, const uint16_t *src_y,
                         uint16_t *dst_yuv, const int dst_stride, const int8_t *filter_offset, uint8_t *src_cls0,
                         uint8_t *src_cls1, const uint8_t shift_bits, const uint8_t ccso_bo_only) {
   const int pic_height = pd[plane].dst.height;
   const int pic_width  = pd[plane].dst.width;
   const int max_val    = (1 << pcs->scs->static_config.encoder_bit_depth) - 1;
   const int blk_log2   = plane > 0 ? CCSO_BLK_SIZE : CCSO_BLK_SIZE + 1;
   const int blk_size   = 1 << blk_log2;
   src_y += CCSO_PADDING_SIZE * ccso_stride_ext + CCSO_PADDING_SIZE;
   for (int y = 0; y < pic_height; y += blk_size) {
       for (int x = 0; x < pic_width; x += blk_size) {
           if (ccso_bo_only) {
               ccso_filter_block_hbd_with_buf(src_y,
                                              dst_yuv,
                                              src_cls0,
                                              src_cls1,
                                              ccso_stride_ext,
                                              dst_stride,
                                              ccso_stride,
                                              x,
                                              y,
                                              pic_width,
                                              pic_height,
                                              filter_offset,
                                              blk_size,
                                              // y_uv_scale in h and v shall be zero
                                              0,
                                              0,
                                              max_val,
                                              shift_bits,
                                              ccso_bo_only);
           } else {
               ccso_filter_block_hbd_with_buf(src_y,
                                              dst_yuv,
                                              src_cls0,
                                              src_cls1,
                                              ccso_stride_ext,
                                              dst_stride,
                                              ccso_stride,
                                              x,
                                              y,
                                              pic_width,
                                              pic_height,
                                              filter_offset,
                                              blk_size,
                                              // y_uv_scale in h and v shall be zero
                                              0,
                                              0,
                                              max_val,
                                              shift_bits,
                                              0);
           }
       }
       dst_yuv += (dst_stride << blk_log2);
       src_y += (ccso_stride_ext << blk_log2);
       src_cls0 += (ccso_stride << blk_log2);
       src_cls1 += (ccso_stride << blk_log2);
   }
}

/* Apply CCSO on chroma component at encoder (high bit-depth) */
void ccso_try_chroma_filter(PictureControlSet *pcs, MacroblockdPlane *pd, const int plane, const uint16_t *src_y,
                           uint16_t *dst_yuv, const int dst_stride, const int8_t *filter_offset, uint8_t *src_cls0,
                           uint8_t *src_cls1, const uint8_t shift_bits, const uint8_t ccso_bo_only) {
   const int pic_height  = pd[plane].dst.height;
   const int pic_width   = pd[plane].dst.width;
   const int y_uv_hscale = pd[plane].subsampling_x;
   const int y_uv_vscale = pd[plane].subsampling_y;
   const int max_val     = (1 << pcs->scs->static_config.encoder_bit_depth) - 1;
   const int blk_log2    = plane > 0 ? CCSO_BLK_SIZE : CCSO_BLK_SIZE + 1;
   const int blk_size    = 1 << blk_log2;
   src_y += CCSO_PADDING_SIZE * ccso_stride_ext + CCSO_PADDING_SIZE;
   for (int y = 0; y < pic_height; y += blk_size) {
       for (int x = 0; x < pic_width; x += blk_size) {
           if (ccso_bo_only) {
               ccso_filter_block_hbd_with_buf(src_y,
                                              dst_yuv,
                                              src_cls0,
                                              src_cls1,
                                              ccso_stride_ext,
                                              dst_stride,
                                              ccso_stride,
                                              x,
                                              y,
                                              pic_width,
                                              pic_height,
                                              filter_offset,
                                              blk_size,
                                              y_uv_hscale,
                                              y_uv_vscale,
                                              max_val,
                                              shift_bits,
                                              ccso_bo_only);
           } else {
               ccso_filter_block_hbd_with_buf(src_y,
                                              dst_yuv,
                                              src_cls0,
                                              src_cls1,
                                              ccso_stride_ext,
                                              dst_stride,
                                              ccso_stride,
                                              x,
                                              y,
                                              pic_width,
                                              pic_height,
                                              filter_offset,
                                              blk_size,
                                              y_uv_hscale,
                                              y_uv_vscale,
                                              max_val,
                                              shift_bits,
                                              0);
           }
       }
       dst_yuv += (dst_stride << blk_log2);
       src_y += (ccso_stride_ext << (blk_log2 + y_uv_vscale));
       src_cls0 += (ccso_stride << (blk_log2 + y_uv_vscale));
       src_cls1 += (ccso_stride << (blk_log2 + y_uv_vscale));
   }
}

/* Count the bits for signaling the offset index */
static INLINE int count_lut_bits(int8_t *temp_filter_offset, const int max_band_log2, const int max_edge_interval,
                                const uint8_t ccso_bo_only) {
   const int ccso_offset_reordered[8]  = {0, 1, -1, 3, -3, 7, -7, -10};
   int       temp_bits                 = 0;
   int       num_edge_offset_intervals = ccso_bo_only ? 1 : max_edge_interval;
   for (int d0 = 0; d0 < num_edge_offset_intervals; d0++) {
       for (int d1 = 0; d1 < num_edge_offset_intervals; d1++) {
           for (int band_num = 0; band_num < (1 << max_band_log2); band_num++) {
               const int lut_idx_ext = (band_num << 4) + (d0 << 2) + d1;
               for (int idx = 0; idx < 7; ++idx) {
                   temp_bits++;
                   if (ccso_offset_reordered[idx] == temp_filter_offset[lut_idx_ext])
                       break;
               }
           }
       }
   }
   return temp_bits;
}

/* Compute the aggregated residual between original and reconstructed sample for
* each entry of the LUT */
void ccso_pre_compute_class_err(MacroblockdPlane *pd, const int plane, const uint16_t *src_y, const uint16_t *ref,
                               const uint16_t *dst, uint8_t *src_cls0, uint8_t *src_cls1, const uint8_t shift_bits) {
   const int pic_height  = pd[plane].dst.height;
   const int pic_width   = pd[plane].dst.width;
   const int y_uv_hscale = pd[plane].subsampling_x;
   const int y_uv_vscale = pd[plane].subsampling_y;
   int       fb_idx      = 0;
   uint8_t   cur_src_cls0;
   uint8_t   cur_src_cls1;
   const int blk_log2          = plane > 0 ? CCSO_BLK_SIZE : CCSO_BLK_SIZE + 1;
   const int blk_size          = 1 << blk_log2;
   const int scaled_ext_stride = (ccso_stride_ext << y_uv_vscale);
   const int scaled_stride     = (ccso_stride << y_uv_vscale);
   src_y += CCSO_PADDING_SIZE * ccso_stride_ext + CCSO_PADDING_SIZE;
   for (int y = 0; y < pic_height; y += blk_size) {
       for (int x = 0; x < pic_width; x += blk_size) {
           fb_idx++;
           const int y_end = AOMMIN(pic_height - y, blk_size);
           const int x_end = AOMMIN(pic_width - x, blk_size);
           for (int y_start = 0; y_start < y_end; y_start++) {
               for (int x_start = 0; x_start < x_end; x_start++) {
                   const int x_pos    = x + x_start;
                   cur_src_cls0       = src_cls0[x_pos << y_uv_hscale];
                   cur_src_cls1       = src_cls1[x_pos << y_uv_hscale];
                   const int band_num = src_y[x_pos << y_uv_hscale] >> shift_bits;
                   total_class_err[cur_src_cls0][cur_src_cls1][band_num][fb_idx - 1] += ref[x_pos] - dst[x_pos];
                   total_class_cnt[cur_src_cls0][cur_src_cls1][band_num][fb_idx - 1]++;
               }
               ref += ccso_stride;
               dst += ccso_stride;
               src_y += scaled_ext_stride;
               src_cls0 += scaled_stride;
               src_cls1 += scaled_stride;
           }
           ref -= ccso_stride * y_end;
           dst -= ccso_stride * y_end;
           src_y -= scaled_ext_stride * y_end;
           src_cls0 -= scaled_stride * y_end;
           src_cls1 -= scaled_stride * y_end;
       }
       ref += (ccso_stride << blk_log2);
       dst += (ccso_stride << blk_log2);
       src_y += (ccso_stride_ext << (blk_log2 + y_uv_vscale));
       src_cls0 += (ccso_stride << (blk_log2 + y_uv_vscale));
       src_cls1 += (ccso_stride << (blk_log2 + y_uv_vscale));
   }
}

/* Derive the offset value in the look-up table */
void derive_lut_offset(int8_t *temp_filter_offset, const int max_band_log2, const int max_edge_interval,
                      const uint8_t ccso_bo_only) {
   float temp_offset               = 0;
   int   num_edge_offset_intervals = ccso_bo_only ? 1 : max_edge_interval;
   for (int d0 = 0; d0 < num_edge_offset_intervals; d0++) {
       for (int d1 = 0; d1 < num_edge_offset_intervals; d1++) {
           for (int band_num = 0; band_num < (1 << max_band_log2); band_num++) {
               const int lut_idx_ext = (band_num << 4) + (d0 << 2) + d1;
               if (chroma_count[lut_idx_ext]) {
                   temp_offset = (float)chroma_error[lut_idx_ext] / chroma_count[lut_idx_ext];
                   if ((temp_offset < ccso_offset[0]) || (temp_offset >= ccso_offset[7])) {
                       temp_filter_offset[lut_idx_ext] = clamp((int)temp_offset, ccso_offset[0], ccso_offset[7]);
                   } else {
                       for (int offset_idx = 0; offset_idx < 7; offset_idx++) {
                           if ((temp_offset >= ccso_offset[offset_idx]) &&
                               (temp_offset <= ccso_offset[offset_idx + 1])) {
                               if (fabs(temp_offset - ccso_offset[offset_idx]) >
                                   fabs(temp_offset - ccso_offset[offset_idx + 1])) {
                                   temp_filter_offset[lut_idx_ext] = ccso_offset[offset_idx + 1];
                               } else {
                                   temp_filter_offset[lut_idx_ext] = ccso_offset[offset_idx];
                               }
                               break;
                           }
                       }
                   }
               }
           }
       }
   }
}

/* Derive the look-up table for a color component */
void derive_ccso_filter(PictureControlSet *pcs, const int plane, MacroblockdPlane *pd, const uint16_t *org_uv,
                       const uint16_t *ext_rec_y, const uint16_t *rec_uv, int rdmult) {
   struct PictureParentControlSet *ppcs    = pcs->ppcs;
   FrameHeader                    *frm_hdr = &pcs->ppcs->frm_hdr;
   Av1Common                      *cm      = ppcs->av1_cm;

   // const CommonModeInfoParams *const mi_params = &cm->mi_params;
   const int log2_filter_unit_size = plane > 0 ? CCSO_BLK_SIZE : CCSO_BLK_SIZE + 1;

   const int ccso_nvfb = ((cm->mi_rows >> pd[plane].subsampling_y) + (1 << log2_filter_unit_size >> 2) - 1) /
       (1 << log2_filter_unit_size >> 2);
   const int ccso_nhfb = ((cm->mi_cols >> pd[plane].subsampling_x) + (1 << log2_filter_unit_size >> 2) - 1) /
       (1 << log2_filter_unit_size >> 2);
   const int sb_count     = ccso_nvfb * ccso_nhfb;
   const int pic_height_c = pd[plane].dst.height;
   const int pic_width_c  = pd[plane].dst.width;

   uint16_t *temp_rec_uv_buf;
   unfiltered_dist_frame = 0;
   unfiltered_dist_block = svt_aom_malloc(sizeof(*unfiltered_dist_block) * sb_count);
   memset(unfiltered_dist_block, 0, sizeof(*unfiltered_dist_block) * sb_count);
   training_dist_block = svt_aom_malloc(sizeof(*training_dist_block) * sb_count);
   memset(training_dist_block, 0, sizeof(*training_dist_block) * sb_count);
   filter_control = svt_aom_malloc(sizeof(*filter_control) * sb_count);
   memset(filter_control, 0, sizeof(*filter_control) * sb_count);
   best_filter_control = svt_aom_malloc(sizeof(*best_filter_control) * sb_count);
   memset(best_filter_control, 0, sizeof(*best_filter_control) * sb_count);
   final_filter_control = svt_aom_malloc(sizeof(*final_filter_control) * sb_count);
   memset(final_filter_control, 0, sizeof(*final_filter_control) * sb_count);
   temp_rec_uv_buf = svt_aom_malloc(sizeof(*temp_rec_uv_buf) * pd[0].dst.height * ccso_stride);
   for (int d0 = 0; d0 < CCSO_INPUT_INTERVAL; d0++) {
       for (int d1 = 0; d1 < CCSO_INPUT_INTERVAL; d1++) {
           for (int band_num = 0; band_num < CCSO_BAND_NUM; band_num++) {
               total_class_err[d0][d1][band_num] = svt_aom_malloc(sizeof(*total_class_err[d0][d1][band_num]) *
                                                                  sb_count);
               total_class_cnt[d0][d1][band_num] = svt_aom_malloc(sizeof(*total_class_cnt[d0][d1][band_num]) *
                                                                  sb_count);
           }
       }
   }
   for (int band_num = 0; band_num < CCSO_BAND_NUM; band_num++) {
       total_class_err_bo[band_num] = svt_aom_malloc(sizeof(*total_class_err_bo[band_num]) * sb_count);
       total_class_cnt_bo[band_num] = svt_aom_malloc(sizeof(*total_class_cnt_bo[band_num]) * sb_count);
   }
   compute_distortion(org_uv,
                      ccso_stride,
                      rec_uv,
                      ccso_stride,
                      log2_filter_unit_size,
                      pic_height_c,
                      pic_width_c,
                      unfiltered_dist_block,
                      ccso_nhfb,
                      &unfiltered_dist_frame);

   const double best_unfiltered_cost = RDCOST_DBL_WITH_NATIVE_BD_DIST1(
       rdmult, 1, unfiltered_dist_frame, pcs->scs->static_config.encoder_bit_depth);
   double    best_filtered_cost;
   double    final_filtered_cost   = DBL_MAX;
   uint8_t   best_edge_classifier  = 0;
   uint8_t   final_edge_classifier = 0;
   const int total_edge_classifier = 2;
   int8_t    filter_offset[CCSO_BAND_NUM * 16];
   const int total_filter_support  = 6;
   const int total_quant_idx       = 4;
   const int total_band_log2_plus1 = 4;
   uint8_t   frame_bits            = 1;
#if CONFIG_CCSO_SIGFIX
   uint8_t frame_bits_bo_only = 1; // enabling flag
   frame_bits_bo_only += 1; // bo only flag
   frame_bits += 1; // bo only flag
#endif // CONFIG_CCSO_SIGFIX
   frame_bits += 2; // quant step size
   frame_bits += 3; // filter support index
   frame_bits += 1; // edge_clf
   frame_bits += 2; // band number log2
#if CONFIG_CCSO_SIGFIX
   frame_bits_bo_only += 3; // band number log2
#endif // CONFIG_CCSO_SIGFIX
   uint8_t *src_cls0;
   uint8_t *src_cls1;
   src_cls0 = svt_aom_malloc(sizeof(*src_cls0) * pd[0].dst.height * pd[0].dst.width);
   src_cls1 = svt_aom_malloc(sizeof(*src_cls1) * pd[0].dst.height * pd[0].dst.width);
   memset(src_cls0, 0, sizeof(*src_cls0) * pd[0].dst.height * pd[0].dst.width);
   memset(src_cls1, 0, sizeof(*src_cls1) * pd[0].dst.height * pd[0].dst.width);

   for (uint8_t ccso_bo_only = 0; ccso_bo_only < 2; ccso_bo_only++) {
#if CONFIG_CCSO_SIGFIX
       int num_filter_iter   = ccso_bo_only ? 1 : total_filter_support;
       int num_quant_iter    = ccso_bo_only ? 1 : total_quant_idx;
       int num_edge_clf_iter = ccso_bo_only ? 1 : total_edge_classifier;
#else
       int num_filter_iter   = total_filter_support;
       int num_quant_iter    = total_quant_idx;
       int num_edge_clf_iter = total_edge_classifier;
#endif // CONFIG_CCSO_SIGFIX
       for (int ext_filter_support = 0; ext_filter_support < num_filter_iter; ext_filter_support++) {
           for (int quant_idx = 0; quant_idx < num_quant_iter; quant_idx++) {
               for (int edge_clf = 0; edge_clf < num_edge_clf_iter; edge_clf++) {
                   const int max_edge_interval = edge_clf_to_edge_interval[edge_clf];
                   if (!ccso_bo_only) {
                       ccso_derive_src_info(pd,
                                            plane,
                                            ext_rec_y,
                                            quant_sz[quant_idx],
                                            ext_filter_support,
                                            src_cls0,
                                            src_cls1,
                                            edge_clf);
                   }
                   int num_band_iter = total_band_log2_plus1;
                   if (ccso_bo_only) {
                       num_band_iter = total_band_log2_plus1 + 4;
                   }
                   for (int max_band_log2 = 0; max_band_log2 < num_band_iter; max_band_log2++) {
                       const int shift_bits   = pcs->scs->static_config.encoder_bit_depth - max_band_log2;
                       const int max_band     = 1 << max_band_log2;
                       best_filtered_cost     = DBL_MAX;
                       bool   ccso_enable     = true;
                       bool   keep_training   = true;
                       bool   improvement     = false;
                       double prev_total_cost = DBL_MAX;
                       int    control_idx     = 0;
                       for (int y = 0; y < ccso_nvfb; y++) {
                           for (int x = 0; x < ccso_nhfb; x++) {
                               filter_control[control_idx] = 1;
                               control_idx++;
                           }
                       }
                       if (ccso_bo_only) {
                           for (int band_num = 0; band_num < CCSO_BAND_NUM; band_num++) {
                               memset(
                                   total_class_err_bo[band_num], 0, sizeof(*total_class_err_bo[band_num]) * sb_count);
                               memset(
                                   total_class_cnt_bo[band_num], 0, sizeof(*total_class_cnt_bo[band_num]) * sb_count);
                           }
                           ccso_pre_compute_class_err_bo(pd, plane, ext_rec_y, org_uv, rec_uv, shift_bits);
                       } else {
                           for (int d0 = 0; d0 < max_edge_interval; d0++) {
                               for (int d1 = 0; d1 < max_edge_interval; d1++) {
                                   for (int band_num = 0; band_num < max_band; band_num++) {
                                       memset(total_class_err[d0][d1][band_num],
                                              0,
                                              sizeof(*total_class_err[d0][d1][band_num]) * sb_count);
                                       memset(total_class_cnt[d0][d1][band_num],
                                              0,
                                              sizeof(*total_class_cnt[d0][d1][band_num]) * sb_count);
                                   }
                               }
                           }
                           ccso_pre_compute_class_err(
                               pd, plane, ext_rec_y, org_uv, rec_uv, src_cls0, src_cls1, shift_bits);
                       }
                       int training_iter_count = 0;
                       while (keep_training) {
                           improvement = false;
                           if (ccso_enable) {
                               memset(chroma_error, 0, sizeof(chroma_error));
                               memset(chroma_count, 0, sizeof(chroma_count));
                               memset(filter_offset, 0, sizeof(filter_offset));
                               memcpy(
                                   temp_rec_uv_buf, rec_uv, sizeof(*temp_rec_uv_buf) * pd[0].dst.height * ccso_stride);
                               ccso_compute_class_err(cm, plane, pd, max_band_log2, max_edge_interval, ccso_bo_only);
                               derive_lut_offset(filter_offset, max_band_log2, max_edge_interval, ccso_bo_only);
                           }
                           memcpy(temp_rec_uv_buf, rec_uv, sizeof(*temp_rec_uv_buf) * pd[0].dst.height * ccso_stride);
                           if (plane > 0)
                               ccso_try_chroma_filter(pcs,
                                                      pd,
                                                      plane,
                                                      ext_rec_y,
                                                      temp_rec_uv_buf,
                                                      ccso_stride,
                                                      filter_offset,
                                                      src_cls0,
                                                      src_cls1,
                                                      shift_bits,
                                                      ccso_bo_only);
                           else
                               ccso_try_luma_filter(pcs,
                                                    pd,
                                                    plane,
                                                    ext_rec_y,
                                                    temp_rec_uv_buf,
                                                    ccso_stride,
                                                    filter_offset,
                                                    src_cls0,
                                                    src_cls1,
                                                    shift_bits,
                                                    ccso_bo_only);
                           filtered_dist_frame = 0;
                           compute_distortion(org_uv,
                                              ccso_stride,
                                              temp_rec_uv_buf,
                                              ccso_stride,
                                              log2_filter_unit_size,
                                              pic_height_c,
                                              pic_width_c,
                                              training_dist_block,
                                              ccso_nhfb,
                                              &filtered_dist_frame);
                           uint64_t cur_total_dist = 0;
                           int      cur_total_rate = 0;
                           derive_blk_md(cm,
                                         pd,
                                         plane,
                                         unfiltered_dist_block,
                                         training_dist_block,
                                         filter_control,
                                         &cur_total_dist,
                                         &cur_total_rate,
                                         &ccso_enable);
                           if (ccso_enable) {
                               const int lut_bits = count_lut_bits(
                                   filter_offset, max_band_log2, max_edge_interval, ccso_bo_only);
                               const int cur_total_bits = lut_bits +
                                   (
#if CONFIG_CCSO_SIGFIX
                                                              ccso_bo_only ? frame_bits_bo_only :
#endif // CONFIG_CCSO_SIGFIX
                                                                           frame_bits) +
                                   sb_count;
                               const double cur_total_cost = RDCOST_DBL_WITH_NATIVE_BD_DIST1(
                                   rdmult, cur_total_bits, cur_total_dist, pcs->scs->static_config.encoder_bit_depth);
                               if (cur_total_cost < prev_total_cost) {
                                   prev_total_cost = cur_total_cost;
                                   improvement     = true;
                               }
                               if (cur_total_cost < best_filtered_cost) {
                                   best_filtered_cost         = cur_total_cost;
                                   best_filter_enabled[plane] = ccso_enable;
                                   memcpy(best_filter_offset[plane], filter_offset, sizeof(filter_offset));
                                   best_edge_classifier = edge_clf;
                                   memcpy(best_filter_control, filter_control, sizeof(*filter_control) * sb_count);
                               }
                           }
                           training_iter_count++;
                           if (!improvement || training_iter_count > CCSO_MAX_ITERATIONS) {
                               keep_training = false;
                           }
                       }

                       if (best_filtered_cost < final_filtered_cost) {
                           final_filtered_cost             = best_filtered_cost;
                           final_filter_enabled[plane]     = best_filter_enabled[plane];
                           final_quant_idx[plane]          = quant_idx;
                           final_ext_filter_support[plane] = ext_filter_support;
                           final_ccso_bo_only[plane]       = ccso_bo_only;
                           memcpy(final_filter_offset[plane],
                                  best_filter_offset[plane],
                                  sizeof(best_filter_offset[plane]));
                           final_band_log2       = max_band_log2;
                           final_edge_classifier = best_edge_classifier;
                           memcpy(final_filter_control, best_filter_control, sizeof(*best_filter_control) * sb_count);
                       }
                   }
               }
           }
       }
   }

   if (best_unfiltered_cost < final_filtered_cost) {
       memset(final_filter_control, 0, sizeof(*final_filter_control) * sb_count);
       frm_hdr->ccso_info.ccso_enable[plane] = false;
   } else {
       frm_hdr->ccso_info.ccso_enable[plane] = true;
   }
   if (frm_hdr->ccso_info.ccso_enable[plane]) {
       for (int y_sb = 0; y_sb < ccso_nvfb; y_sb++) {
           for (int x_sb = 0; x_sb < ccso_nhfb; x_sb++) {
               if (plane == AOM_PLANE_Y) {
                   pcs->mi_grid_base[(1 << CCSO_BLK_SIZE >> (MI_SIZE_LOG2 - pd[1].subsampling_y)) * y_sb *
                                         pcs->mi_stride +
                                     (1 << CCSO_BLK_SIZE >> (MI_SIZE_LOG2 - pd[1].subsampling_x)) * x_sb]
                       ->mbmi.ccso_blk_y = final_filter_control[y_sb * ccso_nhfb + x_sb];
               } else if (plane == AOM_PLANE_U) {
                   pcs->mi_grid_base[(1 << CCSO_BLK_SIZE >> (MI_SIZE_LOG2 - pd[1].subsampling_y)) * y_sb *
                                         pcs->mi_stride +
                                     (1 << CCSO_BLK_SIZE >> (MI_SIZE_LOG2 - pd[1].subsampling_x)) * x_sb]
                       ->mbmi.ccso_blk_u = final_filter_control[y_sb * ccso_nhfb + x_sb];
               } else {
                   pcs->mi_grid_base[(1 << CCSO_BLK_SIZE >> (MI_SIZE_LOG2 - pd[2].subsampling_y)) * y_sb *
                                         pcs->mi_stride +
                                     (1 << CCSO_BLK_SIZE >> (MI_SIZE_LOG2 - pd[2].subsampling_x)) * x_sb]
                       ->mbmi.ccso_blk_v = final_filter_control[y_sb * ccso_nhfb + x_sb];
               }
           }
       }
       memcpy(frm_hdr->ccso_info.filter_offset[plane], final_filter_offset[plane], sizeof(final_filter_offset[plane]));
       frm_hdr->ccso_info.quant_idx[plane]          = final_quant_idx[plane];
       frm_hdr->ccso_info.ext_filter_support[plane] = final_ext_filter_support[plane];
       frm_hdr->ccso_info.ccso_bo_only[plane]       = final_ccso_bo_only[plane];
       frm_hdr->ccso_info.max_band_log2[plane]      = final_band_log2;
       frm_hdr->ccso_info.edge_clf[plane]           = final_edge_classifier;
   }
   svt_aom_free(unfiltered_dist_block);
   svt_aom_free(training_dist_block);
   svt_aom_free(filter_control);
   svt_aom_free(final_filter_control);
   svt_aom_free(temp_rec_uv_buf);
   svt_aom_free(best_filter_control);
   svt_aom_free(src_cls0);
   svt_aom_free(src_cls1);
   for (int d0 = 0; d0 < CCSO_INPUT_INTERVAL; d0++) {
       for (int d1 = 0; d1 < CCSO_INPUT_INTERVAL; d1++) {
           for (int band_num = 0; band_num < CCSO_BAND_NUM; band_num++) {
               svt_aom_free(total_class_err[d0][d1][band_num]);
               svt_aom_free(total_class_cnt[d0][d1][band_num]);
           }
       }
   }
   for (int band_num = 0; band_num < CCSO_BAND_NUM; band_num++) {
       svt_aom_free(total_class_err_bo[band_num]);
       svt_aom_free(total_class_cnt_bo[band_num]);
   }
}

/* Derive the look-up table for a frame */
void ccso_search(PictureControlSet *pcs, MacroblockdPlane *pd, int rdmult, const uint16_t *ext_rec_y,
                uint16_t *rec_uv[3], uint16_t *org_uv[3]) {
   FrameHeader *frm_hdr       = &pcs->ppcs->frm_hdr;
   int          rdmult_weight = clamp(frm_hdr->quantization_params.base_q_idx >> 3, 1, 37);
   int64_t      rdmult_temp   = (int64_t)rdmult * (int64_t)rdmult_weight;
   if (rdmult_temp < INT_MAX)
       rdmult = (int)rdmult_temp;
   else
       return;

   const int            num_planes = av1_num_planes(&pcs->scs->seq_header.color_config);
   EbPictureBufferDesc *recon_pic;
   svt_aom_get_recon_pic(pcs, &recon_pic, pcs->scs->is_16bit_pipeline);
   svt_av1_setup_dst_planes(pcs, pd, pcs->ppcs->scs->seq_header.sb_size, recon_pic, 0, 0, 0, num_planes);

   ccso_stride     = pd[0].dst.width;
   ccso_stride_ext = pd[0].dst.width + (CCSO_PADDING_SIZE << 1);

   derive_ccso_filter(pcs, AOM_PLANE_Y, pd, org_uv[AOM_PLANE_Y], ext_rec_y, rec_uv[AOM_PLANE_Y], rdmult);
#if CONFIG_D143_CCSO_FM_FLAG
   frm_hdr->ccso_info.ccso_frame_flag = frm_hdr->ccso_info.ccso_enable[0];
#endif // CONFIG_D143_CCSO_FM_FLAG
   if (num_planes > 1) {
       derive_ccso_filter(pcs, AOM_PLANE_U, pd, org_uv[AOM_PLANE_U], ext_rec_y, rec_uv[AOM_PLANE_U], rdmult);
       derive_ccso_filter(pcs, AOM_PLANE_V, pd, org_uv[AOM_PLANE_V], ext_rec_y, rec_uv[AOM_PLANE_V], rdmult);
#if CONFIG_D143_CCSO_FM_FLAG
       frm_hdr->ccso_info.ccso_frame_flag |= frm_hdr->ccso_info.ccso_enable[1];
       frm_hdr->ccso_info.ccso_frame_flag |= frm_hdr->ccso_info.ccso_enable[2];
#endif // CONFIG_D143_CCSO_FM_FLAG
   }
}
//
///* ****************************************************************************************************************** */

void ccso_filter_block_hbd_wo_buf(const uint16_t *src_y, uint8_t *dst_yuv, const int x, const int y,
                                 const int pic_width, const int pic_height, int *src_cls, const int8_t *offset_buf,
                                 const int src_y_stride, const int dst_stride, const int y_uv_hscale,
                                 const int y_uv_vscale, const int thr, const int neg_thr, const int *src_loc,
                                 const int max_val, const int blk_size, const bool isSingleBand,
                                 const uint8_t shift_bits, const int edge_clf, const uint8_t ccso_bo_only) {
   const int y_end = AOMMIN(pic_height - y, blk_size);
   const int x_end = AOMMIN(pic_width - x, blk_size);
   for (int y_start = 0; y_start < y_end; y_start++) {
       const int y_pos = y_start;
       for (int x_start = 0; x_start < x_end; x_start++) {
           const int x_pos = x + x_start;
           if (!ccso_bo_only) {
               cal_filter_support(src_cls,
                                  &src_y[(y_pos << y_uv_vscale) * src_y_stride + (x_pos << y_uv_hscale)],
                                  thr,
                                  neg_thr,
                                  src_loc,
                                  edge_clf);
           } else {
               src_cls[0] = 0;
               src_cls[1] = 0;
           }
           const int band_num                  = isSingleBand
                                ? 0
                                : src_y[(y_pos << y_uv_vscale) * src_y_stride + (x_pos << y_uv_hscale)] >> shift_bits;
           const int lut_idx_ext               = (band_num << 4) + (src_cls[0] << 2) + src_cls[1];
           const int offset_val                = offset_buf[lut_idx_ext];
        //    Bool       is_16bit      = scs->is_16bit_pipeline;

           dst_yuv[y_pos * dst_stride + x_pos] = clamp(offset_val + dst_yuv[y_pos * dst_stride + x_pos], 0, max_val);
       }
   }
}

/* Apply CCSO on luma component when multiple bands are applied */
void ccso_apply_luma_mb_filter(PictureControlSet *pcs, MacroblockdPlane *pd, const int plane, const uint16_t *src_y,
                              uint8_t *dst_yuv, const int dst_stride, const uint8_t thr, const uint8_t filter_sup,
                              const uint8_t max_band_log2, const int edge_clf) {
   //   const CommonModeInfoParams *const mi_params = &cm->mi_params;
   FrameHeader *frm_hdr = &pcs->ppcs->frm_hdr;

   const int     ccso_ext_stride = pd[0].dst.width + (CCSO_PADDING_SIZE << 1);
   const int     pic_height      = pd[plane].dst.height;
   const int     pic_width       = pd[plane].dst.width;
   const uint8_t shift_bits      = pcs->scs->static_config.encoder_bit_depth - max_band_log2;
   const int     max_val         = (1 << pcs->scs->static_config.encoder_bit_depth) - 1;
   int           src_cls[2];
   const int     neg_thr = thr * -1;
   int           src_loc[2];
   derive_ccso_sample_pos(src_loc, ccso_ext_stride, filter_sup);
   const int blk_log2 = plane > 0 ? CCSO_BLK_SIZE : CCSO_BLK_SIZE + 1;
   const int blk_size = 1 << blk_log2;
   src_y += CCSO_PADDING_SIZE * ccso_ext_stride + CCSO_PADDING_SIZE;
   for (int y = 0; y < pic_height; y += blk_size) {
       for (int x = 0; x < pic_width; x += blk_size) {
           const int ccso_blk_idx = (blk_size >> (MI_SIZE_LOG2 - pd[plane].subsampling_y)) * (y >> blk_log2) *
                   pcs->mi_stride +
               (blk_size >> (MI_SIZE_LOG2 - pd[plane].subsampling_x)) * (x >> blk_log2);
           const bool use_ccso = pcs->mi_grid_base[ccso_blk_idx]->mbmi.ccso_blk_y;
           if (!use_ccso)
               continue;
           if (frm_hdr->ccso_info.ccso_bo_only[plane]) {
               ccso_filter_block_hbd_wo_buf(src_y,
                                            dst_yuv,
                                            x,
                                            y,
                                            pic_width,
                                            pic_height,
                                            src_cls,
                                            frm_hdr->ccso_info.filter_offset[plane],
                                            ccso_ext_stride,
                                            dst_stride,
                                            0,
                                            0,
                                            thr,
                                            neg_thr,
                                            src_loc,
                                            max_val,
                                            blk_size,
                                            false,
                                            shift_bits,
                                            edge_clf,
                                            frm_hdr->ccso_info.ccso_bo_only[plane]);
           } else {
               ccso_filter_block_hbd_wo_buf(src_y,
                                            dst_yuv,
                                            x,
                                            y,
                                            pic_width,
                                            pic_height,
                                            src_cls,
                                            frm_hdr->ccso_info.filter_offset[plane],
                                            ccso_ext_stride,
                                            dst_stride,
                                            0,
                                            0,
                                            thr,
                                            neg_thr,
                                            src_loc,
                                            max_val,
                                            blk_size,
                                            false,
                                            shift_bits,
                                            edge_clf,
                                            0);
           }
       }
       dst_yuv += (dst_stride << blk_log2);
       src_y += (ccso_ext_stride << blk_log2);
   }
}

/* Apply CCSO on luma component when single band is applied */
void ccso_apply_luma_sb_filter(PictureControlSet *pcs, MacroblockdPlane *pd, const int plane, const uint16_t *src_y,
                              uint8_t *dst_yuv, const int dst_stride, const uint8_t thr, const uint8_t filter_sup,
                              const uint8_t max_band_log2, const int edge_clf) {
   (void)max_band_log2;
   FrameHeader *frm_hdr = &pcs->ppcs->frm_hdr;

   //   const CommonModeInfoParams *const mi_params = &cm->mi_params;
   const int ccso_ext_stride = pd[0].dst.width + (CCSO_PADDING_SIZE << 1);
   const int pic_height      = pd[plane].dst.height;
   const int pic_width       = pd[plane].dst.width;

   const uint8_t shift_bits = pcs->scs->static_config.encoder_bit_depth;
   const int     max_val    = (1 << pcs->scs->static_config.encoder_bit_depth) - 1;
   int           src_cls[2];
   const int     neg_thr = thr * -1;
   int           src_loc[2];
   derive_ccso_sample_pos(src_loc, ccso_ext_stride, filter_sup);
   const int blk_log2 = plane > 0 ? CCSO_BLK_SIZE : CCSO_BLK_SIZE + 1;
   const int blk_size = 1 << blk_log2;
   src_y += CCSO_PADDING_SIZE * ccso_ext_stride + CCSO_PADDING_SIZE;
   for (int y = 0; y < pic_height; y += blk_size) {
       for (int x = 0; x < pic_width; x += blk_size) {
           const int ccso_blk_idx = (blk_size >> (MI_SIZE_LOG2 - pd[plane].subsampling_y)) * (y >> blk_log2) *
                   pcs->mi_stride +
               (blk_size >> (MI_SIZE_LOG2 - pd[plane].subsampling_x)) * (x >> blk_log2);
           const bool use_ccso = pcs->mi_grid_base[ccso_blk_idx]->mbmi.ccso_blk_y;
           if (!use_ccso)
               continue;
           if (frm_hdr->ccso_info.ccso_bo_only[plane]) {
               ccso_filter_block_hbd_wo_buf(src_y,
                                            dst_yuv,
                                            x,
                                            y,
                                            pic_width,
                                            pic_height,
                                            src_cls,
                                            frm_hdr->ccso_info.filter_offset[plane],
                                            ccso_ext_stride,
                                            dst_stride,
                                            0,
                                            0,
                                            thr,
                                            neg_thr,
                                            src_loc,
                                            max_val,
                                            blk_size,
                                            true,
                                            shift_bits,
                                            edge_clf,
                                            frm_hdr->ccso_info.ccso_bo_only[plane]);
           } else {
               ccso_filter_block_hbd_wo_buf(src_y,
                                            dst_yuv,
                                            x,
                                            y,
                                            pic_width,
                                            pic_height,
                                            src_cls,
                                            frm_hdr->ccso_info.filter_offset[plane],
                                            ccso_ext_stride,
                                            dst_stride,
                                            0,
                                            0,
                                            thr,
                                            neg_thr,
                                            src_loc,
                                            max_val,
                                            blk_size,
                                            true,
                                            shift_bits,
                                            edge_clf,
                                            0);
           }
       }
       dst_yuv += (dst_stride << blk_log2);
       src_y += (ccso_ext_stride << blk_log2);
   }
}

/* Apply CCSO on chroma component when multiple bands are applied */
void ccso_apply_chroma_mb_filter(PictureControlSet *pcs, MacroblockdPlane *pd, const int plane, const uint16_t *src_y,
                                uint8_t *dst_yuv, const int dst_stride, const uint8_t thr, const uint8_t filter_sup,
                                const uint8_t max_band_log2, const int edge_clf) {
   FrameHeader *frm_hdr = &pcs->ppcs->frm_hdr;

   //   const CommonModeInfoParams *const mi_params = &cm->mi_params;
   const int     ccso_ext_stride = pd[0].dst.width + (CCSO_PADDING_SIZE << 1);
   const int     pic_height      = pd[plane].dst.height;
   const int     pic_width       = pd[plane].dst.width;
   const int     y_uv_hscale     = pd[plane].subsampling_x;
   const int     y_uv_vscale     = pd[plane].subsampling_y;
   const uint8_t shift_bits      = pcs->scs->static_config.encoder_bit_depth - max_band_log2;
   const int     max_val         = (1 << pcs->scs->static_config.encoder_bit_depth) - 1;
   int           src_cls[2];
   const int     neg_thr = thr * -1;
   int           src_loc[2];
   derive_ccso_sample_pos(src_loc, ccso_ext_stride, filter_sup);
   const int blk_log2 = plane > 0 ? CCSO_BLK_SIZE : CCSO_BLK_SIZE + 1;
   const int blk_size = 1 << blk_log2;
   src_y += CCSO_PADDING_SIZE * ccso_ext_stride + CCSO_PADDING_SIZE;
   for (int y = 0; y < pic_height; y += blk_size) {
       for (int x = 0; x < pic_width; x += blk_size) {
           const int ccso_blk_idx = (blk_size >> (MI_SIZE_LOG2 - pd[plane].subsampling_y)) * (y >> blk_log2) *
                   pcs->mi_stride +
               (blk_size >> (MI_SIZE_LOG2 - pd[plane].subsampling_x)) * (x >> blk_log2);
           // pcs->mi_grid_base[(1 << CCSO_BLK_SIZE >> (MI_SIZE_LOG2 - pd[1].subsampling_y)) * y_sb * pcs->mi_stride +
           //                     (1 << CCSO_BLK_SIZE >> (MI_SIZE_LOG2 - pd[1].subsampling_x)) * x_sb]
           //                         ->mbmi.ccso_blk_y = final_filter_control[y_sb * ccso_nhfb + x_sb];
           const bool use_ccso = (plane == 1) ? pcs->mi_grid_base[ccso_blk_idx]->mbmi.ccso_blk_u
                                              : pcs->mi_grid_base[ccso_blk_idx]->mbmi.ccso_blk_v;
           if (!use_ccso)
               continue;
           if (frm_hdr->ccso_info.ccso_bo_only[plane]) {
               ccso_filter_block_hbd_wo_buf(src_y,
                                            dst_yuv,
                                            x,
                                            y,
                                            pic_width,
                                            pic_height,
                                            src_cls,
                                            frm_hdr->ccso_info.filter_offset[plane],
                                            ccso_ext_stride,
                                            dst_stride,
                                            y_uv_hscale,
                                            y_uv_vscale,
                                            thr,
                                            neg_thr,
                                            src_loc,
                                            max_val,
                                            blk_size,
                                            false,
                                            shift_bits,
                                            edge_clf,
                                            frm_hdr->ccso_info.ccso_bo_only[plane]);
           } else {
               ccso_filter_block_hbd_wo_buf(src_y,
                                            dst_yuv,
                                            x,
                                            y,
                                            pic_width,
                                            pic_height,
                                            src_cls,
                                            frm_hdr->ccso_info.filter_offset[plane],
                                            ccso_ext_stride,
                                            dst_stride,
                                            y_uv_hscale,
                                            y_uv_vscale,
                                            thr,
                                            neg_thr,
                                            src_loc,
                                            max_val,
                                            blk_size,
                                            false,
                                            shift_bits,
                                            edge_clf,
                                            0);
           }
       }
       dst_yuv += (dst_stride << blk_log2);
       src_y += (ccso_ext_stride << (blk_log2 + y_uv_vscale));
   }
}

/* Apply CCSO on chroma component when single bands is applied */
void ccso_apply_chroma_sb_filter(PictureControlSet *pcs, MacroblockdPlane *pd, const int plane, const uint16_t *src_y,
                                uint8_t *dst_yuv, const int dst_stride, const uint8_t thr, const uint8_t filter_sup,
                                const uint8_t max_band_log2, const int edge_clf) {
   (void)max_band_log2;
   FrameHeader *frm_hdr = &pcs->ppcs->frm_hdr;

   //   const CommonModeInfoParams *const mi_params = &cm->mi_params;
   const int     ccso_ext_stride = pd[0].dst.width + (CCSO_PADDING_SIZE << 1);
   const int     pic_height      = pd[plane].dst.height;
   const int     pic_width       = pd[plane].dst.width;
   const uint8_t shift_bits      = pcs->scs->static_config.encoder_bit_depth;
   const int     y_uv_hscale     = pd[plane].subsampling_x;
   const int     y_uv_vscale     = pd[plane].subsampling_y;
   const int     max_val         = (1 << pcs->scs->static_config.encoder_bit_depth) - 1;
   int           src_cls[2];
   const int     neg_thr = thr * -1;
   int           src_loc[2];
   derive_ccso_sample_pos(src_loc, ccso_ext_stride, filter_sup);
   const int blk_log2 = plane > 0 ? CCSO_BLK_SIZE : CCSO_BLK_SIZE + 1;
   const int blk_size = 1 << blk_log2;
   src_y += CCSO_PADDING_SIZE * ccso_ext_stride + CCSO_PADDING_SIZE;
   for (int y = 0; y < pic_height; y += blk_size) {
       for (int x = 0; x < pic_width; x += blk_size) {
           const int ccso_blk_idx = (blk_size >> (MI_SIZE_LOG2 - pd[plane].subsampling_y)) * (y >> blk_log2) *
                   pcs->mi_stride +
               (blk_size >> (MI_SIZE_LOG2 - pd[plane].subsampling_x)) * (x >> blk_log2);
           const bool use_ccso = (plane == 1) ? pcs->mi_grid_base[ccso_blk_idx]->mbmi.ccso_blk_u
                                              : pcs->mi_grid_base[ccso_blk_idx]->mbmi.ccso_blk_v;
           if (!use_ccso)
               continue;
           if (frm_hdr->ccso_info.ccso_bo_only[plane]) {
               ccso_filter_block_hbd_wo_buf(src_y,
                                            dst_yuv,
                                            x,
                                            y,
                                            pic_width,
                                            pic_height,
                                            src_cls,
                                            frm_hdr->ccso_info.filter_offset[plane],
                                            ccso_ext_stride,
                                            dst_stride,
                                            y_uv_hscale,
                                            y_uv_vscale,
                                            thr,
                                            neg_thr,
                                            src_loc,
                                            max_val,
                                            blk_size,
                                            true,
                                            shift_bits,
                                            edge_clf,
                                            frm_hdr->ccso_info.ccso_bo_only[plane]);
           } else {
               ccso_filter_block_hbd_wo_buf(src_y,
                                            dst_yuv,
                                            x,
                                            y,
                                            pic_width,
                                            pic_height,
                                            src_cls,
                                            frm_hdr->ccso_info.filter_offset[plane],
                                            ccso_ext_stride,
                                            dst_stride,
                                            y_uv_hscale,
                                            y_uv_vscale,
                                            thr,
                                            neg_thr,
                                            src_loc,
                                            max_val,
                                            blk_size,
                                            true,
                                            shift_bits,
                                            edge_clf,
                                            0);
           }
       }
       dst_yuv += (dst_stride << blk_log2);
       src_y += (ccso_ext_stride << (blk_log2 + y_uv_vscale));
   }
}

/* Apply CCSO for one frame */
// EbPictureBufferDesc *recon_pic
void ccso_frame(EbPictureBufferDesc *frame, PictureControlSet *pcs, MacroblockdPlane *pd, uint16_t *ext_rec_y) {
//    struct PictureParentControlSet *ppcs    = pcs->ppcs;
   FrameHeader                    *frm_hdr = &pcs->ppcs->frm_hdr;
//    Av1Common                      *cm      = ppcs->av1_cm;

   const int num_planes = av1_num_planes(&pcs->scs->seq_header.color_config);
   svt_av1_setup_dst_planes(pcs, pd, pcs->ppcs->scs->seq_header.sb_size, frame, 0, 0, 0, num_planes);

   const uint8_t quant_sz[4] = {16, 8, 32, 64};
   for (int plane = 0; plane < num_planes; plane++) {
       const int     dst_stride      = pd[plane].dst.stride;
       const uint8_t quant_step_size = quant_sz[frm_hdr->ccso_info.quant_idx[plane]];
       if (frm_hdr->ccso_info.ccso_enable[plane]) {
           CCSO_FILTER_FUNC apply_ccso_filter_func = frm_hdr->ccso_info.max_band_log2[plane]
               ? (plane > 0 ? ccso_apply_chroma_mb_filter : ccso_apply_luma_mb_filter)
               : (plane > 0 ? ccso_apply_chroma_sb_filter : ccso_apply_luma_sb_filter);
           apply_ccso_filter_func(pcs,
                                  pd,
                                  plane,
                                  ext_rec_y,
                                  &(pd[plane].dst.buf)[0],
                                  dst_stride,
                                  quant_step_size,
                                  frm_hdr->ccso_info.ext_filter_support[plane],
                                  frm_hdr->ccso_info.max_band_log2[plane],
                                  frm_hdr->ccso_info.edge_clf[plane]);
       }
   }
}