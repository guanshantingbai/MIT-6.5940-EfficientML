# Codes
## loop unrolling
```cpp
#include <assert.h>

#include <pthread.h>

#include <stdio.h>

  

#include <cmath>

#include <cstdlib>

  

#include "../matmul.h"

#include "common.h"

  

namespace matmul {

void MatmulOperator::mat_mul_loop_unrolling(struct matmul_params *params) {

const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;

const int block_size = params->block_size; // block_size = 32

float *scale = params->scales, *offset = params->offset;

  

quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

  

int m = C->row, n = C->column, k = A->column;

// A: m x k; B: n x k; C: m x n

for (int row = 0; row < m; row++) {

for (int col = 0; col < n; col += 4) {

float acc0 = 0;

float acc1 = 0;

float acc2 = 0;

float acc3 = 0;

// Compute each block

for (int ch = 0; ch < k;) {

// pointer of the int8 activation

const signed char *a_int8 = &A->int8_data_ptr[row * k + ch];

// pointer of the int4 weights

uint8_t *w0_int4 = &B->int4_data_ptr[(col * k + ch) / 2];

uint8_t *w1_int4 = &B->int4_data_ptr[((col + 1) * k + ch) / 2];

uint8_t *w2_int4 = &B->int4_data_ptr[((col + 2) * k + ch) / 2];

uint8_t *w3_int4 = &B->int4_data_ptr[((col + 3) * k + ch) / 2];

// scale of activation

float s_a = params->A_scales[(row * k + ch) / block_size];

// scale of weight

float s_w0 = params->scales[(col * k + ch) / block_size];

float s_w1 = params->scales[((col + 1) * k + ch) / block_size];

float s_w2 = params->scales[((col + 2) * k + ch) / block_size];

float s_w3 = params->scales[((col + 3) * k + ch) / block_size];

#ifdef QM_ARM

// order of weights with QM_ARM:

// origin order: (w0,w1), (w2,w3), (w4,w5), (w6,w7), (w8, w9), ... (w30,w31)

// QM_ARM order: (w0,w16),(w1,w17),(w2,w18),(w3,w19),(w4, w20),... (w15,w31)

// |--|

// 4 bits

// |------|

// 8 bits (byte)

// low|----------------------------------------------------------|high

// 0 128 bit 127

// process 16 bytes of weigths (128 bit) = 1 block for each of unrolled `col`

// intermediate variable to store sum of integer multiplication and accumulation

int intermediate_sum_0 = 0, intermediate_sum_1 = 0, intermediate_sum_2 = 0, intermediate_sum_3 = 0;

for (int qj = 0; qj < 16; qj++) {

// TODO: decode a packed byte into two int8 in the range of (-8, 7)

uint8_t packed_int4_0 = w0_int4[qj];

uint8_t packed_int4_1 = w1_int4[qj];

uint8_t packed_int4_2 = w2_int4[qj];

uint8_t packed_int4_3 = w3_int4[qj];

signed char w_de_0_0 = (packed_int4_0 & 0x0F) - 8.0;

signed char w_de_0_16 = (packed_int4_0 >> 4) - 8.0;

signed char w_de_1_0 = (packed_int4_1 & 0x0F) - 8.0;

signed char w_de_1_16 = (packed_int4_1 >> 4) - 8.0;

signed char w_de_2_0 = (packed_int4_2 & 0x0F) - 8.0;

signed char w_de_2_16 = (packed_int4_2 >> 4) - 8.0;

signed char w_de_3_0 = (packed_int4_3 & 0x0F) - 8.0;

signed char w_de_3_16 = (packed_int4_3 >> 4) - 8.0;

  

intermediate_sum_0 += a_int8[qj] * w_de_0_0;

intermediate_sum_0 += a_int8[qj + 16] * w_de_0_16;

intermediate_sum_1 += a_int8[qj] * w_de_1_0;

intermediate_sum_1 += a_int8[qj + 16] * w_de_1_16;

intermediate_sum_2 += a_int8[qj] * w_de_2_0;

intermediate_sum_2 += a_int8[qj + 16] * w_de_2_16;

intermediate_sum_3 += a_int8[qj] * w_de_3_0;

intermediate_sum_3 += a_int8[qj + 16] * w_de_3_16;

// TODO: int8 multiply and accumulate operation

}

// dequantize the sum into floating point

acc0 += (float)intermediate_sum_0 * s_a * s_w0;

acc1 += (float)intermediate_sum_1 * s_a * s_w1;

acc2 += (float)intermediate_sum_2 * s_a * s_w2;

acc3 += (float)intermediate_sum_3 * s_a * s_w3;

ch += block_size;

#endif

}

C->data_ptr[row * n + col] = acc0;

C->data_ptr[row * n + col + 1] = acc1;

C->data_ptr[row * n + col + 2] = acc2;

C->data_ptr[row * n + col + 3] = acc3;

}

}

};

} // namespace matmul
```

## multithreading
```cpp
#include <assert.h>

#include <pthread.h>

#include <stdio.h>

  

#include <cmath>

#include <cstdlib>

  

#include "../matmul.h"

#include "common.h"

struct multithreading_thread_args {

int start, end;

const struct matmul_params* params;

};

static void* multithreading_worker_func(void* args) {

struct multithreading_thread_args* mat_args = (struct multithreading_thread_args*)args;

const struct matmul_params* params = mat_args->params;

const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;

const int block_size = params->block_size;

  

int m = C->row, n = C->column, k = A->column;

// A: m x k; B: n x k; C: m x n

for (int row = 0; row < m; row++) {

for (int col = mat_args->start; col < mat_args->end; col++) {

float acc = 0;

// Compute each block

for (int ch = 0; ch < k;) {

// pointer of the int4 weights

uint8_t* w_int4 = &B->int4_data_ptr[(col * k + ch) / 2];

// pointer of the int8 activation

const signed char* a_int8 = &A->int8_data_ptr[row * k + ch];

// scale of weight

float s_w = params->scales[(col * k + ch) / block_size];

// scale of activation

float s_a = params->A_scales[(row * k + ch) / block_size];

#ifdef QM_ARM

// order of weights with QM_ARM:

// origin order: (w0,w1), (w2,w3), (w4,w5), (w6,w7), (w8, w9), ... (w30,w31)

// QM_ARM order: (w0,w16),(w1,w17),(w2,w18),(w3,w19),(w4, w20),... (w15,w31)

// |--|

// 4 bits

// |------|

// 8 bits (byte)

// low|----------------------------------------------------------|high

// 0 128 bit 127

// process 16 bytes of weigths (128 bit) = 1 block

// intermediate variable to store sum of integer multiplication and accumulation

int intermediate_sum = 0;

// process 16 bytes of weigths (128 bit)

for (int qj = 0; qj < 16; qj++) {

// decode a packed byte into two int8 in the range of (-8, 7)

uint8_t packed_int4_0 = w_int4[qj];

signed char w_de_0 = (packed_int4_0 & 0x0F) - 8.0;

signed char w_de_16 = (packed_int4_0 >> 4) - 8.0;

// int8 multiply and accumulate operation

intermediate_sum += a_int8[qj] * w_de_0;

intermediate_sum += a_int8[qj + 16] * w_de_16;

}

// dequantize the sum into floating point

acc += (float)intermediate_sum * s_a * s_w;

ch += block_size;

#endif

}

C->data_ptr[row * n + col] = acc;

}

}

return NULL;

}

  

namespace matmul {

void MatmulOperator::mat_mul_multithreading(struct matmul_params* params) {

const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;

const int block_size = params->block_size;

  

quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

  

int m = C->row, n = C->column, k = A->column;

  

const int num_thread = 4;

pthread_t thread_pool[num_thread];

struct multithreading_thread_args threads_args[num_thread];

  

// TODO: Thread creation

int cols_per_thread = n / num_thread;

int extra_cols = n % num_thread;

  

int start_col = 0;

  

for (int i = 0; i < num_thread; i++) {

int end_col = start_col + cols_per_thread + (i < extra_cols);

  

threads_args[i].start = start_col;

threads_args[i].end = end_col;

threads_args[i].params = params;

  

if (pthread_create(&thread_pool[i], NULL, multithreading_worker_func, &threads_args[i]) != 0) {

perror("pthread_create failed");

exit(EXIT_FAILURE);

}

  

start_col = end_col;

}

// TODO: Join threads

for(int i = 0; i < num_thread; i++) {

pthread_join(thread_pool[i], NULL);

}

  

};

} // namespace matmul
```
## loop_unrolling & multithreading
```cpp
#include <assert.h>

#include <pthread.h>

#include <stdio.h>

  

#include <cmath>

#include <cstdlib>

  

#include "../matmul.h"

#include "common.h"

struct multithreading_loop_unrolling_thread_args {

int start, end;

const struct matmul_params *params;

};

static void *multithreading_loop_unrolling_worker_func(void *args) {

struct multithreading_loop_unrolling_thread_args *mat_args =

(struct multithreading_loop_unrolling_thread_args *)args;

const struct matmul_params *params = mat_args->params;

const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;

const int block_size = params->block_size;

  

int m = C->row, n = C->column, k = A->column;

// A: m x k; B: n x k; C: m x n

for (int row = 0; row < m; row++) {

for (int col = mat_args->start; col < mat_args->end; col += 4) {

float acc0 = 0;

float acc1 = 0;

float acc2 = 0;

float acc3 = 0;

// Compute each block

for (int ch = 0; ch < k;) {

// pointer of the int8 activation

const signed char *a_int8 = &A->int8_data_ptr[row * k + ch];

// pointer of the int4 weights

uint8_t *w0_int4 = &B->int4_data_ptr[(col * k + ch) / 2];

uint8_t *w1_int4 = &B->int4_data_ptr[((col + 1) * k + ch) / 2];

uint8_t *w2_int4 = &B->int4_data_ptr[((col + 2) * k + ch) / 2];

uint8_t *w3_int4 = &B->int4_data_ptr[((col + 3) * k + ch) / 2];

// scale of activation

float s_a = params->A_scales[(row * k + ch) / block_size];

// scale of weight

float s_w0 = params->scales[(col * k + ch) / block_size];

float s_w1 = params->scales[((col + 1) * k + ch) / block_size];

float s_w2 = params->scales[((col + 2) * k + ch) / block_size];

float s_w3 = params->scales[((col + 3) * k + ch) / block_size];

#ifdef QM_ARM

// order of weights with QM_ARM:

// origin order: (w0,w1), (w2,w3), (w4,w5), (w6,w7), (w8, w9), ... (w30,w31)

// QM_ARM order: (w0,w16),(w1,w17),(w2,w18),(w3,w19),(w4, w20),... (w15,w31)

// |--|

// 4 bits

// |------|

// 8 bits (byte)

// low|----------------------------------------------------------|high

// 0 128 bit 127

// process 16 bytes of weigths (128 bit) = 1 block for each of unrolled `col`

// intermediate variable to store sum of integer multiplication and accumulation

int intermediate_sum0 = 0, intermediate_sum1 = 0, intermediate_sum2 = 0, intermediate_sum3 = 0;

for (int qj = 0; qj < 16; qj++) {

// TODO: decode a packed byte into two int8 in the range of (-8, 7)

uint8_t packed_int4_0 = w0_int4[qj];

uint8_t packed_int4_1 = w1_int4[qj];

uint8_t packed_int4_2 = w2_int4[qj];

uint8_t packed_int4_3 = w3_int4[qj];

signed char w_de_0_0 = (packed_int4_0 & 0x0F) - 8.0;

signed char w_de_0_16 = (packed_int4_0 >> 4) - 8.0;

signed char w_de_1_0 = (packed_int4_1 & 0x0F) - 8.0;

signed char w_de_1_16 = (packed_int4_1 >> 4) - 8.0;

signed char w_de_2_0 = (packed_int4_2 & 0x0F) - 8.0;

signed char w_de_2_16 = (packed_int4_2 >> 4) - 8.0;

signed char w_de_3_0 = (packed_int4_3 & 0x0F) - 8.0;

signed char w_de_3_16 = (packed_int4_3 >> 4) - 8.0;

  

intermediate_sum0 += a_int8[qj] * w_de_0_0;

intermediate_sum0 += a_int8[qj + 16] * w_de_0_16;

intermediate_sum1 += a_int8[qj] * w_de_1_0;

intermediate_sum1 += a_int8[qj + 16] * w_de_1_16;

intermediate_sum2 += a_int8[qj] * w_de_2_0;

intermediate_sum2 += a_int8[qj + 16] * w_de_2_16;

intermediate_sum3 += a_int8[qj] * w_de_3_0;

intermediate_sum3 += a_int8[qj + 16] * w_de_3_16;

// TODO: int8 multiply and accumulate operation

}

// dequantize the sum into floating point

acc0 += (float)intermediate_sum0 * s_a * s_w0;

acc1 += (float)intermediate_sum1 * s_a * s_w1;

acc2 += (float)intermediate_sum2 * s_a * s_w2;

acc3 += (float)intermediate_sum3 * s_a * s_w3;

ch += block_size;

#endif

#ifdef QM_x86

// scales of the second block

float s_w0_2nd = params->scales[(col * k + ch) / block_size + 1];

float s_w1_2nd = params->scales[((col + 1) * k + ch) / block_size + 1];

float s_w2_2nd = params->scales[((col + 2) * k + ch) / block_size + 1];

float s_w3_2nd = params->scales[((col + 3) * k + ch) / block_size + 1];

float s_a_2nd = params->A_scales[(row * k + ch) / block_size + 1];

// order of weights with QM_x86:

// origin order: (w0,w1), (w2,w3), (w4,w5), (w6,w7), (w8, w9), ... (w62,w63)

// QM_ARM order: (w0,w32),(w1,w33),(w2,w34),(w3,w35),(w4, w36),... (w31,w63)

// |--|

// 4 bits

// |------|

// 8 bits (byte)

// low|----------------------------------------------------------|high

// 0 256 bit

// process 32 bytes of weigths (256 bit) = 2 blocks for each of unrolled `col`

// intermediate variable to store sum of integer multiplication and accumulation

int intermediate_sum0 = 0, intermediate_sum1 = 0, intermediate_sum2 = 0, intermediate_sum3 = 0;

int intermediate_sum0_2nd = 0, intermediate_sum1_2nd = 0, intermediate_sum2_2nd = 0,

intermediate_sum3_2nd = 0;

for (int qj = 0; qj < 32; qj++) {

// TODO: decode a packed byte into two int8 in the range of (-8, 7)

  

// TODO: int8 multiply and accumulate operation

}

// dequantize the sum into floating point

acc0 += (float)intermediate_sum0 * s_a * s_w0;

acc0 += (float)intermediate_sum0_2nd * s_a_2nd * s_w0_2nd;

acc1 += (float)intermediate_sum1 * s_a * s_w1;

acc1 += (float)intermediate_sum1_2nd * s_a_2nd * s_w1_2nd;

acc2 += (float)intermediate_sum2 * s_a * s_w2;

acc2 += (float)intermediate_sum2_2nd * s_a_2nd * s_w2_2nd;

acc3 += (float)intermediate_sum3 * s_a * s_w3;

acc3 += (float)intermediate_sum3_2nd * s_a_2nd * s_w3_2nd;

ch += block_size * 2;

#endif

}

C->data_ptr[row * n + col] = acc0;

C->data_ptr[row * n + col + 1] = acc1;

C->data_ptr[row * n + col + 2] = acc2;

C->data_ptr[row * n + col + 3] = acc3;

}

}

return NULL;

}

  

namespace matmul {

void MatmulOperator::mat_mul_multithreading_loop_unrolling(struct matmul_params *params) {

const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;

const int block_size = params->block_size;

assert(params->block_size % 32 == 0); // support block size to be multiples of 32

assert(A->row == C->row); // support block size to be multiples of 32

  

quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

  

int m = C->row, n = C->column, k = A->column;

  

const int num_thread = 4;

pthread_t thread_pool[num_thread];

struct multithreading_loop_unrolling_thread_args threads_args[num_thread];

assert(params->block_size == 32); // support block size 32 for now

  

// TODO: Thread creation

int cols_per_thread = n / num_thread;

int extra_cols = n % num_thread;

  

int start_col = 0;

  

for (int i = 0; i < num_thread; i++) {

int end_col = start_col + cols_per_thread + (i < extra_cols);

  

threads_args[i].start = start_col;

threads_args[i].end = end_col;

threads_args[i].params = params;

  

if (pthread_create(&thread_pool[i], NULL, multithreading_loop_unrolling_worker_func, &threads_args[i]) != 0) {

perror("pthread_create failed");

exit(EXIT_FAILURE);

}

  

start_col = end_col;

}

// TODO: Join threads

for(int i = 0; i < num_thread; i++) {

pthread_join(thread_pool[i], NULL);

}

};

} // namespace matmul
```
## simd_programming
```cpp
#include <assert.h>

#include <pthread.h>

#include <stdio.h>

  

#include <cmath>

#include <cstdlib>

  

#include "../matmul.h"

#include "common.h"

  

#ifdef QM_ARM

#include <arm_neon.h>

#endif

#ifdef QM_x86

#include <immintrin.h>

#endif

namespace matmul {

void MatmulOperator::mat_mul_simd_programming(struct matmul_params *params) {

const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;

const int block_size = params->block_size; // block_size = 32

  

quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

  

int m = C->row, n = C->column, k = A->column;

// A: m x k; B: n x k; C: m x n

for (int row = 0; row < m; row++) {

for (int col = 0; col < n; col++) {

#ifdef QM_ARM

// order of weights with QM_ARM:

// origin order: (w0,w1), (w2,w3), (w4,w5), (w6,w7), (w8, w9), ... (w30,w31)

// QM_ARM order: (w0,w16),(w1,w17),(w2,w18),(w3,w19),(w4, w20),... (w15,w31)

// |--|

// 4 bits

// |------|

// 8 bits (byte)

// low|----------------------------------------------------------|high

// 0 128 bit 127

/* --- Added: 使用 Neon 向量初始化累加器 --- */

float32x4_t sumv0 = vdupq_n_f32(0.0f);

// pointer of the int4 weights

const unsigned char *w_start = &B->int4_data_ptr[col * k / 2];

// pointer of the int8 activation

const signed char *a_start = &A->int8_data_ptr[row * k];

// scale of activation

float *s_a = &params->A_scales[row * k / 32];

// scale of weight

float *s_w = &params->scales[col * k / 32];

  

const int num_block = k / block_size;

// Compute each block

for (int q = 0; q < num_block; q++) {

/* --- Added: 加载 16 字节 (即 32 个 4-bit 权重) --- */

const uint8x16_t w0 = vld1q_u8(w_start);

w_start += 16;

  

/*

我们将使用 ARM Neon Intrinsics 加速解码和点积计算：

*/

/* --- Added: 步骤1，提取低4位 --- */

const uint8x16_t mask_low4bit = vdupq_n_u8(0xF);

uint8x16_t lower_u8 = vandq_u8(w0, mask_low4bit);

  

/* --- Added: 步骤2，右移4位获得高4位 --- */

uint8x16_t upper_u8 = vshrq_n_u8(w0, 4);

  

/* --- Added: 步骤3，reinterpret为 int8 类型 --- */

int8x16_t lower = vreinterpretq_s8_u8(lower_u8);

int8x16_t upper = vreinterpretq_s8_u8(upper_u8);

  

/* --- Added: 步骤4，调整数值范围从 (0,15) 到 (-8,7) --- */

lower = vsubq_s8(lower, vdupq_n_s8(8));

upper = vsubq_s8(upper, vdupq_n_s8(8));

  

/* --- Added: 加载32个 8-bit 激活值 --- */

const int8x16_t a0 = vld1q_s8(a_start);

const int8x16_t a1 = vld1q_s8(a_start + 16);

a_start += 32;

  

/* --- Added: 步骤5，使用 vdotq_s32 计算点积 ---

分别计算 lower 部分与 a0 的点积，再加上 upper 与 a1 的点积 */

int32x4_t int_sum0 = vdupq_n_s32(0);

int_sum0 = vdotq_s32(int_sum0, a0, lower);

int_sum0 = vdotq_s32(int_sum0, a1, upper);

  

/* --- Added: 步骤6，获取当前 block 的缩放因子并更新指针 --- */

float s_0 = (*s_a) * (*s_w);

s_a++;

s_w++;

  

/* --- Added: 步骤7，将 int_sum0 转换为浮点数后乘以缩放因子累加到 sumv0 --- */

sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(int_sum0), s_0);

}

/* --- Added: 将 Neon 向量中所有元素求和得到标量结果，存入 C --- */

C->data_ptr[row * n + col] = vaddvq_f32(sumv0);

#endif

}

}

};

} // namespace matmul
```


## all_techniques
```cpp
#include <assert.h>

#include <pthread.h>

#include <stdio.h>

  

#include <cmath>

#include <cstdlib>

  

#include "../matmul.h"

#include "common.h"

  

#ifdef QM_ARM

#include <arm_neon.h>

#endif

#ifdef QM_x86

#include <immintrin.h>

#endif

struct w4a8_thread_args {

int start_j, end_j;

const struct matmul_params *params;

};

static void *all_techniques_worker_func(void *args) {

struct w4a8_thread_args *mat_args = (struct w4a8_thread_args *)args;

const struct matmul_params *params = mat_args->params;

const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;

int n = params->C.column, m = params->C.row, k = params->A.column, block_size = params->block_size;

const int num_block = k / block_size; // block_size = 32

  

for (int row = 0; row < m; row++) {

for (int col = mat_args->start_j; col < mat_args->end_j; col++) {

#ifdef QM_ARM

// order of weights with QM_ARM:

// origin order: (w0,w1), (w2,w3), (w4,w5), (w6,w7), (w8, w9), ... (w30,w31)

// QM_ARM order: (w0,w16),(w1,w17),(w2,w18),(w3,w19),(w4, w20),... (w15,w31)

// |--|

// 4 bits

// |------|

// 8 bits (byte)

// low|----------------------------------------------------------|high

// 0 128 bit 127

float32x4_t sumv0 = vdupq_n_f32(0.0f);

/*

float32x4_t sumv1 = vdupq_n_f32(0.0f);

float32x4_t sumv2 = vdupq_n_f32(0.0f);

float32x4_t sumv3 = vdupq_n_f32(0.0f);

*/

// pointer of the int4 weights

const unsigned char *w_start = &params->B.int4_data_ptr[col * k / 2];

// pointer of the int8 activation

const signed char *a_start = &params->A.int8_data_ptr[row * k];

// scale of activation

float *s_a = &params->A_scales[row * k / 32];

// scale of weight

float *s_w = &params->scales[col * k / 32];

  

// process four blocks each iteration

for (int q = 0; q < num_block; q += 4) {

// load 32x4bit (16 bytes) weight

const uint8x16_t w0 = vld1q_u8(w_start); // 32 4bit weight

const uint8x16_t w1 = vld1q_u8(w_start + 16); // 32 4bit weight

const uint8x16_t w2 = vld1q_u8(w_start + 32); // 32 4bit weight

const uint8x16_t w3 = vld1q_u8(w_start + 48); // 32 4bit weight

w_start += 64;

  

// TODO: decode each uint8x16_t weight vector into the lower and upper half of the weights as int8x16_t

// Hint:

// (1) use `vandq_u8` with the mask_low4bit to get the lower half

// (2) use `vshrq_n_u8` to right shift 4 bits and get the upper half

// (3) use `vreinterpretq_s8_u8` to interpret the vector as int8

// lowbit mask

const uint8x16_t mask_low4bit = vdupq_n_u8(0xF);

uint8x16_t lower_u8_0 = vandq_u8(w0, mask_low4bit);

uint8x16_t lower_u8_1 = vandq_u8(w1, mask_low4bit);

uint8x16_t lower_u8_2 = vandq_u8(w2, mask_low4bit);

uint8x16_t lower_u8_3 = vandq_u8(w3, mask_low4bit);

  

uint8x16_t upper_u8_0 = vshrq_n_u8(w0, 4);

uint8x16_t upper_u8_1 = vshrq_n_u8(w1, 4);

uint8x16_t upper_u8_2 = vshrq_n_u8(w2, 4);

uint8x16_t upper_u8_3 = vshrq_n_u8(w3, 4);

  

int8x16_t lower_0 = vreinterpretq_s8_u8(lower_u8_0);

int8x16_t upper_0 = vreinterpretq_s8_u8(upper_u8_0);

int8x16_t lower_1 = vreinterpretq_s8_u8(lower_u8_1);

int8x16_t upper_1 = vreinterpretq_s8_u8(upper_u8_1);

int8x16_t lower_2 = vreinterpretq_s8_u8(lower_u8_2);

int8x16_t upper_2 = vreinterpretq_s8_u8(upper_u8_2);

int8x16_t lower_3 = vreinterpretq_s8_u8(lower_u8_3);

int8x16_t upper_3 = vreinterpretq_s8_u8(upper_u8_3);

  

// TODO: apply zero_point to weights and convert the range from (0, 15) to (-8, 7)

// Hint: using `vsubq_s8` to the lower-half and upper-half vectors of weights

const int8x16_t offsets = vdupq_n_s8(8);

  

lower_0 = vsubq_s8(lower_0, offsets);

upper_0 = vsubq_s8(upper_0, offsets);

lower_1 = vsubq_s8(lower_1, offsets);

upper_1 = vsubq_s8(upper_1, offsets);

lower_2 = vsubq_s8(lower_2, offsets);

upper_2 = vsubq_s8(upper_2, offsets);

lower_3 = vsubq_s8(lower_3, offsets);

upper_3 = vsubq_s8(upper_3, offsets);

  

// load 128 8-bit activation

const int8x16_t a0 = vld1q_s8(a_start);

const int8x16_t a1 = vld1q_s8(a_start + 16);

const int8x16_t a2 = vld1q_s8(a_start + 32);

const int8x16_t a3 = vld1q_s8(a_start + 48);

const int8x16_t a4 = vld1q_s8(a_start + 64);

const int8x16_t a5 = vld1q_s8(a_start + 80);

const int8x16_t a6 = vld1q_s8(a_start + 96);

const int8x16_t a7 = vld1q_s8(a_start + 112);

a_start += 128;

  

// TODO: perform dot product and store the result into the intermediate sum, int_sum0

// Hint: use `vdotq_s32` and store the sum for each block in int_sum{0-3}

int32x4_t int_sum0, int_sum1, int_sum2, int_sum3;

int_sum0 = vdupq_n_s32(0);

int_sum1 = vdupq_n_s32(0);

int_sum2 = vdupq_n_s32(0);

int_sum3 = vdupq_n_s32(0);

  

int_sum0 = vdotq_s32(int_sum0, a0, lower_0);

int_sum0 = vdotq_s32(int_sum0, a1, upper_0);

int_sum1 = vdotq_s32(int_sum1, a2, lower_1);

int_sum1 = vdotq_s32(int_sum1, a3, upper_1);

int_sum2 = vdotq_s32(int_sum2, a4, lower_2);

int_sum2 = vdotq_s32(int_sum2, a5, upper_2);

int_sum3 = vdotq_s32(int_sum3, a6, lower_3);

int_sum3 = vdotq_s32(int_sum3, a7, upper_3);

  
  

float s_0 = *s_a++ * *s_w++;

float s_1 = *s_a++ * *s_w++;

float s_2 = *s_a++ * *s_w++;

float s_3 = *s_a++ * *s_w++;

  

sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(int_sum0), s_0);

sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(int_sum1), s_1);

sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(int_sum2), s_2);

sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(int_sum3), s_3);

}

params->C.data_ptr[row * n + col] = vaddvq_f32(sumv0);

#endif

#ifdef QM_x86

// order of weights with QM_x86:

// origin order: (w0,w1), (w2,w3), (w4,w5), (w6,w7), (w8, w9), ... (w62,w63)

// QM_ARM order: (w0,w32),(w1,w33),(w2,w34),(w3,w35),(w4, w36),... (w31,w63)

// |--|

// 4 bits

// |------|

// 8 bits (byte)

// low|----------------------------------------------------------|high

// 0 256 bit

__m256 accumulator = _mm256_setzero_ps();

float *s_ptr = &params->scales[col * k / 32];

float *sa_ptr = &params->A_scales[row * k / 32];

const __m256i *w_start = (__m256i *)&B->int4_data_ptr[col * k / 2];

const __m256i *a_start = (__m256i *)&A->int8_data_ptr[row * k];

const int num_block = k / block_size;

// Compute four blocks = 128 4-bit weights in each iteration

for (int q = 0; q < num_block; q += 4) {

// lowbit mask

const __m256i lowMask = _mm256_set1_epi8(0xF);

  

// TODO: Unpack 128 4-bit (two __mm256i) weights into 128 8-bit (four __mm256i)

// (1) load 256 bit from w_strat with _mm256_loadu_si256

// (2) use _mm256_and_si256 and lowMask to extract the lower half of wegihts

// (3) use _mm256_srli_epi16 and _mm256_and_si256 with lowMask to extract the upper half of weights

__m256i raw_w = _mm256_loadu_si256(w_start);

__m256i raw_w_next = _mm256_loadu_si256(w_start + 1);

  

// TODO: apply zero_point to weights and convert the range from (0, 15) to (-8, 7)

// Hint: using `_mm256_sub_epi8` to the lower-half and upper-half vectors of weights

// Note: For the first two blocks, store the lower half and upper half of weights into `w_0` and

// `w_128`, respectively For the last two blocks store the lower half and upper half of weights into

// `w_0_next` and `w_128_next`, respectively

const __m256i zero_point = _mm256_set1_epi8(8);

__m256i w_0, w_128, w_0_next, w_128_next;

  

// Perform int8 dot product with _mm256_maddubs_epi16

/* Syntax of _mm256_maddubs_epi16:

__m256i _mm256_maddubs_epi16(__m256i s1, __m256i s2): Multiplies vertically each unsigned byte of

source vector s1 with the corresponding signed byte of source vector s2, producing intermediate,

signed 16-bit integers. Each adjacent pair of signed words is added, and the saturated result is

packed to the destination vector.

*/

// To utilize _mm256_maddubs_epi16 which only takes unsigned s1, we need to:

// (1) Get the absolute values of weights (for both lower and upper halves)

// (2) Change the sign of activation (a0-a31 and a32-a63) depending on the sign of corresponding weights

// (stored as another variable) (3) Perform dot product with _mm256_maddubs_epi16 and store the lower

// and upper halves sum in `dot` and `dot2`

__m256i dot, dot2, dot3, dot4;

// Get absolute values of x vectors

const __m256i ax = _mm256_sign_epi8(w_0, w_0);

const __m256i ax_next = _mm256_sign_epi8(w_0_next, w_0_next);

const __m256i ax2 = _mm256_sign_epi8(w_128, w_128);

const __m256i ax2_next = _mm256_sign_epi8(w_128_next, w_128_next);

// Load activation

__m256i activation = a_start[0];

__m256i activation2 = a_start[1];

__m256i activation_next = a_start[2];

__m256i activation2_next = a_start[3];

// Sign the values of the y vectors

const __m256i sy = _mm256_sign_epi8(activation, w_0);

const __m256i sy_next = _mm256_sign_epi8(activation_next, w_0_next);

const __m256i sy2 = _mm256_sign_epi8(activation2, w_128);

const __m256i sy2_next = _mm256_sign_epi8(activation2_next, w_128_next);

  

// TODO: Perform int8 dot product with `_mm256_maddubs_epi16`

// Hint: use `_mm256_maddubs_epi16` to complete the following computation

// dot = ax * sy

// dot2 = ax2 * sy2

// dot3 = ax_next * sy_next

// dot4 = ax2_next * sy2_next

  

// Convert int32 vectors to floating point vectors

const __m256i ones = _mm256_set1_epi16(1);

const __m256i summed_pairs = _mm256_madd_epi16(ones, dot);

const __m256i summed_pairs2 = _mm256_madd_epi16(ones, dot2);

const __m256i summed_pairs3 = _mm256_madd_epi16(ones, dot3);

const __m256i summed_pairs4 = _mm256_madd_epi16(ones, dot4);

__m256 intermediate = _mm256_cvtepi32_ps(summed_pairs);

__m256 intermediate2 = _mm256_cvtepi32_ps(summed_pairs2);

__m256 intermediate3 = _mm256_cvtepi32_ps(summed_pairs3);

__m256 intermediate4 = _mm256_cvtepi32_ps(summed_pairs4);

  

// Create vectors for scales and apply them to intermediate results

__m256 v_s = _mm256_set1_ps(s_ptr[0] * sa_ptr[0]);

__m256 v_s2 = _mm256_set1_ps(s_ptr[1] * sa_ptr[1]);

__m256 v_s3 = _mm256_set1_ps(s_ptr[2] * sa_ptr[2]);

__m256 v_s4 = _mm256_set1_ps(s_ptr[3] * sa_ptr[3]);

accumulator = _mm256_fmadd_ps(intermediate, v_s, accumulator);

accumulator = _mm256_fmadd_ps(intermediate2, v_s2, accumulator);

accumulator = _mm256_fmadd_ps(intermediate3, v_s3, accumulator);

accumulator = _mm256_fmadd_ps(intermediate4, v_s4, accumulator);

s_ptr += 4;

sa_ptr += 4;

w_start += 2;

a_start += 4;

}

float *ptr = (float *)&accumulator;

C->data_ptr[row * n + col] = ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7];

#endif

}

}

  

return NULL;

}

  

namespace matmul {

void MatmulOperator::mat_mul_all_techniques(struct matmul_params *params) {

int i, j, k;

const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;

const int block_size = params->block_size;

float *scale = params->scales, *offset = params->offset;

  

assert(params->block_size % 32 == 0); // support block size to be multiples of 32

assert(A->row == C->row); // support block size to be multiples of 32

  

quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

  

const int num_thread = 8;

pthread_t thread_pool[num_thread];

struct w4a8_thread_args threads_args[num_thread];

assert(params->block_size == 32); // support block size 32 for now

  

// TODO: Thread creation

int m = C->row, n = C->column;

int cols_per_thread = n / num_thread;

int extra_cols = n % num_thread;

  

int start_col = 0;

  

for (int i = 0; i < num_thread; i++) {

int end_col = start_col + cols_per_thread + (i < extra_cols);

  

threads_args[i].start_j = start_col;

threads_args[i].end_j = end_col;

threads_args[i].params = params;

  

if (pthread_create(&thread_pool[i], NULL, all_techniques_worker_func, &threads_args[i]) != 0) {

perror("pthread_create failed");

exit(EXIT_FAILURE);

}

  

start_col = end_col;

}

// TODO: Join threads

for(int i = 0; i < num_thread; i++) {

pthread_join(thread_pool[i], NULL);

}

};

} // namespace matmul
```
# Results
1. reference:![[Pasted image 20250303225640.png]]
2. loop_unrolling:![[Pasted image 20250303225706.png]]
3. multithreading(based on columns): ![[Pasted image 20250304170701.png]]
4. loop_unrolling & multithreading:![[Pasted image 20250304212453.png]]
5. simd_programming:![[Pasted image 20250304200226.png]]
6. all_techniques:![[Pasted image 20250305001820.png]]
# Analysis
| Section                        | Total time (ms) | Average time (ms) | GOPs      | Total time Improvement (%) | Average time Improvement (%) | GOPs Improvement (%) |
| :-----------------------------: | :-------------: | :---------------: | :-------: | :------------------------: | :-------------------------: | :-------------------: |
| reference                      |    2690.630     |     269.063       |   0.97429 |            0%              |            0%               |          0%           |
| loop_unrolling                 |    2068.798     |     206.879       |   1.26713 |         -23.45%            |         -23.21%             |        30.07%         |
| simd_programming               |    1275.758     |     127.575       |   2.05481 |         -52.67%            |         -52.47%             |       111.17%         |
| multithreading                 |     935.601     |      93.560       |   2.80188 |         -65.30%            |         -65.24%             |       187.71%         |
| multithreading_loop_unrolling  |     653.355     |      65.335       |   4.01228 |         -75.68%            |         -75.65%             |       313.56%         |
| all_techniques                 |     278.998     |      27.899       |   9.39591 |         -89.64%            |         -89.62%             |       864.86%         |

### **1. Which single method is the most effective?**

From the table, we can observe the reduction in **Total time (ms)** and the increase in **GOPs** for each optimization technique compared to the `reference` implementation:

- **Loop Unrolling**: Reduces total time by **23.45%**, increases GOPs by **30.07%**.
- **SIMD Programming**: Reduces total time by **52.67%**, increases GOPs by **111.17%**.
- **Multithreading**: Reduces total time by **65.30%**, increases GOPs by **187.71%**.

Among these, **Multithreading** shows the most significant reduction in execution time (**65.30%**) and the highest increase in GOPs (**187.71%**).

 **Conclusion:** Among single optimization techniques, `Multithreading` is the most effective.

---

### **2. Does the efficiency improvement follow a multiplicative relationship when combining methods?**

If the effects of different optimizations were **completely independent and linearly stackable**, we would expect their performance improvements to follow a **multiplicative** pattern. However, analyzing the actual results:

1. **Multithreading + Loop Unrolling**:
    
    - `Multithreading` alone increases GOPs by **187.71%**.
    - `Loop Unrolling` alone increases GOPs by **30.07%**.
    - Expected improvement (multiplicative): **(1 + 1.8771) * (1 + 0.3007) - 1 = 338.36%**.
    - Actual improvement: **313.56%**.
    - **Observation**: The improvement is **less than expected** under a purely multiplicative assumption.
2. **All Techniques (Multithreading + SIMD + Loop Unrolling)**:
    
    - `Multithreading` boosts GOPs by **187.71%**, `SIMD` by **111.17%**, `Loop Unrolling` by **30.07%**.
    - Expected improvement (multiplicative): **(1 + 1.8771) * (1 + 1.1117) * (1 + 0.3007) - 1 = 1004.4%**.
    - Actual improvement: **864.86%**.
    - **Observation**: The combined effect is **lower than the theoretical multiplicative gain**.

This deviation from the multiplicative relationship is likely due to:

- **Overlapping effects**: Some optimizations target the same performance bottlenecks, leading to diminishing returns when combined. For instance, both SIMD and Loop Unrolling aim to reduce instruction overhead.
- **Hardware resource constraints**: Limitations such as CPU core count, vector register bandwidth, and memory bandwidth may bottleneck further improvements.
- **Conflicting optimizations**: Certain parallelization techniques (e.g., multithreading) might introduce overheads like synchronization costs, reducing the effectiveness of instruction-level optimizations like SIMD.

 **Conclusion:** Efficiency improvements **do not strictly follow a multiplicative relationship**. Instead, **they exhibit sub-multiplicative growth**, where combining multiple techniques still results in significant gains, but with diminishing returns.