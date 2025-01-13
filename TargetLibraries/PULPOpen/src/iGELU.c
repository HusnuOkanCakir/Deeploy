/* =====================================================================
 * Title:        iGELU.c
 * Description:
 *
 * $Date:        13.11.2023
 *
 * ===================================================================== */
/*
 * Copyright (C) 2020 ETH Zurich and University of Bologna.
 *
 * Author: Moritz Scherer, ETH Zurich
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"

void PULPiGELU_s8_s8(int8_t *data_in, int8_t *data_out, int32_t dataSize,
                     int8_t b, int16_t one, int32_t input_offset,
                     int32_t output_offset, int32_t *mul, int32_t *add,
                     int32_t *shift) {
    
    int8_t core_id    = pi_core_id();
    int8_t log2_core  = log2(NUM_CORES);

    printf("core id: %d\n", core_id);
    printf("core num: %d\n", NUM_CORES);
    printf("log2 core: %d\n", log2_core);

    int16_t chunk       = (dataSize >> log2_core) + ((dataSize & (NUM_CORES - 1)) != 0);
    int16_t chunk_start = MIN(chunk * core_id, dataSize);
    int16_t chunk_stop = MIN(chunk_start + chunk, dataSize + 1);

    printf("chunk: %d\n", chunk);
    printf("chunk_start: %d\n", chunk_start);
    printf("chunk_stop: %d\n", chunk_stop);

    int32_t rq_mul   = mul[0];
    int32_t rq_add   = add[0];
    int32_t rq_shift = shift[0];

    // #pragma unroll 8
    for (int i = chunk_start; i < chunk_stop; i++)
    {
        int32_t x = (int32_t)data_in[i] + input_offset;

        int32_t sign = (x > 0) - (x < 0);
        int32_t x_abs = sign * x;

        int32_t q;
        if (x_abs > -b)
            q = -b;
        else
            q = x_abs;

        int32_t d  = q + b;
        int32_t L  = sign * (-(d * d) + one);
        int32_t y  = x * ((one + L) >> 1);

        int32_t intermediate = (y * rq_mul) + rq_add;
        
        intermediate = ((intermediate + (1 << (rq_shift - 1))) >> rq_shift);

        intermediate += output_offset;
        if (intermediate > 127)  intermediate = 127;
        if (intermediate < -128) intermediate = -128;

        data_out[i] = (int8_t)intermediate;
        
    }
    // pi_cl_team_barrier();

}
