from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _PULPRQSiGELUTemplate(NodeTemplate):
    """
    Parallelized version of RQSiGELU for PULPOpen.
    Uses PULPiGELU_s8_s8(...) from iGELU.c to do the actual iGELU operation,
    splitting the data across cores.
    """

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(
        self,
        ctxt: NetworkContext,
        operatorRepresentation: OperatorRepresentation
    ) -> Tuple[NetworkContext, Dict, List[str]]:
        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])

        operatorRepresentation['input_offset'] = 0
        if hasattr(data_in, "_signed") and hasattr(data_in, "nLevels"):
            operatorRepresentation['input_offset'] = (data_in._signed == 0) * int(data_in.nLevels / 2)

        operatorRepresentation['output_offset'] = 0
        if hasattr(data_out, "_signed") and hasattr(data_out, "nLevels"):
            operatorRepresentation['output_offset'] = -(data_out._signed == 0) * int(data_out.nLevels / 2)

        
        operatorRepresentation['mul_scalar'] = operatorRepresentation['mul']
        operatorRepresentation['add_scalar'] = operatorRepresentation['add']
        operatorRepresentation['shift_scalar'] = operatorRepresentation['shift']

        return ctxt, operatorRepresentation, []




referenceTemplate = _PULPRQSiGELUTemplate("""
// Requantized iGELU parallelized for PULP (Name: ${nodeName}, Op: ${nodeOp})

int8_t core_id         = pi_core_id();
int8_t log2_core       = log2(NUM_CORES);
int16_t chunk       = (${size} >> log2_core) + (((${size}) & (NUM_CORES - 1)) != 0);
int16_t chunk_start = MIN(chunk * core_id, ${size});
int16_t chunk_stop  = MIN(chunk_start + chunk, ${size});

if (chunk_start < chunk_stop) {
    int32_t local_size = chunk_stop - chunk_start;

    PULPiGELU_s8_s8(
        &${data_in}[chunk_start],     // pointer to input slice
        &${data_out}[chunk_start],    // pointer to output slice
        local_size,                   // number of elements to process
        ${b},                         // 'b' parameter (int8)
        ${one},                       // 'one' parameter (int16)
        ${input_offset},              // input offset
        ${output_offset},             // output offset
        &${mul_scalar}[0],           // scalar mul pointer
        &${add_scalar}[0],           // scalar add pointer
        &${shift_scalar}[0]          // scalar shift pointer
    );
}

// pi_cl_team_barrier();

""")

