/**
 * LLM Hardware Advisor — Memory Calculation Engine
 * All formulas run client-side with zero latency.
 */

// Quantization bit map — bits per weight for each format
export const QUANT_BITS = {
  'FP32':   32.0,
  'FP16':   16.0,
  'BF16':   16.0,
  'INT8':    8.0,
  'Q8_0':    8.0,
  'Q6_K':    6.5,
  'Q5_K_M':  5.5,
  'Q5_K_S':  5.0,
  'Q4_K_M':  4.5,
  'Q4_K_S':  4.25,
  'Q4_0':    4.0,
  'Q3_K_M':  3.35,
  'Q3_K_S':  3.0,
  'Q2_K':    2.5,
  'IQ2_XXS': 2.0,
  'AWQ':     4.0,
  'GPTQ':    4.0,
};

// Size as percentage of FP16
export const QUANT_SIZE_VS_FP16 = {};
for (const [k, v] of Object.entries(QUANT_BITS)) {
  QUANT_SIZE_VS_FP16[k] = Math.round((v / 16.0) * 100);
}

// Quant display metadata
export const QUANT_INFO = {
  'FP32':    { label: 'FP32',    desc: 'Full precision — maximum quality, huge size' },
  'FP16':    { label: 'FP16',    desc: 'Half precision — baseline quality' },
  'BF16':    { label: 'BF16',    desc: 'Brain Float 16 — same size as FP16, better training stability' },
  'INT8':    { label: 'INT8',    desc: 'Integer 8-bit — 50% size, minimal quality loss' },
  'Q8_0':    { label: 'Q8_0',    desc: 'GGUF 8-bit — near-lossless quantization' },
  'Q6_K':    { label: 'Q6_K',    desc: 'GGUF 6-bit — excellent quality, good compression' },
  'Q5_K_M':  { label: 'Q5_K_M',  desc: 'GGUF 5.5-bit — great balance of quality and size' },
  'Q5_K_S':  { label: 'Q5_K_S',  desc: 'GGUF 5-bit small — slightly smaller than Q5_K_M' },
  'Q4_K_M':  { label: 'Q4_K_M',  desc: 'GGUF 4.5-bit — best quality-to-size ratio for most users' },
  'Q4_K_S':  { label: 'Q4_K_S',  desc: 'GGUF 4.25-bit — slightly smaller, marginal quality loss' },
  'Q4_0':    { label: 'Q4_0',    desc: 'GGUF 4-bit basic — fast, legacy format' },
  'Q3_K_M':  { label: 'Q3_K_M',  desc: 'GGUF 3.35-bit — noticeable quality loss, very small' },
  'Q3_K_S':  { label: 'Q3_K_S',  desc: 'GGUF 3-bit small — aggressive compression' },
  'Q2_K':    { label: 'Q2_K',    desc: 'GGUF 2.5-bit — extreme compression, significant quality loss' },
  'IQ2_XXS': { label: 'IQ2_XXS', desc: 'GGUF 2-bit — maximum compression, for experimentation' },
  'AWQ':     { label: 'AWQ',     desc: 'Activation-aware 4-bit — GPU optimized quantization' },
  'GPTQ':    { label: 'GPTQ',    desc: 'GPU 4-bit — optimized for GPU inference' },
};

/**
 * Calculate model weight memory in GB
 * @param {number} paramsB - Parameters in billions
 * @param {string} quant - Quantization format key
 * @returns {number} Weight memory in GB
 */
export function calcWeightMemory(paramsB, quant) {
  const bitsPerWeight = QUANT_BITS[quant] || 16.0;
  return (paramsB * 1e9 * bitsPerWeight) / (8 * 1e9);
}

/**
 * Calculate KV-cache memory in GB
 * @param {object} model - Model record from models.json
 * @param {number} contextLen - Context length in tokens
 * @param {string} quant - Quantization format key
 * @param {number} [batch=1] - Batch size
 * @returns {number} KV-cache memory in GB
 */
export function calcKVCacheMemory(model, contextLen, quant, batch = 1) {
  const layers = model.layers || 32;
  const kvHeads = model.kv_heads || model.attention_heads || 32;
  const headDim = model.head_dim || 128;
  
  // fp16/bf16 uses 2 bytes per KV element, GGUF quants use 1 byte
  const isFP = ['FP32', 'FP16', 'BF16'].includes(quant);
  const bytesPerKV = isFP ? 2 : 1;
  
  return (2 * layers * kvHeads * headDim * contextLen * batch * bytesPerKV) / 1e9;
}

/**
 * Framework/runtime overhead in GB
 * Varies slightly by backend
 */
export const BACKEND_OVERHEAD = {
  'ollama':       0.75,
  'llama.cpp':    0.50,
  'vllm':         1.20,
  'transformers': 1.00,
  'exllamav2':    0.60,
};

/**
 * Calculate total VRAM required
 * @param {number} paramsB - Parameters in billions
 * @param {object} model - Model record
 * @param {string} quant - Quantization format
 * @param {number} contextLen - Context length in tokens
 * @param {string} [backend='ollama'] - Inference backend
 * @param {number} [batch=1] - Batch size
 * @returns {{ weightsGB: number, kvCacheGB: number, overheadGB: number, totalGB: number }}
 */
export function calcTotalVRAM(paramsB, model, quant, contextLen, backend = 'ollama', batch = 1) {
  const weightsGB = calcWeightMemory(paramsB, quant);
  const kvCacheGB = calcKVCacheMemory(model, contextLen, quant, batch);
  const overheadGB = BACKEND_OVERHEAD[backend] || 0.75;
  const totalGB = weightsGB + kvCacheGB + overheadGB;
  
  return { weightsGB, kvCacheGB, overheadGB, totalGB };
}

/**
 * Calculate disk space required for a quantized model
 * @param {number} paramsB - Parameters in billions
 * @param {string} quant - Quantization format
 * @returns {number} Disk space in GB (weights only, no KV cache)
 */
export function calcDiskSpace(paramsB, quant) {
  return calcWeightMemory(paramsB, quant);
}

/**
 * Calculate the maximum context length that fits in available VRAM
 * @param {number} paramsB - Parameters in billions
 * @param {object} model - Model record
 * @param {string} quant - Quantization format
 * @param {number} availableVRAM - Available VRAM in GB
 * @param {string} [backend='ollama'] - Inference backend
 * @returns {number} Maximum context length in tokens
 */
export function calcMaxContext(paramsB, model, quant, availableVRAM, backend = 'ollama') {
  const weightsGB = calcWeightMemory(paramsB, quant);
  const overheadGB = BACKEND_OVERHEAD[backend] || 0.75;
  const remainingGB = availableVRAM - weightsGB - overheadGB;
  
  if (remainingGB <= 0) return 0;
  
  const layers = model.layers || 32;
  const kvHeads = model.kv_heads || model.attention_heads || 32;
  const headDim = model.head_dim || 128;
  const isFP = ['FP32', 'FP16', 'BF16'].includes(quant);
  const bytesPerKV = isFP ? 2 : 1;
  
  const bytesPerToken = (2 * layers * kvHeads * headDim * bytesPerKV);
  const maxTokens = Math.floor((remainingGB * 1e9) / bytesPerToken);
  
  return Math.max(0, maxTokens);
}

/**
 * Calculate minimum system RAM for CPU-offload inference
 * @param {number} paramsB - Parameters in billions
 * @param {string} quant - Quantization format
 * @returns {number} Minimum RAM in GB (rounded up to nearest common RAM size)
 */
export function calcMinRAM(paramsB, quant) {
  const modelSize = calcWeightMemory(paramsB, quant);
  // Need ~1.3x model size for CPU inference (loading + runtime overhead)
  const needed = modelSize * 1.3;
  // Round up to nearest common RAM size
  const sizes = [8, 16, 32, 48, 64, 96, 128, 192, 256, 512];
  for (const s of sizes) {
    if (s >= needed) return s;
  }
  return Math.ceil(needed);
}
