/**
 * LLM Hardware Advisor — Speed Estimation Engine
 * Approximate tokens/sec based on memory bandwidth and model size.
 * All estimates are labeled as approximations in the UI.
 */

/**
 * Estimate tokens per second for GPU inference
 * Formula: tokens/sec ≈ GPU_bandwidth_GBs / (model_size_GB × overhead_factor)
 * 
 * @param {number} bandwidthGBs - GPU memory bandwidth in GB/s
 * @param {number} modelSizeGB - Model weight size on device in GB
 * @param {boolean} [isCPU=false] - Whether this is CPU inference
 * @returns {number} Estimated tokens per second
 */
export function estimateTokensPerSec(bandwidthGBs, modelSizeGB, isCPU = false) {
  if (modelSizeGB <= 0) return 0;
  const overheadFactor = isCPU ? 2.5 : 1.2;
  return bandwidthGBs / (modelSizeGB * overheadFactor);
}

/**
 * Get speed estimate tiers for a model across reference hardware
 * @param {number} modelSizeGB - Model weight size in GB
 * @returns {{ low: {min:number, max:number, label:string}, mid: {min:number, max:number, label:string}, high: {min:number, max:number, label:string} }}
 */
export function getSpeedTiers(modelSizeGB) {
  if (modelSizeGB <= 0) {
    return {
      low:  { min: 0, max: 0, label: 'CPU-only', sub: 'N/A' },
      mid:  { min: 0, max: 0, label: 'Mid GPU', sub: 'N/A' },
      high: { min: 0, max: 0, label: 'High-end', sub: 'N/A' },
    };
  }

  // Reference hardware bandwidths
  const cpuBandwidth = 50;        // ~50 GB/s DDR5
  const midGPU_bandwidth = 936;   // RTX 3090 = 936 GB/s
  const highGPU_bandwidth = 2039; // A100 80G = 2039 GB/s (or 2x A100)

  const lowSpeed = estimateTokensPerSec(cpuBandwidth, modelSizeGB, true);
  const midSpeed = estimateTokensPerSec(midGPU_bandwidth, modelSizeGB, false);
  const highSpeed = estimateTokensPerSec(highGPU_bandwidth, modelSizeGB, false);

  return {
    low: {
      min: Math.max(1, Math.floor(lowSpeed * 0.7)),
      max: Math.ceil(lowSpeed * 1.1),
      label: 'Low',
      sub: 't/s · CPU-only',
    },
    mid: {
      min: Math.max(1, Math.floor(midSpeed * 0.7)),
      max: Math.ceil(midSpeed * 1.1),
      label: 'Mid',
      sub: 't/s · RTX 3090',
    },
    high: {
      min: Math.max(1, Math.floor(highSpeed * 0.7)),
      max: Math.ceil(highSpeed * 1.1),
      label: 'High',
      sub: 't/s · 2× A100',
    },
  };
}

/**
 * Estimate tokens/sec for a specific GPU
 * @param {number} gpuBandwidthGBs - GPU memory bandwidth in GB/s
 * @param {number} modelSizeGB - Model weight size in GB
 * @returns {{ speed: number, rangeMin: number, rangeMax: number }}
 */
export function estimateSpeedForGPU(gpuBandwidthGBs, modelSizeGB) {
  const speed = estimateTokensPerSec(gpuBandwidthGBs, modelSizeGB, false);
  return {
    speed: Math.round(speed),
    rangeMin: Math.max(1, Math.floor(speed * 0.7)),
    rangeMax: Math.ceil(speed * 1.1),
  };
}

/**
 * Format speed display string
 * @param {number} min - Min tokens/sec
 * @param {number} max - Max tokens/sec
 * @returns {string} Formatted speed range
 */
export function formatSpeedRange(min, max) {
  if (min === max) return `~${min}`;
  return `${min}–${max}`;
}

/**
 * Format a single speed estimate with ~ prefix
 * @param {number} speed - Tokens per second
 * @returns {string}
 */
export function formatSpeed(speed) {
  return `~${Math.round(speed)} t/s`;
}