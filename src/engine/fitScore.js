/**
 * LLM Hardware Advisor — Fit Score Engine
 * Evaluates how well a model fits on a given GPU.
 */
import { calcTotalVRAM } from './memoryCalc.js';

/**
 * Fit score categories
 */
export const FIT_LABELS = {
  BEST_FIT:    { key: 'BEST_FIT',    label: 'BEST FIT',    cssClass: 'fit-best',     color: '#00D4AA' },
  COMFORTABLE: { key: 'COMFORTABLE', label: 'COMFORTABLE', cssClass: 'fit-ok',       color: '#00D4AA' },
  TIGHT:       { key: 'TIGHT',       label: 'TIGHT',       cssClass: 'fit-tight',    color: '#F59E0B' },
  FEASIBLE:    { key: 'FEASIBLE',    label: 'FEASIBLE',    cssClass: 'fit-feasible', color: '#F59E0B' },
  STRETCH:     { key: 'STRETCH',     label: 'STRETCH',     cssClass: 'fit-stretch',  color: '#F59E0B' },
  WONT_FIT:    { key: 'WONT_FIT',    label: "WON'T FIT",   cssClass: 'fit-fail',     color: '#EF4444' },
};

/**
 * Calculate the fit score for a model on a GPU
 * @param {number} gpuVRAM_GB - GPU VRAM in GB
 * @param {number} totalVRAM_GB - Total VRAM required by the model
 * @returns {number} Fit score (> 1 means it fits)
 */
export function calcFitScore(gpuVRAM_GB, totalVRAM_GB) {
  if (totalVRAM_GB <= 0) return Infinity;
  return gpuVRAM_GB / totalVRAM_GB;
}

/**
 * Get the fit label for a given score
 * @param {number} score - Fit score
 * @returns {object} Fit label object with key, label, cssClass, color
 */
export function getFitLabel(score) {
  if (score >= 1.30) return FIT_LABELS.BEST_FIT;
  if (score >= 1.15) return FIT_LABELS.COMFORTABLE;
  if (score >= 1.00) return FIT_LABELS.TIGHT;
  if (score >= 0.85) return FIT_LABELS.FEASIBLE;
  return FIT_LABELS.WONT_FIT;
}

/**
 * Get fit label for Mode B results (uses STRETCH instead of FEASIBLE for consistency)
 * @param {number} score - Fit score
 * @returns {object} Fit label object
 */
export function getFitLabelModeB(score) {
  if (score >= 1.30) return FIT_LABELS.BEST_FIT;
  if (score >= 1.15) return FIT_LABELS.COMFORTABLE;
  if (score >= 1.00) return FIT_LABELS.TIGHT;
  if (score >= 0.85) return FIT_LABELS.STRETCH;
  return FIT_LABELS.WONT_FIT;
}

/**
 * Get the fill bar percentage for visual display
 * @param {number} totalVRAM_GB - Total VRAM required
 * @param {number} gpuVRAM_GB - GPU VRAM available
 * @returns {number} Percentage (0-100) of how much VRAM is used
 */
export function getUsagePercent(totalVRAM_GB, gpuVRAM_GB) {
  if (gpuVRAM_GB <= 0) return 100;
  return Math.min(100, Math.round((totalVRAM_GB / gpuVRAM_GB) * 100));
}

/**
 * Get the bar color based on usage percentage
 * @param {number} usagePercent - Usage percentage
 * @returns {string} CSS color string
 */
export function getUsageColor(usagePercent) {
  if (usagePercent <= 75) return '#00D4AA';
  if (usagePercent <= 90) return '#F59E0B';
  return '#EF4444';
}

/**
 * Determine the best quantization for a model to fit a GPU
 * @param {number} paramsB - Model params in billions
 * @param {object} model - Model record
 * @param {number} gpuVRAM_GB - GPU VRAM in GB
 * @param {string[]} supportedQuants - Available quantization formats
 * @param {number} contextLen - Context length in tokens
 * @param {string} backend - Inference backend
 * @returns {{ quant: string, score: number, label: object } | null}
 */
export function findBestQuant(paramsB, model, gpuVRAM_GB, supportedQuants, contextLen, backend, gpuCount = 1) {

  // Order quants from highest quality to lowest
  const qualityOrder = ['FP16', 'BF16', 'Q8_0', 'Q6_K', 'Q5_K_M', 'Q5_K_S', 'Q4_K_M', 'Q4_K_S', 'Q4_0', 'Q3_K_M', 'Q3_K_S', 'Q2_K', 'IQ2_XXS'];
  
  let bestResult = null;
  
  for (const quant of qualityOrder) {
    if (!supportedQuants.includes(quant)) continue;
    
    const vram = calcTotalVRAM(paramsB, model, quant, contextLen, backend, 1, gpuCount);
    const score = calcFitScore(gpuVRAM_GB, vram.totalGB);
    
    if (score >= 0.85) {
      const label = getFitLabelModeB(score);
      if (!bestResult || score > bestResult.score) {
        bestResult = { quant, score, label, vram };
      }
      // We want the highest quality quant that fits, so take the first one that fits comfortably
      if (score >= 1.15) {
        return { quant, score, label, vram };
      }
    }
  }
  
  return bestResult;
}