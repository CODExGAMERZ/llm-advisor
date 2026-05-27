# ⚡ LLM Hardware Advisor

An elegant, client-side single-page application (SPA) that acts as a precision advisor for running Large Language Models (LLMs) on local consumer rigs, workstations, servers, and laptop hardware. 

Know exactly what quantization levels and context sizes your rig can handle before you download a model.

---

## 🚀 Key Features

*   **100% Client-Side:** No server requirements, no trackers, and zero data leaving your browser. Calculations run instantly in pure JavaScript.
*   **"Terminal Precision" Design System:** A high-contrast, premium, dark-mode terminal aesthetic designed with custom CSS variables, custom typography (JetBrains Mono & DM Sans), micro-animations, and full mobile responsiveness.
*   **Mode A: Model ➔ Specs:** Select from 64 models across 16 families. Configure context length, batch size, quantizations, and inference backend. Get a real-time, interactive **VRAM usage breakdown** (Weights vs. KV Cache vs. Backend Overhead) and **estimated tokens/sec**. Updates automatically as parameters slide.
*   **Mode B: Specs ➔ Model:** Select your GPU, GPU Count (for multi-GPU setups), CPU class, RAM size, and targets. Instantly see a ranked list of models that fit your hardware.
*   **Mode C: Custom Calculator:** Estimate VRAM, bandwidth, disk, and GPU compatibility for any custom or newly released model by entering custom parameters, quantization types, and attention architectures (GQA, MQA, MHA).
*   **Apple Silicon Unified Memory Support:** If CPU class is set to Apple Silicon, the GPU search is automatically linked to system unified memory (75% allocation limit) with dedicated callouts explaining how to optimize it using macOS terminal commands.
*   **Multi-GPU Configuration:** Support dual, triple, or quad GPU arrangements, automatically pooling VRAM limits and accounting for PCIe interconnect scaling/overhead in estimated generation speeds.
*   **Laptop Hardware Support:** Expanded seed database of **95+ GPUs** including **25 laptop GPUs** sorted from lowest to highest, allowing users to accurately evaluate gaming laptops.
*   **Three Sorting Modes for Specs ➔ Model:**
    1.  **Best Model (Default):** Sorted by parameter size (B) descending, then quantization quality (bits) descending, listing the smartest/most capable models that fit at the top.
    2.  **High Speed:** Sorted by estimated inference speed (tokens/sec) descending.
    3.  **VRAM Fit Margin:** Sorted by VRAM utilization score (more comfortable margins first).
*   **Compatibility Drawer:** Click on any model to slide in a complete quantization compatibility table comparing FP16 down to 2-bit quants for your specific hardware.
*   **Upgrade Callouts:** Intelligently calculates if minor upgrades (e.g. adding 8 GB VRAM) would unlock significant new models.

---

## 📁 Architecture & File Structure

The project has been reorganized into clean, modular directories using browser-native **ES Modules (ESM)**. No bundlers, Node.js, Webpack, or compilers are required.

```text
specs/
├── index.html                 # Core HTML5 shell (views, header, drawer structure)
├── README.md                  # Project documentation
├── .gitignore                 # Excludes OS/IDE metadata and scratch work
└── src/
    ├── app.js                 # Central ESM frontend application controller (DOM/State/Events)
    ├── css/
    │   └── style.css          # Design system stylesheet (layout, transitions, typography, themes)
    ├── data/
    │   ├── gpus.json          # Seed GPU database (NVIDIA, AMD, Intel, Laptop series, Apple Silicon)
    │   └── models.json        # Seed LLM database (Llama 3.x, Qwen 2.5, DeepSeek, Gemma 2/3, Mistral, etc.)
    └── engine/
        ├── fitScore.js        # Engine module for hardware compatibility score and best quant solver
        ├── memoryCalc.js      # Engine module for parameter weights VRAM, KV cache, and RAM overheads
        └── speedEstimate.js   # Engine module for bandwidth-to-token inference speeds
```

---

## 🧮 Mathematical Formulations

The calculations under the hood utilize established LLM resource formulas:

### 1. Model Weights Memory
$$\text{Memory}_{\text{weights}} (\text{GB}) = \frac{\text{Params} \times 10^9 \times \text{Bits per weight}}{8 \times 10^9}$$

### 2. Key-Value (KV) Cache Memory
$$\text{Memory}_{\text{kv}} (\text{GB}) = \frac{2 \times \text{Layers} \times \text{KV Heads} \times \text{Head Dimension} \times \text{Context Length} \times \text{Batch} \times \text{Bytes/KV}}{10^9}$$
*Note: GGUF quants use 1 byte/element for KV, whereas full precision (FP16/BF16) uses 2 bytes/element.*

### 3. VRAM Fit Score
$$\text{Fit Score} = \frac{\text{GPU VRAM available (GB)}}{\text{Weights} + \text{KV-Cache} + \text{Inference Backend Overhead}}$$
*   **Best Fit:** $\text{Fit Score} \ge 1.30$ (VRAM headroom is comfortable)
*   **Comfortable:** $1.15 \le \text{Fit Score} < 1.30$
*   **Tight:** $1.00 \le \text{Fit Score} < 1.15$ (Fully on GPU, but tight)
*   **Stretch:** $0.85 \le \text{Fit Score} < 1.00$ (Partial CPU offloading required)
*   **Won't Fit:** $\text{Fit Score} < 0.85$ (Insufficient memory)

### 4. Speed Estimates (Tokens per Second)
$$\text{Tokens/Sec} \approx \frac{\text{Memory Bandwidth (GB/s)}}{\text{Model Weights Size (GB)} \times \text{Overhead Factor}}$$
*   $\text{Overhead Factor} = 1.2$ for GPU inference.
*   $\text{Overhead Factor} = 2.5$ for CPU/Offloaded inference.

---

## 🛠️ Running Locally

Since the application uses standard ES Modules, it must be run from a local web server (to avoid CORS blockages on local JSON files). 

### Option A: Python (Recommended)
If you have Python installed, run this command in the project root:
```bash
python -m http.server 8080
```
Then navigate to `http://localhost:8080` in your web browser.

### Option B: Node.js (npx)
If you have Node.js installed, run:
```bash
npx serve .
```
And open the link outputted in your console (usually `http://localhost:3000`).

---

## 🛡️ License

This project is open-source. All code is running purely on the client side, maintaining ultimate privacy for your hardware specs and usage statistics.
