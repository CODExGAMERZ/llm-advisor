# ⚡ LLM Hardware Advisor

An elegant, client-side single-page web application (SPA) managed by **Vite** that acts as a precision advisor for running Large Language Models (LLMs) on local consumer rigs, workstations, servers, and laptop hardware. 

Know exactly what quantization levels and context sizes your rig can handle before you download a model.

---

## 🚀 Key Features

*   **100% Client-Side:** No server requirements, no trackers, and zero data leaving your browser. Calculations run instantly in pure JavaScript.
*   **3D Glassmorphic "Space-Terminal" Design:** A high-fidelity, futuristic dark-themed environment featuring:
    *   **3D Horizon Grid Backdrop:** An animated perspective grid floor (`rotateX(65deg)`) with infinite loop scrolling.
    *   **Interactive 3D Mouse Tilt:** Hovering over panels and cards triggers realistic 3D rotation (`rotateX`/`rotateY` transforms) synced to cursor position.
    *   **Specular Glare Effects:** Dynamic radial light highlights sweep across glass panels as the mouse sweeps.
    *   **Performance Budgeted Loop:** A single centralized `requestAnimationFrame` loop in JS updates all tilt elements, using dynamic `will-change: transform` allocation on hover to save GPU memory.
    *   **Reduced Motion Support:** Seamlessly transitions grid animations, transitions, and 3D tilts to static layouts under system `@media (prefers-reduced-motion: reduce)`.
    *   **Mobile Touch Fallbacks:** Autodetects mobile pointer states (`hover: none`) to disable mouse listeners and render static frosted glass surfaces.
*   **Mode A: Model ➔ Specs:** Select from 64 models across 16 families. Configure context length, batch size, quantizations, and inference backend. Get a real-time, interactive **3D VRAM usage breakdown** (Weights vs. KV Cache vs. Backend Overhead) modeled as glowing neon plasma tubes, and **estimated tokens/sec**.
*   **Mode B: Specs ➔ Model:** Select your GPU, GPU Count (for multi-GPU setups), CPU class, RAM size, and targets. Instantly see a ranked list of models that fit your hardware.
*   **Mode C: Custom Calculator:** Estimate VRAM, bandwidth, disk, and GPU compatibility for any custom or newly released model by entering custom parameters, quantization types, and attention architectures (GQA, MQA, MHA).
*   **Apple Silicon Unified Memory Support:** If CPU class is set to Apple Silicon, the GPU search is automatically linked to system unified memory (75% allocation limit) with dedicated callouts explaining how to optimize it using macOS terminal commands.
*   **Multi-GPU Configuration:** Support dual, triple, or quad GPU arrangements, automatically pooling VRAM limits and accounting for PCIe interconnect scaling/overhead in estimated generation speeds.
*   **Laptop Hardware Support:** Expanded seed database of **95+ GPUs** including **25 laptop GPUs** sorted from lowest to highest.
*   **Compatibility Drawer:** Click on any model to slide in a complete quantization compatibility table comparing FP16 down to 2-bit quants for your specific hardware.
*   **Upgrade Callouts:** Intelligently calculates if minor upgrades (e.g. adding 8 GB VRAM) would unlock significant new models.

---

## 📁 Architecture & File Structure

The project is structured as a Vite-managed vanilla ES Modules (ESM) web app. Static data files reside in `public/` to ensure they bundle correctly for production fetch requests.

```text
specs/
├── index.html                 # Core HTML5 shell (views, header, drawer structure)
├── package.json               # Vite project scripts and dev dependencies
├── vite.config.js             # Vite dev server and port configurations
├── README.md                  # Project documentation
├── .gitignore                 # Excludes OS/IDE metadata, scratch work, dist, and node_modules
├── public/                    # Static assets copied verbatim to production dist/
│   └── src/
│       └── data/
│           ├── gpus.json      # Seed GPU database (NVIDIA, AMD, Intel, Laptop series, Apple Silicon)
│           └── models.json    # Seed LLM database (Llama 3.x, Qwen 2.5, DeepSeek, Gemma 2/3, Mistral, etc.)
└── src/
    ├── app.js                 # Central ESM controller (DOM/State/Events/3D Tilt Controller)
    ├── automation.js          # Automation scripts for testing/screenshots
    ├── css/
    │   └── style.css          # Redesigned 3D Space-Terminal stylesheet (layouts, visual tokens)
    └── engine/
        ├── fitScore.js        # Engine module for hardware compatibility score and best quant solver
        ├── memoryCalc.js      # Engine module for parameter weights VRAM, KV cache, and RAM overheads
        └── speedEstimate.js   # Engine module for bandwidth-to-token inference speeds
```

---

## 🛠️ Running Locally

The project utilizes **Vite** for local development and optimized production bundling.

### Prerequisites
Make sure you have [Node.js](https://nodejs.org/) installed.

### 1. Install Dependencies
Run this command in the project root:
```bash
npm install
```

### 2. Start Development Server
Launch the local hot-reloading dev server:
```bash
npm run dev
```
Navigate to the URL output in your terminal (usually `http://localhost:8080`).

### 3. Production Build & Preview
To compile the minified and optimized bundle:
```bash
npm run build
```
To test and preview the production build locally:
```bash
npm run preview
```
This serves the compiled files from the `dist/` directory at the preview port.

---

## 🛡️ License

This project is open-source. All code is running purely on the client side, maintaining ultimate privacy for your hardware specs and usage statistics.
