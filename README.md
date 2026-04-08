# Llama2.c GGUF 64-bit GPU Research Edition 🚀

Developed by: **NARWIT SURAPHAN** (2026)

This project is a research-focused implementation of an LLM (Large Language Model) inference engine. It evolves the original **llama2.c** by Andrej Karpathy into a high-precision system supporting the **GGUF (v3)** standard with **64-bit Double Precision** (FP64) and **AMD GPU Acceleration**.

---

## 🛠 Technical Implementation

### 1. Full 64-bit Precision Architecture
The entire computation pipeline has been upgraded from 32-bit Float to **64-bit Double** to ensure maximum numerical stability and minimize rounding errors during deep layer transitions.
* **Flexible Typedef:** Utilizing `typedef double real_t;` for system-wide precision control.
* **Math Function Migration:** Replaced all single-precision functions with their double-precision counterparts:
    * `sqrtf()` $\rightarrow$ `sqrt()`
    * `expf()` $\rightarrow$ `exp()`
    * `fabsf()` $\rightarrow$ `fabs()`
* **64-byte Alignment:** Memory allocation is strictly aligned to 64 bytes to optimize data throughput for modern CPU/GPU architectures.

### 2. Custom GGUF (v3) Parser
Built from scratch to support the modern GGUF format, allowing the engine to handle industry-standard model files.
* **Metadata KV Extraction:** Automatically parses Key-Value pairs to extract model hyperparameters like `llama.embedding_length` and `llama.block_count`.
* **64-bit Tensor Mapping:** Implements precise offset calculation to map weights directly into memory with 64-bit accuracy.

### 3. Optimized Dequantization (Q4_0 to FP64)
Developed a specialized algorithm to bridge the gap between compressed 4-bit weights and high-precision inference.
* **Nibble Processing:** Extracts 4-bit values from GGUF super-blocks and scales them directly into **64-bit Double** in real-time.

### 4. AMD GPU Acceleration (OpenCL)
Leverages the power of the **AMD RX 570 (Polaris)** via a custom OpenCL backend.
* **Dynamic Loading:** Uses a runtime OpenCL loader, removing the need for users to manually install the full OpenCL SDK.
* **Advanced Kernels:** Written in OpenCL C with **Loop Unrolling** and **Vectorized Loading** for peak Matrix Multiplication (MatMul) performance.
* **FP64 Extension:** Activates `cl_khr_fp64` to maintain computational parity between the CPU and GPU.

---

## ✨ Key Benefits

* **Higher Precision:** Reduces cumulative error across layers, helping the AI maintain context and logic more effectively than standard 32-bit systems.
* **Modern Compatibility:** Runs standard GGUF models available in the AI community.
* **Hardware Empowerment:** Optimizes performance for accessible hardware like the AMD RX 570.

---

## 🚀 Installation & Usage

### 1. Compilation
Compile using GCC with OpenMP and optimization flags:

```bash
gcc -Ofast -fopenmp -D_WIN32 -o run_gguf64.exe -I. run_gguf64.c win.c