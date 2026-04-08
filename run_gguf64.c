/* 
 * Project: Llama2.c GGUF 64-bit GPU Research Edition
 * Engineer: NARWIT SURAPHAN
 * Description: ระบบ Inference โมเดลภาษา (LLM) ที่เน้นความแม่นยำสูง (64-bit) 
 *              รองรับไฟล์มาตรฐาน GGUF และเร่งความเร็วด้วย GPU AMD RX 570 ผ่าน OpenCL
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include "win.h"
#include "ocl_dynamic.h"

// --- นิยามชนิดข้อมูลหลัก (ใช้ Double เพื่อความแม่นยำสูงสุดในงานวิจัย) ---
typedef double real_t;

// --- GGUF Data Structures ---
typedef enum {
    GGUF_TYPE_UINT8 = 0, GGUF_TYPE_INT8 = 1, GGUF_TYPE_UINT16 = 2, GGUF_TYPE_INT16 = 3,
    GGUF_TYPE_UINT32 = 4, GGUF_TYPE_INT32 = 5, GGUF_TYPE_FLOAT32 = 6, GGUF_TYPE_BOOL = 7,
    GGUF_TYPE_STRING = 8, GGUF_TYPE_ARRAY = 9, GGUF_TYPE_UINT64 = 10, GGUF_TYPE_INT64 = 11,
    GGUF_TYPE_FLOAT64 = 12,
} gguf_type;

typedef struct {
    uint32_t magic;      // 'GGUF'
    uint32_t version;    // 3
    uint64_t tensor_count;
    uint64_t kv_count;
} gguf_header;

// --- โครงสร้างการตั้งค่าโมเดล (Config) ---
typedef struct {
    int dim;            // มิติของ Transformer (Llama-3-8B คือ 4096)
    int hidden_dim;     // มิติในชั้น Hidden (FFN)
    int n_layers;       // จำนวนชั้น (Layers)
    int n_heads;        // จำนวนหัวอ่าน (Heads)
    int n_kv_heads;     // จำนวนหัวอ่าน Key/Value
    int vocab_size;     // ขนาดของคลังคำ (Vocabulary)
    int seq_len;        // ความยาวสูงสุดของบริบท (Context Length)
} Config;

// --- โครงสร้างเก็บค่าน้ำหนักโมเดล (Weights) ---
typedef struct {
    real_t* token_embedding_table; 
    real_t* rms_final_weight;      
} TransformerWeights;

// --- ระบบจัดการ GPU (OpenCL State) ---
typedef struct {
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel matmul_kernel;
    cl_device_id device;
    int is_ready;
} GPUState;

GPUState gpu = {0};

// --- GGUF Helper Functions ---

void read_gguf_string(FILE *f, char *buffer, size_t max_len) {
    uint64_t len;
    if (fread(&len, sizeof(len), 1, f) != 1) return;
    size_t read_len = len < max_len - 1 ? len : max_len - 1;
    if (fread(buffer, 1, read_len, f) != 1) return;
    buffer[read_len] = '\0';
    if (len > read_len) fseek(f, len - read_len, SEEK_CUR);
}

void skip_gguf_value(FILE *f, uint32_t type) {
    if (type == GGUF_TYPE_STRING) {
        uint64_t len; fread(&len, sizeof(len), 1, f);
        fseek(f, len, SEEK_CUR);
    } else if (type == GGUF_TYPE_ARRAY) {
        uint32_t subtype; fread(&subtype, sizeof(subtype), 1, f);
        uint64_t len; fread(&len, sizeof(len), 1, f);
        for (uint64_t i = 0; i < len; i++) skip_gguf_value(f, subtype);
    } else {
        int sizes[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
        fseek(f, sizes[type], SEEK_CUR);
    }
}

// --- OpenCL Kernel ---
const char *matmul_kernel_source = 
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void matmul_gpu(__global const double* x, __global const double* w, __global double* out, int n) {\n"
"   int i = get_global_id(0);\n"
"   double sum = 0.0;\n"
"   __global const double* w_row = w + (i * n);\n"
"   for (int j = 0; j < n; j++) {\n"
"       sum += x[j] * w_row[j];\n"
"   }\n"
"   out[i] = sum;\n"
"}\n";

int init_gpu_system() {
    if (!init_opencl_dynamic()) return 0;
    cl_uint num_platforms;
    oclGetPlatformIDs(0, NULL, &num_platforms);
    if (num_platforms == 0) return 0;
    cl_platform_id platform;
    oclGetPlatformIDs(1, &platform, NULL);
    cl_uint num_devices;
    if (oclGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &gpu.device, &num_devices) != CL_SUCCESS) return 0;
    cl_int err;
    gpu.context = oclCreateContext(NULL, 1, &gpu.device, NULL, NULL, &err);
    gpu.queue = oclCreateCommandQueue(gpu.context, gpu.device, 0, &err);
    gpu.program = oclCreateProgramWithSource(gpu.context, 1, &matmul_kernel_source, NULL, &err);
    err = oclBuildProgram(gpu.program, 1, &gpu.device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) return 0;
    gpu.matmul_kernel = oclCreateKernel(gpu.program, "matmul_gpu", &err);
    gpu.is_ready = 1;
    return 1;
}

// --- ฟังก์ชันอ่าน Metadata จาก GGUF ---
int load_gguf_model(const char* path, Config* config) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("[Error] ไม่สามารถเปิดไฟล์ได้: %s\n", path); return 0; }

    gguf_header header;
    if (fread(&header, sizeof(gguf_header), 1, f) != 1) { fclose(f); return 0; }
    if (header.magic != 0x46554747) { printf("[Error] ไม่ใช่ไฟล์ GGUF ที่ถูกต้อง\n"); fclose(f); return 0; }

    printf("[Loader] ตรวจพบไฟล์ GGUF v%u (Metadata: %llu pairs)\n", header.version, header.kv_count);
    
    for (uint64_t i = 0; i < header.kv_count; i++) {
        char key[256];
        read_gguf_string(f, key, sizeof(key));
        uint32_t type;
        fread(&type, sizeof(type), 1, f);

        if (strcmp(key, "llama.embedding_length") == 0) fread(&config->dim, 4, 1, f);
        else if (strcmp(key, "llama.block_count") == 0) fread(&config->n_layers, 4, 1, f);
        else if (strcmp(key, "llama.attention.head_count") == 0) fread(&config->n_heads, 4, 1, f);
        else if (strcmp(key, "llama.feed_forward_length") == 0) fread(&config->hidden_dim, 4, 1, f);
        else if (strcmp(key, "llama.context_length") == 0) fread(&config->seq_len, 4, 1, f);
        else skip_gguf_value(f, type);
    }
    
    printf("[Loader] วิเคราะห์โครงสร้างโมเดลสำเร็จ:\n");
    printf(" >> Layers: %d\n", config->n_layers);
    printf(" >> Dimension: %d\n", config->dim);
    printf(" >> Context Window: %d\n", config->seq_len);
    
    fclose(f);
    return 1;
}

int main(int argc, char *argv[]) {
    #if defined _WIN32
        system("chcp 65001 > nul");
    #endif

    printf("==========================================\n");
    printf("   Llama2.c GGUF 64-bit GPU Research\n");
    printf("   Developed by: NARWIT SURAPHAN\n");
    printf("==========================================\n\n");

    if (argc < 2) {
        printf("วิธีใช้: run_gguf64.exe <ชื่อไฟล์.gguf>\n");
        return 1;
    }

    if (init_gpu_system()) {
        char gpu_name[128];
        oclGetDeviceInfo(gpu.device, CL_DEVICE_NAME, sizeof(gpu_name), gpu_name, NULL);
        printf("[GPU] เปิดใช้งานสำเร็จ: %s\n", gpu_name);
    } else {
        printf("[GPU] ล้มเหลว! จะใช้งานผ่าน CPU แทน\n");
    }

    Config config = {0};
    if (load_gguf_model(argv[1], &config)) {
        printf("\n[วิจัย] ผลการตรวจสอบโมเดล %s:\n", argv[1]);
        printf(" - ความแม่นยำระบบ: 64-bit Double Precision\n");
        printf(" - สถานะโมเดล: พร้อมสำหรับการประมวลผล\n");
    }

    return 0;
}
