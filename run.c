/* Inference for Llama-2 Transformer model in pure C - OPTIMIZED VERSION */
/* เพิ่มประสิทธิภาพ: SIMD, Loop unrolling, Memory alignment, Precomputed RoPE */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>

#if defined _WIN32
    #include "win.h"
    #include <malloc.h>  // สำหรับ _aligned_malloc
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads
    int vocab_size; // vocabulary size
    int seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim)
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls
    float* wq; // (layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_kv_heads * head_size)
    float* wv; // (layer, dim, n_kv_heads * head_size)
    float* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // classifier weights
    float* wcls;
    // Precomputed RoPE caches (เพิ่มมาใหม่)
    float* cos_cache;
    float* sin_cache;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x;      // (dim,)
    float *xb;     // (dim,)
    float *xb2;    // (dim,)
    float *hb;     // (hidden_dim,)
    float *hb2;    // (hidden_dim,)
    float *q;      // (dim,)
    float *att;    // (n_heads, seq_len)
    float *logits; // (vocab_size,)
    // kv cache
    float* key_cache;   // (layer, seq_len, kv_dim)
    float* value_cache; // (layer, seq_len, kv_dim)
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
    int fd;
    float* data;
    ssize_t file_size;
} Transformer;

// ----------------------------------------------------------------------------
// Memory pool for better allocation (เพิ่มมาใหม่)
typedef struct {
    float* pool;
    size_t size;
    size_t used;
} MemoryPool;

void memory_pool_init(MemoryPool* pool, size_t size) {
    pool->size = size;
    pool->used = 0;
    #if defined _WIN32
        pool->pool = (float*)_aligned_malloc(size, 64);
    #else
        posix_memalign((void**)&pool->pool, 64, size);
    #endif
}

void* memory_pool_alloc(MemoryPool* pool, size_t bytes) {
    size_t aligned = (bytes + 63) & ~63;
    if (pool->used + aligned > pool->size) return NULL;
    void* ptr = (char*)pool->pool + pool->used;
    pool->used += aligned;
    return ptr;
}

void memory_pool_free(MemoryPool* pool) {
    #if defined _WIN32
        _aligned_free(pool->pool);
    #else
        free(pool->pool);
    #endif
    pool->pool = NULL;
}

// ----------------------------------------------------------------------------
// Optimized neural net blocks

// RMSNorm with SIMD optimization
void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    #pragma omp simd reduction(+:ss)
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss = 1.0f / sqrtf(ss / size + 1e-5f);
    
    #pragma omp simd
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

// Optimized Softmax with stability check
void softmax(float* x, int size) {
    float max_val = x[0];
    #pragma omp simd reduction(max:max_val)
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    float sum = 0.0f;
    #pragma omp simd reduction(+:sum)
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    
    if (sum < 1e-7f) sum = 1e-7f;
    float inv_sum = 1.0f / sum;
    #pragma omp simd
    for (int i = 0; i < size; i++) {
        x[i] *= inv_sum;
    }
}

// Optimized Matrix Multiplication with loop unrolling
void matmul(float* xout, float* x, float* w, int n, int d) {
    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        int j = 0;
        // Loop unrolling: process 4 elements at a time
        for (; j <= n - 4; j += 4) {
            val += w[i * n + j]     * x[j];
            val += w[i * n + j + 1] * x[j + 1];
            val += w[i * n + j + 2] * x[j + 2];
            val += w[i * n + j + 3] * x[j + 3];
        }
        // Remaining elements
        for (; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

// ----------------------------------------------------------------------------
// Memory management with alignment

void malloc_run_state(RunState* s, Config* p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    
    #if defined _WIN32
        s->x = (float*)_aligned_malloc(p->dim * sizeof(float), 32);
        s->xb = (float*)_aligned_malloc(p->dim * sizeof(float), 32);
        s->xb2 = (float*)_aligned_malloc(p->dim * sizeof(float), 32);
        s->hb = (float*)_aligned_malloc(p->hidden_dim * sizeof(float), 32);
        s->hb2 = (float*)_aligned_malloc(p->hidden_dim * sizeof(float), 32);
        s->q = (float*)_aligned_malloc(p->dim * sizeof(float), 32);
        s->att = (float*)_aligned_malloc(p->n_heads * p->seq_len * sizeof(float), 32);
        s->logits = (float*)_aligned_malloc(p->vocab_size * sizeof(float), 32);
        s->key_cache = (float*)_aligned_malloc(p->n_layers * p->seq_len * kv_dim * sizeof(float), 32);
        s->value_cache = (float*)_aligned_malloc(p->n_layers * p->seq_len * kv_dim * sizeof(float), 32);
    #else
        posix_memalign((void**)&s->x, 32, p->dim * sizeof(float));
        posix_memalign((void**)&s->xb, 32, p->dim * sizeof(float));
        posix_memalign((void**)&s->xb2, 32, p->dim * sizeof(float));
        posix_memalign((void**)&s->hb, 32, p->hidden_dim * sizeof(float));
        posix_memalign((void**)&s->hb2, 32, p->hidden_dim * sizeof(float));
        posix_memalign((void**)&s->q, 32, p->dim * sizeof(float));
        posix_memalign((void**)&s->att, 32, p->n_heads * p->seq_len * sizeof(float));
        posix_memalign((void**)&s->logits, 32, p->vocab_size * sizeof(float));
        posix_memalign((void**)&s->key_cache, 32, p->n_layers * p->seq_len * kv_dim * sizeof(float));
        posix_memalign((void**)&s->value_cache, 32, p->n_layers * p->seq_len * kv_dim * sizeof(float));
    #endif
    
    // Zero out all allocations
    memset(s->x, 0, p->dim * sizeof(float));
    memset(s->xb, 0, p->dim * sizeof(float));
    memset(s->xb2, 0, p->dim * sizeof(float));
    memset(s->hb, 0, p->hidden_dim * sizeof(float));
    memset(s->hb2, 0, p->hidden_dim * sizeof(float));
    memset(s->q, 0, p->dim * sizeof(float));
    memset(s->att, 0, p->n_heads * p->seq_len * sizeof(float));
    memset(s->logits, 0, p->vocab_size * sizeof(float));
    memset(s->key_cache, 0, p->n_layers * p->seq_len * kv_dim * sizeof(float));
    memset(s->value_cache, 0, p->n_layers * p->seq_len * kv_dim * sizeof(float));
    
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    #if defined _WIN32
        _aligned_free(s->x);
        _aligned_free(s->xb);
        _aligned_free(s->xb2);
        _aligned_free(s->hb);
        _aligned_free(s->hb2);
        _aligned_free(s->q);
        _aligned_free(s->att);
        _aligned_free(s->logits);
        _aligned_free(s->key_cache);
        _aligned_free(s->value_cache);
    #else
        free(s->x);
        free(s->xb);
        free(s->xb2);
        free(s->hb);
        free(s->hb2);
        free(s->q);
        free(s->att);
        free(s->logits);
        free(s->key_cache);
        free(s->value_cache);
    #endif
}

// Precompute RoPE caches (เพิ่มมาใหม่)
void precompute_rope_caches(Transformer* t) {
    Config* p = &t->config;
    int head_size = p->dim / p->n_heads;
    size_t cache_size = p->seq_len * head_size * sizeof(float);
    
    #if defined _WIN32
        t->weights.cos_cache = (float*)_aligned_malloc(cache_size, 32);
        t->weights.sin_cache = (float*)_aligned_malloc(cache_size, 32);
    #else
        posix_memalign((void**)&t->weights.cos_cache, 32, cache_size);
        posix_memalign((void**)&t->weights.sin_cache, 32, cache_size);
    #endif
    
    for (int pos = 0; pos < p->seq_len; pos++) {
        for (int i = 0; i < head_size; i++) {
            float freq = 1.0f / powf(10000.0f, i / (float)head_size);
            float val = pos * freq;
            t->weights.cos_cache[pos * head_size + i] = cosf(val);
            t->weights.sin_cache[pos * head_size + i] = sinf(val);
        }
    }
}

void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    unsigned long long n_layers = p->n_layers;
    
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    // Skip RoPE caches in checkpoint file (เราใช้ precomputed แทน)
    ptr += p->seq_len * head_size;
    ptr += p->seq_len * head_size;
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    
    fseek(file, 0, SEEK_END);
    *file_size = ftell(file);
    fclose(file);
    
    *fd = open(checkpoint, O_RDONLY);
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    
    float* weights_ptr = *data + sizeof(Config)/sizeof(float);
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void build_transformer(Transformer *t, char* checkpoint_path) {
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    malloc_run_state(&t->state, &t->config);
    precompute_rope_caches(t);  // Precompute RoPE
}

void free_transformer(Transformer* t) {
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    free_run_state(&t->state);
    #if defined _WIN32
        _aligned_free(t->weights.cos_cache);
        _aligned_free(t->weights.sin_cache);
    #else
        free(t->weights.cos_cache);
        free(t->weights.sin_cache);
    #endif
}

// ----------------------------------------------------------------------------
// Optimized Forward Pass

float* forward(Transformer* transformer, int token, int pos) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    
    // Copy token embedding
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim * sizeof(*x));
    
    for (unsigned long long l = 0; l < p->n_layers; l++) {
        // Attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);
        
        // KV cache pointers
        int loff = l * p->seq_len * kv_dim;
        float* k = s->key_cache + loff + pos * kv_dim;
        float* v = s->value_cache + loff + pos * kv_dim;
        
        // QKV matmuls
        matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
        matmul(k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
        matmul(v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);
        
        // RoPE using precomputed caches (optimized!)
        float* cos_ptr = w->cos_cache + pos * head_size;
        float* sin_ptr = w->sin_cache + pos * head_size;
        
        for (int i = 0; i < dim; i += 2) {
            int head_dim = (i / 2) % head_size;
            float fcr = cos_ptr[head_dim];
            float fci = sin_ptr[head_dim];
            
            // Rotate query
            float q0 = s->q[i];
            float q1 = s->q[i + 1];
            s->q[i] = q0 * fcr - q1 * fci;
            s->q[i + 1] = q0 * fci + q1 * fcr;
            
            // Rotate key if within kv_dim
            if (i < kv_dim) {
                float k0 = k[i];
                float k1 = k[i + 1];
                k[i] = k0 * fcr - k1 * fci;
                k[i + 1] = k0 * fci + k1 * fcr;
            }
        }
        
        // Multihead attention
        #pragma omp parallel for
        for (int h = 0; h < p->n_heads; h++) {
            float* q_head = s->q + h * head_size;
            float* att = s->att + h * p->seq_len;
            
            for (int t = 0; t <= pos; t++) {
                float* k_head = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q_head[i] * k_head[i];
                }
                att[t] = score / sqrtf(head_size);
            }
            
            softmax(att, pos + 1);
            
            float* xb_head = s->xb + h * head_size;
            memset(xb_head, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v_head = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) {
                    xb_head[i] += a * v_head[i];
                }
            }
        }
        
        // Output projection
        matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);
        
        // Residual
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }
        
        // FFN rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);
        
        // FFN matmuls
        matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);
        
        // SwiGLU
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            s->hb[i] = val * s->hb2[i];
        }
        
        // FFN output
        matmul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);
        
        // Residual
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }
    
    // Final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);
    
    // Classifier
    matmul(s->logits, x, w->wcls, dim, p->vocab_size);
    return s->logits;
}

// ----------------------------------------------------------------------------
// Tokenizer, Sampler, and generation functions (unchanged from original)
// ... (ส่วนที่เหลือเหมือนเดิมเพื่อความกระชับ แต่ในไฟล์จริงต้องมีทั้งหมด)

// ----------------------------------------------------------------------------
// CLI

#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature, default 1.0\n");
    fprintf(stderr, "  -p <float>  top-p, default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed\n");
    fprintf(stderr, "  -n <int>    number of steps, default 256\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> tokenizer path\n");
    fprintf(stderr, "  -m <string> mode: generate|chat\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    char *checkpoint_path = NULL;
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;
    float topp = 0.9f;
    int steps = 256;
    char *prompt = NULL;
    unsigned long long rng_seed = 0;
    char *mode = "generate";
    char *system_prompt = NULL;
    
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc) error_usage();
        if (argv[i][0] != '-') error_usage();
        if (strlen(argv[i]) != 2) error_usage();
        
        if (argv[i][1] == 't') temperature = atof(argv[i + 1]);
        else if (argv[i][1] == 'p') topp = atof(argv[i + 1]);
        else if (argv[i][1] == 's') rng_seed = atoi(argv[i + 1]);
        else if (argv[i][1] == 'n') steps = atoi(argv[i + 1]);
        else if (argv[i][1] == 'i') prompt = argv[i + 1];
        else if (argv[i][1] == 'z') tokenizer_path = argv[i + 1];
        else if (argv[i][1] == 'm') mode = argv[i + 1];
        else if (argv[i][1] == 'y') system_prompt = argv[i + 1];
        else error_usage();
    }
    
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;
    
    printf("=== Optimized llama2.c ===\n");
    printf("Building transformer from: %s\n", checkpoint_path);
    
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len;
    
    printf("Model loaded. Vocab size: %d, Dim: %d, Layers: %d\n",
           transformer.config.vocab_size, transformer.config.dim, transformer.config.n_layers);
    
    // Note: Tokenizer and Sampler functions would be here
    // (omitted for brevity - include from original)
    
    printf("Ready for inference!\n");
    
    free_transformer(&transformer);
    return 0;
}
#endif
