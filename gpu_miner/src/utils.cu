#include <stdio.h>
#include <stdint.h>
#include "utils.h"
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA sprintf alternative for nonce finding. Converts integer to its string representation. Returns string's length.
__device__ int intToString(uint64_t num, char* out) {
    if (num == 0) {
        out[0] = '0';
        out[1] = '\0';
        return 1;
    }

    int i = 0;
    while (num != 0) {
        int digit = num % 10;
        num /= 10;
        out[i++] = '0' + digit;
    }

    // Reverse the string
    for (int j = 0; j < i / 2; j++) {
        char temp = out[j];
        out[j] = out[i - j - 1];
        out[i - j - 1] = temp;
    }
    out[i] = '\0';
    return i;
}

// CUDA strlen implementation.
__host__ __device__ size_t d_strlen(const char *str) {
    size_t len = 0;
    while (str[len] != '\0') {
        len++;
    }
    return len;
}

// CUDA strcpy implementation.
__device__ void d_strcpy(char *dest, const char *src){
    int i = 0;
    while ((dest[i] = src[i]) != '\0') {
        i++;
    }
}

// CUDA strcat implementation.
__device__ void d_strcat(char *dest, const char *src){
    while (*dest != '\0') {
        dest++;
    }
    while (*src != '\0') {
        *dest = *src;
        dest++;
        src++;
    }
    *dest = '\0';
}

// Compute SHA256 and convert to hex
__host__ __device__ void apply_sha256(const BYTE *input, BYTE *output) {
    size_t input_length = d_strlen((const char *)input);
    SHA256_CTX ctx;
    BYTE buf[SHA256_BLOCK_SIZE];
    const char hex_chars[] = "0123456789abcdef";

    sha256_init(&ctx);
    sha256_update(&ctx, input, input_length);
    sha256_final(&ctx, buf);

    for (size_t i = 0; i < SHA256_BLOCK_SIZE; i++) {
        output[i * 2]     = hex_chars[(buf[i] >> 4) & 0x0F];  // High nibble
        output[i * 2 + 1] = hex_chars[buf[i] & 0x0F];         // Low nibble
    }
    output[SHA256_BLOCK_SIZE * 2] = '\0'; // Null-terminate
}

// Compare two hashes
__host__ __device__ int compare_hashes(BYTE* hash1, BYTE* hash2) {
    for (int i = 0; i < SHA256_HASH_SIZE - 1; i++) {
        if (hash1[i] < hash2[i]) {
            return -1; // hash1 is lower
        } else if (hash1[i] > hash2[i]) {
            return 1; // hash2 is lower
        }
    }
    return 0; // hashes are equal
}

// Buffere persistente pe GPU - alocate lazy, reutilizate la fiecare apel
static BYTE *g_d_transactions = NULL;
static BYTE *g_d_hashes_a = NULL;
static BYTE *g_d_hashes_b = NULL;
static size_t g_d_tx_capacity = 0;
static size_t g_d_hash_capacity = 0;

static BYTE *g_d_difficulty = NULL;
static BYTE *g_d_block_content = NULL;
static unsigned long long *g_d_result = NULL;

// Buffere pinned pe host pentru transferuri H2D/D2H rapide
static BYTE *g_h_transactions = NULL;
static size_t g_h_tx_capacity = 0;
static unsigned long long *g_h_result = NULL;
static BYTE *g_h_merkle_root = NULL;

// Cache difficulty - se reincarca pe GPU doar daca se schimba
static BYTE g_cached_difficulty[SHA256_HASH_SIZE] = {0};
static int g_difficulty_valid = 0;

static cudaStream_t g_stream = 0;
static int g_num_sms = 1;

static void ensure_merkle_buffers(size_t tx_bytes, size_t hash_bytes) {
    if (tx_bytes > g_d_tx_capacity) {
        if (g_d_transactions) {
            cudaFree(g_d_transactions);
        }
        cudaMalloc(&g_d_transactions, tx_bytes);
        g_d_tx_capacity = tx_bytes;
    }
    if (hash_bytes > g_d_hash_capacity) {
        if (g_d_hashes_a) {
            cudaFree(g_d_hashes_a);
        }
        if (g_d_hashes_b) {
            cudaFree(g_d_hashes_b);
        }
        cudaMalloc(&g_d_hashes_a, hash_bytes);
        cudaMalloc(&g_d_hashes_b, hash_bytes);
        g_d_hash_capacity = hash_bytes;
    }
    if (tx_bytes > g_h_tx_capacity) {
        if (g_h_transactions) {
            cudaFreeHost(g_h_transactions);
        }
        cudaMallocHost(&g_h_transactions, tx_bytes);
        g_h_tx_capacity = tx_bytes;
    }
}

static void ensure_nonce_buffers() {
    if (!g_d_difficulty) {
        cudaMalloc(&g_d_difficulty, SHA256_HASH_SIZE);
        cudaMalloc(&g_d_block_content, BLOCK_SIZE);
        cudaMalloc(&g_d_result, sizeof(unsigned long long));
    }
}

// Kernel: calculeaza SHA256 pentru fiecare tranzactie si stocheaza in hashes[i]
__global__ void kernel_hash_transactions(int transaction_size, BYTE *transactions, BYTE *hashes, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    apply_sha256(transactions + i * transaction_size, hashes + i * SHA256_HASH_SIZE);
}

// Kernel: un nivel de reducere al arborelui Merkle
__global__ void kernel_merkle_level(BYTE *src, BYTE *dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int pairs = (n + 1) / 2;
    if (i >= pairs) {
        return;
    }

    int left = i * 2;
    int right = (left + 1 < n) ? left + 1 : left;

    BYTE combined[SHA256_HASH_SIZE * 2];
    d_strcpy((char *)combined, (const char *)(src + left * SHA256_HASH_SIZE));
    d_strcat((char *)combined, (const char *)(src + right * SHA256_HASH_SIZE));
    apply_sha256(combined, dst + i * SHA256_HASH_SIZE);
}

// Kernel: fiecare thread testeaza un nonce din intervalul [base, base+gridDim.x*blockDim.x).
// Reducere la nivel de warp si bloc pentru a minimiza operatiile atomice globale.
__global__ void kernel_find_nonce(BYTE *difficulty, uint32_t max_nonce, BYTE *block_content,
                        size_t current_length, unsigned long long *result, uint32_t base) {
    // Incarca cel mai bun rezultat global o singura data per bloc in shared memory
    __shared__ unsigned long long s_min;
    if (threadIdx.x == 0) {
        s_min = *result;
    }
    __syncthreads();

    uint32_t nonce = base + blockIdx.x * blockDim.x + threadIdx.x;
    // NOT_FOUND: valoare mai mare decat orice nonce valid (UINT32_MAX + 1 incape in unsigned long long)
    const unsigned long long NOT_FOUND = (unsigned long long)UINT32_MAX + 1ULL;
    unsigned long long local_min = NOT_FOUND;

    if (nonce < max_nonce && (unsigned long long)nonce < s_min) {
        BYTE local_content[BLOCK_SIZE];
        for (int i = 0; i < (int)current_length; i++) {
            local_content[i] = __ldg(&block_content[i]);
        }

        char nonce_str[NONCE_SIZE];
        int len = intToString(nonce, nonce_str);
        for (int i = 0; i <= len; i++) {
            local_content[current_length + i] = nonce_str[i];
        }

        BYTE hash[SHA256_HASH_SIZE];
        apply_sha256(local_content, hash);

        if (compare_hashes(hash, difficulty) <= 0) {
            local_min = (unsigned long long)nonce;
        }
    }

    // Reducere minim la nivel de warp
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_min = min(local_min, __shfl_down_sync(0xffffffff, local_min, offset));
    }

    // Reducere la nivel de bloc prin shared memory - o intrare per warp
    __shared__ unsigned long long s_warp_mins[16]; // up to 512/32 = 16 warps
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    if (lane == 0) {
        s_warp_mins[warp_id] = local_min;
    }
    __syncthreads();

    // Primul warp reduce peste toate warp-urile, apoi un singur atomic global
    if (warp_id == 0) {
        int num_warps = blockDim.x >> 5;
        local_min = (lane < num_warps) ? s_warp_mins[lane] : NOT_FOUND;
        for (int offset = 8; offset > 0; offset >>= 1) {
            local_min = min(local_min, __shfl_down_sync(0xffffffff, local_min, offset));
        }
        if (lane == 0 && local_min < NOT_FOUND) {
            atomicMin(result, local_min);
        }
    }
}

// TODO 1: Implement this function in CUDA
void construct_merkle_root(int transaction_size, BYTE *transactions, int max_transactions_in_a_block, int n, BYTE merkle_root[SHA256_HASH_SIZE]) {
    size_t tx_bytes = (size_t)n * transaction_size;
    size_t hash_bytes = (size_t)max_transactions_in_a_block * SHA256_HASH_SIZE;
    ensure_merkle_buffers(tx_bytes, hash_bytes);

    // Pinned memory + transfer async pentru a suprapune H2D cu alte operatii GPU
    memcpy(g_h_transactions, transactions, tx_bytes);
    cudaMemcpyAsync(g_d_transactions, g_h_transactions, tx_bytes, cudaMemcpyHostToDevice, g_stream);

    int threads = 512;
    int blocks = (n + threads - 1) / threads;
    kernel_hash_transactions<<<blocks, threads, 0, g_stream>>>(transaction_size, g_d_transactions, g_d_hashes_a, n);

    BYTE *src = g_d_hashes_a;
    BYTE *dst = g_d_hashes_b;
    int cur_n = n;

    while (cur_n > 1) {
        int pairs = (cur_n + 1) / 2;
        blocks = (pairs + threads - 1) / threads;
        kernel_merkle_level<<<blocks, threads, 0, g_stream>>>(src, dst, cur_n);
        BYTE *tmp = src; src = dst; dst = tmp;
        cur_n = pairs;
    }

    // Transfer async D2H in buffer pinned, apoi sincronizare
    cudaMemcpyAsync(g_h_merkle_root, src, SHA256_HASH_SIZE, cudaMemcpyDeviceToHost, g_stream);
    cudaStreamSynchronize(g_stream);
    memcpy(merkle_root, g_h_merkle_root, SHA256_HASH_SIZE);
}

// TODO 2: Implement this function in CUDA
int find_nonce(BYTE *difficulty, uint32_t max_nonce, BYTE *block_content, size_t current_length, BYTE *block_hash, uint32_t *valid_nonce) {
    ensure_nonce_buffers();

    // Reincarca difficulty pe GPU doar daca s-a schimbat (constanta pentru toate blocurile din test)
    if (!g_difficulty_valid || memcmp(g_cached_difficulty, difficulty, SHA256_HASH_SIZE) != 0) {
        cudaMemcpyAsync(g_d_difficulty, difficulty, SHA256_HASH_SIZE, cudaMemcpyHostToDevice, g_stream);
        memcpy(g_cached_difficulty, difficulty, SHA256_HASH_SIZE);
        g_difficulty_valid = 1;
    }

    cudaMemcpyAsync(g_d_block_content, block_content, current_length + 1, cudaMemcpyHostToDevice, g_stream);

    // Lansare in batch-uri pentru a putea opri imediat ce s-a gasit un nonce valid.
    // Dimensiunea batch-ului = un val complet de thread-uri pe GPU (num_sms × 8 blocuri × 512 threads).
    const int threads = 512;
    const uint32_t batch = (uint32_t)threads * g_num_sms * 8; // one wave per SM × 8 blocks

    // Santinela NOT_FOUND: mai mare decat orice nonce valid (nonces sunt uint32, aceasta valoare nu)
    const unsigned long long NOT_FOUND = (unsigned long long)UINT32_MAX + 1ULL;
    *g_h_result = NOT_FOUND;
    cudaMemcpyAsync(g_d_result, g_h_result, sizeof(unsigned long long), cudaMemcpyHostToDevice, g_stream);

    unsigned long long result = NOT_FOUND;

    for (uint32_t base = 0; base < max_nonce; base += batch) {
        uint32_t end = (base + batch < max_nonce) ? base + batch : max_nonce;
        uint32_t blk = (end - base + threads - 1) / threads;

        kernel_find_nonce<<<blk, threads, 0, g_stream>>>(g_d_difficulty, end, g_d_block_content, current_length, g_d_result, base);

        cudaMemcpyAsync(g_h_result, g_d_result, sizeof(unsigned long long), cudaMemcpyDeviceToHost, g_stream);
        cudaStreamSynchronize(g_stream);

        result = *g_h_result;
        if (result < NOT_FOUND) {
            break; // s-a gasit un nonce valid - oprire anticipata
        }
    }

    if (result >= NOT_FOUND) {
        return 1; // nu s-a gasit niciun nonce valid
    }

    *valid_nonce = (uint32_t)result;

    // Recalculeaza block_hash pentru nonce-ul valid pe CPU
    char nonce_string[NONCE_SIZE];
    sprintf(nonce_string, "%u", *valid_nonce);
    strcpy((char *)block_content + current_length, nonce_string);
    apply_sha256(block_content, block_hash);

    return 0;
}

__global__ void dummy_kernel() {}

// Warm-up function
void warm_up_gpu() {
    BYTE *dummy_data;
    cudaMalloc((void **)&dummy_data, 256);
    dummy_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    cudaFree(dummy_data);

    // Creeaza stream persistent si buffere pinned folosite de ambele functii
    cudaStreamCreate(&g_stream);
    cudaMallocHost(&g_h_result, sizeof(unsigned long long));
    cudaMallocHost(&g_h_merkle_root, SHA256_HASH_SIZE);

    // Citeste proprietatile GPU-ului la runtime, fara valori hardcodate
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    g_num_sms = prop.multiProcessorCount;

    // Prealoca buffere cu dimensiuni initiale rezonabile pentru a evita latenta de alocare la primul apel.
    // Cresc automat daca vine un input mai mare (ensure_merkle_buffers se ocupa de asta).
    size_t init_tx_bytes = (size_t)16 * 1024 * 1024; // 16 MB
    size_t init_hash_bytes = (size_t)2  * 1024 * 1024; // 2 MB
    cudaMalloc(&g_d_transactions, init_tx_bytes);
    cudaMalloc(&g_d_hashes_a, init_hash_bytes);
    cudaMalloc(&g_d_hashes_b, init_hash_bytes);
    cudaMallocHost(&g_h_transactions, init_tx_bytes);
    g_d_tx_capacity = init_tx_bytes;
    g_d_hash_capacity = init_hash_bytes;
    g_h_tx_capacity = init_tx_bytes;

    // Prealoca bufferele pentru cautarea nonce-ului
    cudaMalloc(&g_d_difficulty, SHA256_HASH_SIZE);
    cudaMalloc(&g_d_block_content, BLOCK_SIZE);
    cudaMalloc(&g_d_result, sizeof(unsigned long long));
}
