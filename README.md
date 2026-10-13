# Bitcoin Mining Simulation: CUDA Implementation

A simplified blockchain **Proof-of-Work** miner. It builds a Merkle root and searches for a nonce, ported from serial C to **CUDA** and benchmarked on an **NVIDIA A100**.

For each block, the miner:
1. Builds the **Merkle root** of the block's transactions (SHA-256, pairwise hashing level by level).
2. Searches for the smallest **nonce** for which `SHA256(prev_hash + merkle_root + nonce)` is below the difficulty target.
3. Uses the resulting block hash as `prev_hash` for the next block.

> Educational model of PoW. It does not reproduce the real Bitcoin consensus algorithm.

## Results (Tesla A100)

| Test | Blocks | Tx / block | CPU serial (s) | GPU Merkle (s) | GPU nonce (s) | Speedup |
|------|-------:|-----------:|---------------:|---------------:|--------------:|--------:|
| test1 | 4  | 10     | 8.417 | 0.00159 | 0.01187 | ~625x |
| test2 | 6  | 10     | 8.516 | 0.00232 | 0.01198 | ~595x |
| test3 | 11 | 1,000  | —     | 0.00912 | 7.00538 | —     |
| test4 | 10 | 10,000 | 1.976 | 0.01462 | 0.00358 | ~108x |

The CPU times come from the serial reference in `cpu_miner/`. The test3 CPU run did not finish: it took over 560 s for only 3 of 11 blocks.

## GPU implementation (`gpu_miner/src/utils.cu`)

### Merkle root
- **`kernel_hash_transactions`** uses one thread per transaction. All leaf hashes are computed in parallel.
- **`kernel_merkle_level`** is launched once per tree level. Each thread hashes one pair (`left = 2i`, `right = 2i+1`, or `left` again when the count is odd, matching the CPU semantics).
- **Ping-pong buffers:** two device buffers alternate as source and destination between levels, so no level overwrites data it still needs.
- **Pinned host memory + `cudaMemcpyAsync`** on a dedicated stream gives DMA transfers that don't block the CPU.

### Nonce search
- Each thread tests `nonce = base + blockIdx.x * blockDim.x + threadIdx.x`. It copies the block prefix through the read-only cache (`__ldg`), appends the nonce, hashes it and compares the result with the difficulty.
- **Hierarchical min-reduction** to find the smallest valid nonce:
  - inside a warp: `__shfl_down_sync`, registers only;
  - inside a block: per-warp minimums in shared memory, then a second shuffle reduction;
  - globally: a single `atomicMin` per block, which means 512x fewer global atomics than one per thread.
- **Batched search with early exit:** the nonce space is scanned in batches of `threads x SMs x 8`, where the SM count is read at runtime with `cudaGetDeviceProperties`. The search stops as soon as a batch finds a valid nonce.
- **Sentinel `UINT32_MAX + 1`** (64-bit) for "not found". It can never collide with a real `uint32_t` nonce, whereas `max_nonce + 1` could.
- The difficulty is uploaded only when it changes; the block content is re-uploaded for every block.

### Allocation off the hot path
- `warm_up_gpu` sets up the CUDA context, the stream and the pinned buffers.
- It preallocates 16 MB for transactions and 2 MB for hashes. Buffers grow only if a test needs more, so the timed functions never call `cudaMalloc`.

## Build & run

The project targets a Slurm cluster. The `make` rules submit jobs with `sbatch`, by default to the `ucsx` partition and with CUDA 13.1 for the GPU build.

```bash
cd gpu_miner          # or cpu_miner for the serial reference
make                  # compile (submitted as a Slurm job)
make run TEST=test4   # results in output/test4.out
make clean
```

Each output line has the format `block,block_hash,merkle_root,nonce,merkle_time,nonce_time,total_time`.

To check correctness, compare the GPU output against the CPU reference:

```bash
diff <(cut -d',' -f1-4 gpu_miner/output/test1.out) <(cut -d',' -f1-4 cpu_miner/output/test1.out)
```

To compile locally without Slurm, run `make compile` in `gpu_miner/` (requires `nvcc`). On WSL you may need to add `-arch=sm_XX` to `CFLAGS` to match your GPU.

## Repository layout

```
common/miner.cpp      driver: reads tests, runs blocks, writes output
include/              shared headers (SHA-256, utils)
cpu_miner/src/        serial reference implementation
gpu_miner/src/        CUDA implementation (utils.cu) + device SHA-256
tests/                input tests (test1 to test4)
```

## Credits

Developed as an assignment for the *Computer Systems Architecture* course at the Faculty of Automatic Control and Computers, Politehnica University of Bucharest (2026). The course team provided the driver (`common/`), the headers, the SHA-256 code, the serial CPU miner, the Makefiles and the tests. The CUDA implementation in `gpu_miner/src/utils.cu` is my own work.
