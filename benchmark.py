#!/usr/bin/env python3
"""
Benchmark script for Pocket-TTS-Spokenword-GPU
Tests single-worker and multi-worker GPU performance.

Usage: python benchmark.py  or  ./benchmark.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
torch.set_float32_matmul_precision("high")

# Linux requires 'spawn' for CUDA multiprocessing (default 'fork' crashes)
import multiprocessing as _mp
_mp.set_start_method('spawn', force=True)

BENCHMARK_TEXTS = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "In the distance, the mountains rose like ancient sentinels against the darkening sky.",
    "She opened the old leather journal and began to write about everything she had witnessed that day.",
    "The laboratory equipment hummed quietly as the scientist prepared the next phase of the experiment.",
    "Wind swept through the valley carrying the scent of pine and distant rain across the open meadow.",
    "Scientists discovered a new species of deep-sea fish living near hydrothermal vents at unprecedented depths.",
    "The old library smelled of aged paper and leather bindings, a scent that transported visitors to another era.",
    "Rain drummed steadily against the windowpane while she read aloud from the worn copy of her favorite novel.",
]


def _worker_fn(worker_id, num_chunks, result_dict):
    import torch
    torch.set_float32_matmul_precision("high")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pocket_tts.models.tts_model import TTSModel

    model = TTSModel.load_model(device="auto")
    voice = model.get_state_for_audio_prompt("alba", truncate=True)

    texts = [BENCHMARK_TEXTS[(worker_id * num_chunks + i) % len(BENCHMARK_TEXTS)] for i in range(num_chunks)]

    durs = []
    start = time.time()
    for t in texts:
        cs = time.time()
        audio = model.generate_audio(voice, t, frames_after_eos=2)
        ce = time.time()
        dur = audio.shape[-1] / model.sample_rate
        durs.append((ce - cs, dur))
    result_dict[worker_id] = durs


def benchmark_single_worker():
    """Benchmark single worker GPU throughput."""
    from pocket_tts.models.tts_model import TTSModel

    print("=" * 60)
    print("SINGLE WORKER BENCHMARK")
    print("=" * 60)
    print(f"torch {torch.__version__}, CUDA {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    model = TTSModel.load_model(device="auto")
    print(f"Model loaded on {model.device}")

    voice = model.get_state_for_audio_prompt("alba", truncate=True)
    print("Voice loaded (built-in alba)\n")

    # Warmup
    model.generate_audio(voice, "Warmup.", frames_after_eos=2)

    durs = []
    start = time.time()
    for i, t in enumerate(BENCHMARK_TEXTS):
        cs = time.time()
        audio = model.generate_audio(voice, t, frames_after_eos=2)
        ce = time.time()
        dur = audio.shape[-1] / model.sample_rate
        durs.append(dur)
        print(f"  Chunk {i+1}: gen={ce - cs:.2f}s, audio={dur:.1f}s")

    elapsed = time.time() - start
    total_audio = sum(durs)
    rtf = total_audio / elapsed
    print(f"\nResult: {elapsed:.2f}s for {total_audio:.1f}s audio = {rtf:.1f}x realtime")
    return rtf


def benchmark_multi_worker(num_workers=6):
    """Benchmark multi-worker GPU throughput."""
    from multiprocessing import Manager

    print("\n" + "=" * 60)
    print(f"MULTI-WORKER BENCHMARK ({num_workers} workers)")
    print("=" * 60)

    spawn_ctx = _mp.get_context('spawn')
    manager = Manager()
    result_dict = manager.dict()
    chunks_per_worker = 8

    start = time.time()
    procs = []
    for i in range(num_workers):
        p = spawn_ctx.Process(target=_worker_fn, args=(i, chunks_per_worker, result_dict))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    elapsed = time.time() - start

    total_audio = 0
    for wid in sorted(result_dict.keys()):
        for gen_time, audio_dur in result_dict[wid]:
            print(f"  Worker {wid}: gen={gen_time:.2f}s, audio={audio_dur:.1f}s")
            total_audio += audio_dur

    rtf = total_audio / elapsed
    print(f"\nResult: {elapsed:.2f}s for {total_audio:.1f}s audio = {rtf:.1f}x realtime")
    return rtf


def benchmark_worker_sweep(worker_counts=(1, 2, 3, 4, 6)):
    """Sweep worker counts to find the throughput plateau on this platform.

    Skips re-running 1 worker if the single-worker benchmark already ran.
    """
    print("\n" + "=" * 60)
    print("WORKER COUNT SWEEP")
    print("=" * 60)
    results = {}
    for n in worker_counts:
        results[n] = benchmark_multi_worker(num_workers=n)

    print("\n" + "=" * 60)
    print("SWEEP SUMMARY (Windows plateau test)")
    print("=" * 60)
    print(f"  {'Workers':<10}{'RTF':<12}{'Delta':<12}{'Efficiency':<12}")
    print(f"  {'-------':<10}{'---':<12}{'-----':<12}{'----------':<12}")
    prev = None
    for n in worker_counts:
        rtf = results[n]
        delta = f"+{(rtf - prev):.1f}x" if prev is not None else "-"
        eff = f"{(rtf / n):.2f}x/worker" if n > 0 else "-"
        print(f"  {n:<10}{rtf:<12.1f}x{delta:<12}{eff:<12}")
        prev = rtf
    best = max(results, key=results.get)
    print(f"\n  Best worker count: {best} ({results[best]:.1f}x realtime)")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pocket-TTS-GPU benchmark")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep worker counts (1,2,3,4,6) to find plateau")
    parser.add_argument("--workers", type=int, default=6,
                        help="Worker count for multi-worker benchmark (default: 6)")
    args = parser.parse_args()

    single = benchmark_single_worker()

    if args.sweep:
        multi = benchmark_worker_sweep()
    else:
        multi = {args.workers: benchmark_multi_worker(num_workers=args.workers)}
        best_rtf = multi[args.workers]
        best_workers = args.workers
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Single worker:  {single:.1f}x realtime")
        print(f"  Multi worker:   {best_rtf:.1f}x realtime")
        print(f"  Parallel speedup: {best_rtf / single:.1f}x")

    print()
    print("Environment info:")
    print(f"  OS: {sys.platform}")
    print(f"  Python: {sys.version}")
    print(f"  PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  CUDA: {torch.version.cuda}")
        print(f"  cuDNN: {torch.backends.cudnn.version()}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  cudnn.benchmark: {torch.backends.cudnn.benchmark}")
        print(f"  float32_matmul_precision: {torch.get_float32_matmul_precision()}")
