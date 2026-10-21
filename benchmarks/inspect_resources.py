"""Offline resource inspection for the six fixed Avey tile configurations.

Run with: python3 -m benchmarks.inspect_resources
No GPU driver or profiling-counter access is needed. Counts describe code
compiled by the installed Triton/ptxas; they do not measure runtime traffic.
"""
import triton
from triton import knobs
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource
from avey_kernels._triton import _mix_kernel
from benchmarks.compare_ordering import TILE_CONFIGS


def main():
    knobs.nvidia.dump_ptxas_log = True
    knobs.compilation.always_compile = True
    print(f'Triton={triton.__version__}; target=sm86; shape=(32,256,256); dtype=bf16', flush=True)
    for bm, bd in TILE_CONFIGS:
        print(f'\nBM={bm} BD={bd}', flush=True)
        signature = {name: '*bf16' for name in ('Z_TR', 'V', 'Z_TL', 'Bias', 'Out')}
        signature.update(Z_TR_NORM='*fp32', Valid='*i1')
        constants = dict(C=256, D_PRIME=256, EPS=1e-6, HAS_V=True,
                         HAS_Z_TL=True, HAS_BIAS=True, HAS_VALID=False,
                         CAUSAL=True, NORMALIZE=False, ACT=0,
                         BM=bm, BN=64, BK=32, BD=bd, BATCH=32,
                         DEVICE=0, GROUP_M=1, SKIP_CAUSAL=True)
        kernel = triton.compile(ASTSource(_mix_kernel, signature, constants),
                                target=GPUTarget('cuda', 86, 32),
                                options={'num_warps': 4, 'num_stages': 3})
        print(f'shared_bytes={kernel.metadata.shared}; programs={32*(256//bm)*(256//bd)}; score_repeats={256//bd}', flush=True)


if __name__ == "__main__":
    main()
