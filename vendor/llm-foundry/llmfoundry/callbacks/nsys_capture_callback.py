import torch
import torch.distributed as dist
from composer.core import Callback, State
from composer.loggers import Logger
import torch.cuda.nvtx as nvtx


class NsysCaptureCallback(Callback):
    def __init__(self, start_batch: int, stop_batch: int):
        if start_batch >= stop_batch:
            raise ValueError("start_batch must be smaller than stop_batch")
        self.start_batch = start_batch
        self.stop_batch = stop_batch
        self.started = False
        self.stopped = False

    @staticmethod
    def _barrier() -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def batch_start(self, state: State, logger: Logger) -> None:
        # The batch we are about to start is (completed batches + 1)
        batch = int(state.timestamp.batch) + 1
        if batch == self.start_batch and not self.started:
            print(f"[nsys] starting capture at batch {batch}", flush=True)
            self._barrier()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.synchronize()
            self.started = True

    def batch_end(self, state: State, logger: Logger) -> None:
        batch = int(state.timestamp.batch)
        if batch == self.stop_batch and self.started and not self.stopped:
            print(f"[nsys] stopping capture at batch {batch}", flush=True)
            self._barrier()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            torch.cuda.synchronize()
            self._barrier()
            self.stopped = True
