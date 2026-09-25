import threading
import time

from core import Kernel
from core.module import Module


class BlockingShutdownModule(Module):
    def __init__(self, kernel):
        super().__init__(
            name="BlockingShutdown",
            event_bus=kernel.event_bus,
            kernel=kernel,
        )
        self.entered = threading.Event()
        self.release = threading.Event()

    def shutdown(self):
        self.entered.set()
        self.release.wait(timeout=5)


def test_kernel_run_waits_for_shutdown_started_by_worker_thread():
    kernel = Kernel()
    module = BlockingShutdownModule(kernel)
    kernel.register_module(module)
    kernel.start()

    run_thread = threading.Thread(target=kernel.run, name="KernelRunTest")
    run_thread.start()

    time.sleep(0.05)

    shutdown_thread = threading.Thread(
        target=kernel.shutdown,
        name="KernelShutdownTest",
    )
    shutdown_thread.start()

    assert module.entered.wait(timeout=1.0)

    time.sleep(0.05)
    assert run_thread.is_alive()

    module.release.set()

    shutdown_thread.join(timeout=2.0)
    run_thread.join(timeout=2.0)

    assert not shutdown_thread.is_alive()
    assert not run_thread.is_alive()
