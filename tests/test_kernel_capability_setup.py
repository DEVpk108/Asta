from core import Kernel
from core.autonomy import CapabilitySetupManager


def test_kernel_wires_capability_setup_manager():
    kernel = Kernel()

    assert isinstance(
        kernel.capability_setup_manager,
        CapabilitySetupManager,
    )
    assert kernel.capability_setup_manager.media_manager is kernel.media_manager
    assert kernel.capability_setup_manager.event_bus is kernel.event_bus
