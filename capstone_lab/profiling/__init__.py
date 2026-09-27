"""S6 resource profiling adapters; never formal training or evaluation."""

from .contracts import (
    S6Campaign,
    launch_guard,
    load_s6_campaign,
    reservation_cap_mib,
    select_common_batch,
)

__all__ = [
    "S6Campaign",
    "launch_guard",
    "load_s6_campaign",
    "reservation_cap_mib",
    "select_common_batch",
]
