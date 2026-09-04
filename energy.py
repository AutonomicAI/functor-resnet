"""
RAPL-based energy measurement. Mirrors measure_rapl.sh's logic (same package
domains, same wraparound handling) but as an in-process Python context manager,
so it can wrap individual commits/corrections without spawning a subprocess
per measurement.

Requires /sys/class/powercap/intel-rapl/*/energy_uj to be world-readable
(see the udev rule at /etc/udev/rules.d/60-rapl-permissions.rules).
"""
import time
from contextlib import contextmanager
from dataclasses import dataclass

RAPL_BASE = "/sys/class/powercap/intel-rapl"
PACKAGES = ["intel-rapl:0", "intel-rapl:1"]

# Idle baseline captured separately (measure_rapl.sh, box otherwise quiet):
# three runs -> 135.21W, 135.23W, 135.09W. Mean used for marginal-energy subtraction.
IDLE_BASELINE_WATTS = (135.21 + 135.23 + 135.09) / 3


def _read_uj(pkg: str) -> int:
    with open(f"{RAPL_BASE}/{pkg}/energy_uj") as f:
        return int(f.read().strip())


def _read_max_range_uj(pkg: str) -> int:
    with open(f"{RAPL_BASE}/{pkg}/max_energy_range_uj") as f:
        return int(f.read().strip())


def _delta(end: int, start: int, max_range: int) -> int:
    if end >= start:
        return end - start
    return (max_range - start) + end  # counter wrapped around


@dataclass
class EnergyReading:
    seconds: float
    joules_by_package: dict
    total_joules: float
    watts: float
    idle_joules_estimate: float
    marginal_joules: float  # total - idle estimate for this window


@contextmanager
def measure_energy():
    """
    Usage:
        with measure_energy() as reading:
            do_work()
        print(reading.result.total_joules)
    `reading` is a mutable holder; `.result` is populated after the block exits.
    """
    max_ranges = {pkg: _read_max_range_uj(pkg) for pkg in PACKAGES}
    starts = {pkg: _read_uj(pkg) for pkg in PACKAGES}
    t0 = time.time()

    holder = _ResultHolder()
    try:
        yield holder
    finally:
        t1 = time.time()
        ends = {pkg: _read_uj(pkg) for pkg in PACKAGES}

        seconds = t1 - t0
        joules_by_package = {
            pkg: _delta(ends[pkg], starts[pkg], max_ranges[pkg]) / 1e6 for pkg in PACKAGES
        }
        total_joules = sum(joules_by_package.values())
        watts = total_joules / seconds if seconds > 0 else 0.0
        idle_joules_estimate = IDLE_BASELINE_WATTS * seconds
        marginal_joules = total_joules - idle_joules_estimate

        holder.result = EnergyReading(
            seconds=seconds,
            joules_by_package=joules_by_package,
            total_joules=total_joules,
            watts=watts,
            idle_joules_estimate=idle_joules_estimate,
            marginal_joules=marginal_joules,
        )


class _ResultHolder:
    result: EnergyReading = None


if __name__ == "__main__":
    with measure_energy() as r:
        time.sleep(3)
    print(r.result)
