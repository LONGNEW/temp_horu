from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import threading
import time
from typing import Any

try:
    import pynvml
except ImportError:  # pragma: no cover - optional dependency
    pynvml = None


@dataclass
class EnergyMeasurement:
    measurement_requested: bool
    elapsed_seconds: float = 0.0
    gpu_supported: bool = False
    gpu_device_index: int | None = None
    gpu_name: str | None = None
    gpu_energy_j: float | None = None
    gpu_avg_power_w: float | None = None
    gpu_mean_sampled_power_w: float | None = None
    gpu_peak_power_w: float | None = None
    gpu_sample_count: int = 0
    gpu_measurement_source: str | None = None
    gpu_error: str | None = None
    cpu_supported: bool = False
    cpu_package_count: int = 0
    cpu_energy_j: float | None = None
    cpu_avg_power_w: float | None = None
    cpu_measurement_source: str | None = None
    cpu_error: str | None = None
    total_measured_energy_j: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        known_total = 0.0
        has_known = False
        if self.gpu_energy_j is not None:
            known_total += float(self.gpu_energy_j)
            has_known = True
        if self.cpu_energy_j is not None:
            known_total += float(self.cpu_energy_j)
            has_known = True
        payload["total_measured_energy_j"] = known_total if has_known else None
        return payload


@dataclass(frozen=True)
class _RAPLDomain:
    name: str
    energy_path: Path
    max_energy_range_uj: int | None


def _safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _discover_rapl_domains() -> list[_RAPLDomain]:
    base = Path("/sys/class/powercap")
    if not base.exists():
        return []
    domains: list[_RAPLDomain] = []
    for path in sorted(base.glob("intel-rapl:*")):
        if path.name.count(":") != 1:
            continue
        name_path = path / "name"
        energy_path = path / "energy_uj"
        if not name_path.exists() or not energy_path.exists():
            continue
        try:
            name = _safe_read_text(name_path)
        except OSError:
            continue
        if not name.startswith("package"):
            continue
        max_range_path = path / "max_energy_range_uj"
        max_range = None
        if max_range_path.exists():
            try:
                max_range = int(_safe_read_text(max_range_path))
            except (OSError, ValueError):
                max_range = None
        domains.append(
            _RAPLDomain(
                name=name,
                energy_path=energy_path,
                max_energy_range_uj=max_range,
            )
        )
    return domains


def _read_rapl_uj(domains: list[_RAPLDomain]) -> tuple[list[int], str | None]:
    values: list[int] = []
    for domain in domains:
        try:
            values.append(int(_safe_read_text(domain.energy_path)))
        except PermissionError as exc:
            return [], f"{type(exc).__name__}: {exc}"
        except (OSError, ValueError) as exc:
            return [], f"{type(exc).__name__}: {exc}"
    return values, None


def _delta_with_wrap(start_value: int, end_value: int, max_range: int | None) -> int | None:
    if end_value >= start_value:
        return end_value - start_value
    if max_range is None or max_range <= 0:
        return None
    return (max_range - start_value) + end_value


class RunEnergyMonitor:
    def __init__(
        self,
        *,
        device: Any,
        enabled: bool,
        sample_interval_sec: float = 0.2,
    ) -> None:
        self._device = device
        self._enabled = bool(enabled)
        self._sample_interval_sec = max(float(sample_interval_sec), 0.05)
        self._start_perf = 0.0
        self._stopped = False
        self._measurement = EnergyMeasurement(measurement_requested=bool(enabled))

        self._gpu_handle = None
        self._gpu_start_energy_mj: int | None = None
        self._gpu_samples: list[tuple[float, float]] = []
        self._gpu_stop_event = threading.Event()
        self._gpu_thread: threading.Thread | None = None
        self._gpu_nvml_initialized = False

        self._cpu_domains: list[_RAPLDomain] = []
        self._cpu_start_values_uj: list[int] = []

    def __enter__(self) -> RunEnergyMonitor:
        self._start_perf = time.perf_counter()
        if not self._enabled:
            return self
        self._start_gpu_monitoring()
        self._start_cpu_monitoring()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        self.stop()

    def stop(self, *, elapsed_seconds: float | None = None) -> dict[str, Any]:
        if self._stopped:
            if elapsed_seconds is not None:
                self._measurement.elapsed_seconds = float(max(elapsed_seconds, 0.0))
                elapsed = max(float(self._measurement.elapsed_seconds), 1e-12)
                if self._measurement.gpu_energy_j is not None:
                    self._measurement.gpu_avg_power_w = self._measurement.gpu_energy_j / elapsed
                if self._measurement.cpu_energy_j is not None:
                    self._measurement.cpu_avg_power_w = self._measurement.cpu_energy_j / elapsed
            return self._measurement.to_dict()
        if elapsed_seconds is None:
            elapsed_seconds = time.perf_counter() - self._start_perf
        self._measurement.elapsed_seconds = float(max(elapsed_seconds, 0.0))
        self._stopped = True
        if not self._enabled:
            return self._measurement.to_dict()
        self._stop_gpu_monitoring()
        self._stop_cpu_monitoring()
        return self._measurement.to_dict()

    def _resolved_device_type(self) -> str:
        device_type = getattr(self._device, "type", None)
        if device_type is not None:
            return str(device_type)
        return str(self._device).split(":", 1)[0]

    def _resolved_gpu_index(self) -> int:
        device_index = getattr(self._device, "index", None)
        if device_index is not None:
            return int(device_index)
        device_str = str(self._device)
        if ":" in device_str:
            _, index_text = device_str.split(":", 1)
            if index_text.isdigit():
                return int(index_text)
        return 0

    def _record_gpu_sample(self) -> None:
        if self._gpu_handle is None or pynvml is None:
            return
        try:
            power_w = float(pynvml.nvmlDeviceGetPowerUsage(self._gpu_handle)) / 1000.0
        except Exception as exc:  # pragma: no cover - NVML errors are environment-specific
            if self._measurement.gpu_error is None:
                self._measurement.gpu_error = f"{type(exc).__name__}: {exc}"
            return
        self._gpu_samples.append((time.perf_counter(), power_w))

    def _gpu_sampler_loop(self) -> None:
        while not self._gpu_stop_event.wait(self._sample_interval_sec):
            self._record_gpu_sample()

    def _start_gpu_monitoring(self) -> None:
        if self._resolved_device_type() != "cuda":
            return
        if pynvml is None:
            self._measurement.gpu_error = "pynvml not installed"
            return
        try:
            pynvml.nvmlInit()
            self._gpu_nvml_initialized = True
            gpu_index = self._resolved_gpu_index()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode("utf-8")
            self._gpu_handle = handle
            self._measurement.gpu_supported = True
            self._measurement.gpu_device_index = gpu_index
            self._measurement.gpu_name = str(gpu_name)
            self._measurement.gpu_measurement_source = "nvml_total_energy"
            if hasattr(pynvml, "nvmlDeviceGetTotalEnergyConsumption"):
                self._gpu_start_energy_mj = int(pynvml.nvmlDeviceGetTotalEnergyConsumption(handle))
            self._record_gpu_sample()
            self._gpu_thread = threading.Thread(target=self._gpu_sampler_loop, daemon=True)
            self._gpu_thread.start()
        except Exception as exc:  # pragma: no cover - NVML errors are environment-specific
            self._measurement.gpu_error = f"{type(exc).__name__}: {exc}"
            self._measurement.gpu_supported = False
            self._gpu_handle = None
            if self._gpu_nvml_initialized:
                pynvml.nvmlShutdown()
                self._gpu_nvml_initialized = False

    def _stop_gpu_monitoring(self) -> None:
        if self._gpu_thread is not None:
            self._gpu_stop_event.set()
            self._gpu_thread.join(timeout=max(self._sample_interval_sec * 4.0, 1.0))
            self._gpu_thread = None
        if self._gpu_handle is None or pynvml is None:
            return
        self._record_gpu_sample()
        elapsed = max(float(self._measurement.elapsed_seconds), 1e-12)
        if self._gpu_start_energy_mj is not None and hasattr(pynvml, "nvmlDeviceGetTotalEnergyConsumption"):
            try:
                gpu_end_energy_mj = int(pynvml.nvmlDeviceGetTotalEnergyConsumption(self._gpu_handle))
                delta_mj = max(gpu_end_energy_mj - self._gpu_start_energy_mj, 0)
                self._measurement.gpu_energy_j = float(delta_mj) / 1000.0
                self._measurement.gpu_avg_power_w = self._measurement.gpu_energy_j / elapsed
            except Exception as exc:  # pragma: no cover - NVML errors are environment-specific
                if self._measurement.gpu_error is None:
                    self._measurement.gpu_error = f"{type(exc).__name__}: {exc}"
        if self._gpu_samples:
            powers = [sample[1] for sample in self._gpu_samples]
            self._measurement.gpu_sample_count = len(powers)
            self._measurement.gpu_mean_sampled_power_w = float(sum(powers) / len(powers))
            self._measurement.gpu_peak_power_w = float(max(powers))
        if self._gpu_nvml_initialized:
            pynvml.nvmlShutdown()
            self._gpu_nvml_initialized = False
        self._gpu_handle = None

    def _start_cpu_monitoring(self) -> None:
        domains = _discover_rapl_domains()
        if not domains:
            self._measurement.cpu_error = "intel_rapl_unavailable"
            return
        values, error = _read_rapl_uj(domains)
        if error is not None:
            self._measurement.cpu_error = error
            return
        self._cpu_domains = domains
        self._cpu_start_values_uj = values
        self._measurement.cpu_supported = True
        self._measurement.cpu_package_count = len(domains)
        self._measurement.cpu_measurement_source = "intel_rapl_energy_uj"

    def _stop_cpu_monitoring(self) -> None:
        if not self._cpu_domains or not self._cpu_start_values_uj:
            return
        end_values, error = _read_rapl_uj(self._cpu_domains)
        if error is not None:
            self._measurement.cpu_error = error
            return
        total_delta_uj = 0
        for domain, start_value, end_value in zip(self._cpu_domains, self._cpu_start_values_uj, end_values):
            delta_uj = _delta_with_wrap(start_value, end_value, domain.max_energy_range_uj)
            if delta_uj is None:
                self._measurement.cpu_error = "rapl_counter_wrap_without_range"
                return
            total_delta_uj += int(delta_uj)
        elapsed = max(float(self._measurement.elapsed_seconds), 1e-12)
        self._measurement.cpu_energy_j = float(total_delta_uj) / 1_000_000.0
        self._measurement.cpu_avg_power_w = self._measurement.cpu_energy_j / elapsed
