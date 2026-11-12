"""Configurable packet-delivery primitives; no paper scenario constants."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
EVENT_ON_TIME = np.uint8(0)
EVENT_LATE = np.uint8(1)
EVENT_DROPPED = np.uint8(2)
DROP_LATENCY = np.int16(-1)

@dataclass(frozen=True)
class Scenario:
    """Controlled virtual transport condition.

    ``late_probability`` and ``drop_probability`` apply independently to each
    scalar telemetry packet while the link is good.  A burst scenario changes
    these probabilities using either independent device-level Markov states or
    one shared Markov congestion state.  This keeps all output masks at scalar
    resolution while making the source of temporal correlation explicit.
    """
    name: str
    late_probability: float
    drop_probability: float
    minimum_late_steps: int
    maximum_late_steps: int
    device_extra_delay_steps: int
    device_extra_delay_jitter: int
    burst_good_to_bad_probability: float
    burst_bad_to_good_probability: float
    burst_late_probability: float
    burst_drop_probability: float
    shared_burst_state: bool

def _validate_scenario(scenario: Scenario) -> None:
    probabilities = (scenario.late_probability, scenario.drop_probability, scenario.burst_good_to_bad_probability, scenario.burst_bad_to_good_probability, scenario.burst_late_probability, scenario.burst_drop_probability)
    if any((value < 0.0 or value > 1.0 for value in probabilities)):
        raise ValueError(f'Scenario {scenario.name} has a probability outside [0, 1].')
    if scenario.late_probability + scenario.drop_probability > 1.0:
        raise ValueError(f'Scenario {scenario.name} has incompatible packet probabilities.')
    if scenario.burst_late_probability + scenario.burst_drop_probability > 1.0:
        raise ValueError(f'Scenario {scenario.name} has incompatible burst probabilities.')
    if scenario.minimum_late_steps < 1 or scenario.maximum_late_steps < scenario.minimum_late_steps:
        raise ValueError(f'Scenario {scenario.name} has an invalid delay range.')
    if scenario.device_extra_delay_steps < 0 or scenario.device_extra_delay_jitter < 0:
        raise ValueError(f'Scenario {scenario.name} has a negative device-level delay.')

def _sample_burst_state(generator: np.random.Generator, length: int, device_count: int, scenario: Scenario) -> np.ndarray:
    """Return device or shared Markov bad-channel states at every update step."""
    bad = np.zeros((length, device_count), dtype=bool)
    if scenario.burst_good_to_bad_probability == 0.0:
        return bad
    if scenario.shared_burst_state:
        for time_index in range(1, length):
            previous = bool(bad[time_index - 1, 0])
            if previous:
                current = bool(generator.random() >= scenario.burst_bad_to_good_probability)
            else:
                current = bool(generator.random() < scenario.burst_good_to_bad_probability)
            bad[time_index] = current
        return bad
    for time_index in range(1, length):
        previous = bad[time_index - 1]
        become_bad = ~previous & (generator.random(device_count) < scenario.burst_good_to_bad_probability)
        remain_bad = previous & (generator.random(device_count) >= scenario.burst_bad_to_good_probability)
        bad[time_index] = become_bad | remain_bad
    return bad

def simulate_packets(*, time_steps: int, device_count: int, parameter_count: int, scenario: Scenario, seed: int, deadline_steps: int, straggler_device_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate scalar-packet delivery in aligned ``[time, device, parameter]`` coordinates.

    Returns an event code, a virtual latency in update steps (``-1`` for a
    dropped packet), and a boolean current-observation mask.  A delay greater
    than ``deadline_steps`` is represented as a late event even when it is
    eventually delivered.
    """
    _validate_scenario(scenario)
    if min(time_steps, device_count, parameter_count) < 1:
        raise ValueError('time_steps, device_count, and parameter_count must all be positive.')
    if not 0 <= straggler_device_index < device_count:
        raise ValueError('straggler_device_index is outside the device range.')
    generator = np.random.default_rng(seed)
    shape = (time_steps, device_count, parameter_count)
    event_code = np.full(shape, EVENT_ON_TIME, dtype=np.uint8)
    latency_steps = np.zeros(shape, dtype=np.int16)
    bad_channel = _sample_burst_state(generator, time_steps, device_count, scenario)
    late_probability = np.full((time_steps, device_count), scenario.late_probability, dtype=np.float32)
    drop_probability = np.full((time_steps, device_count), scenario.drop_probability, dtype=np.float32)
    if bad_channel.any():
        late_probability[bad_channel] = scenario.burst_late_probability
        drop_probability[bad_channel] = scenario.burst_drop_probability
    for device_index in range(device_count):
        packet_draw = generator.random((time_steps, parameter_count), dtype=np.float32)
        device_drop = drop_probability[:, device_index, None]
        device_late = late_probability[:, device_index, None]
        dropped = packet_draw < device_drop
        late = ~dropped & (packet_draw >= device_drop) & (packet_draw < device_drop + device_late)
        device_latency = latency_steps[:, device_index, :]
        if scenario.device_extra_delay_steps and device_index == straggler_device_index:
            device_jitter = generator.integers(0, scenario.device_extra_delay_jitter + 1, size=(time_steps, parameter_count), dtype=np.int16)
            device_latency[:] = scenario.device_extra_delay_steps + device_jitter
            late = ~dropped
        if late.any():
            sampled_delay = generator.integers(scenario.minimum_late_steps, scenario.maximum_late_steps + 1, size=(time_steps, parameter_count), dtype=np.int16)
            device_latency[late] = np.maximum(device_latency[late], sampled_delay[late])
        device_latency[dropped] = DROP_LATENCY
    event_code[latency_steps > deadline_steps] = EVENT_LATE
    event_code[latency_steps == DROP_LATENCY] = EVENT_DROPPED
    on_time_mask = event_code == EVENT_ON_TIME
    return (event_code, latency_steps, on_time_mask)
