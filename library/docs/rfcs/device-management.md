# Device Management (Deferred)

| Field | Value |
|-------|-------|
| Status | Placeholder |
| Parent | model-contract.md |
| Tracking issue | TBD |

---

## Context

The model contract RFC explicitly defers device management. This placeholder captures the current state and open threads for a future discussion.

## Current behaviour

- **OV models**: take `device: str` in `__init__` (e.g. "AUTO", "CPU", "GPU.0"). OpenVINO handles device selection internally.
- **Torch models**: use standard `.to(device)`. No abstraction on top.
- **Backend app**: `DeviceService` in `application/backend/app/runtime/services/device.py` manages available devices. Stays as-is for now.

## What we might want later

- A `DevicePool` that tracks which devices are busy (multiple models sharing a GPU).
- Hot-migration between devices without re-instantiating the model.
- Unified device naming across OV and torch (e.g. "gpu:0" → maps to "GPU.0" for OV, "cuda:0" for torch).

## Why defer

None of the above has a concrete consumer requesting it today. Adding an abstraction preemptively would constrain OV and torch in ways we can't predict without real usage data.

## Next step

When we hit a real use case (e.g. multi-model scheduling on limited GPUs), open a proper RFC branching from here.
