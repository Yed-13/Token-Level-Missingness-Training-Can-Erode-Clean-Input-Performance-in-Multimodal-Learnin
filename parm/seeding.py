"""Stable seed derivation shared by independently launched experiment jobs."""

import hashlib


def stable_eval_seed(dataset: str, regime: str, repeat: int,
                     stream: str = "mask", base: int = 20260915) -> int:
    """Return a process-stable seed for one evaluation stream.

    Python salts its built-in ``hash`` independently for each process. A short
    BLAKE2 digest instead gives the same seed on every host and Python process.
    """
    payload = f"{dataset}\0{regime}\0{repeat}\0{stream}".encode("utf-8")
    offset = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(),
                            "little")
    return (base + offset) % (2**63 - 1)
