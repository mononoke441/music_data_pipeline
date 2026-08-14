"""Shared MuQ/MusicFM chunk batching without padding or model replicas."""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import torch


@torch.inference_mode()
def extract_muq_musicfm_chunks(
    chunks: Sequence[torch.Tensor],
    muq: Any,
    musicfm: Any,
    *,
    batch_size: int = 1,
    empty_cuda_cache: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Extract ordered chunk embeddings, batching only equal-length chunks.

    No padding is introduced because it can alter the representations near a
    real track boundary.  A final short chunk is therefore inferred in its own
    shape group.  Returned tensors retain the original per-chunk ``[1,T,D]``
    contract used by SongFormer and the Instrumental overlap-add decoder.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not chunks:
        return [], []
    for chunk in chunks:
        if chunk.ndim != 1:
            raise ValueError("each audio chunk must be a one-dimensional waveform")

    muq_results: List[torch.Tensor] = []
    musicfm_results: List[torch.Tensor] = []
    cursor = 0
    while cursor < len(chunks):
        length = int(chunks[cursor].numel())
        finish = cursor + 1
        while (
            finish < len(chunks)
            and finish - cursor < batch_size
            and int(chunks[finish].numel()) == length
        ):
            finish += 1
        batch = torch.stack(list(chunks[cursor:finish]), dim=0)

        muq_output = muq(batch, output_hidden_states=True)
        muq_hidden = muq_output["hidden_states"][10]
        if empty_cuda_cache:
            del muq_output
            torch.cuda.empty_cache()
        _, musicfm_states = musicfm.get_predictions(batch)
        musicfm_hidden = musicfm_states[10]
        if empty_cuda_cache:
            del musicfm_states
            torch.cuda.empty_cache()
        expected = finish - cursor
        if int(muq_hidden.shape[0]) != expected or int(musicfm_hidden.shape[0]) != expected:
            raise RuntimeError(
                "MuQ/MusicFM batch cardinality mismatch: "
                f"expected={expected}, muq={muq_hidden.shape[0]}, "
                f"musicfm={musicfm_hidden.shape[0]}"
            )
        for index in range(expected):
            muq_results.append(muq_hidden[index : index + 1])
            musicfm_results.append(musicfm_hidden[index : index + 1])
        cursor = finish

    return muq_results, musicfm_results
