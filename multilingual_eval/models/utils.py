from typing import Optional
import torch


def sum_ranges_and_put_together(
    reprs: torch.Tensor, positions: torch.Tensor, ids: Optional[torch.Tensor] = None
):
    """
    In a Transformer output (batch_size, length, dim), this function sums ranges of consecutive tokens
    indicated by position, and optionally, by a subset of these positions indicated by id
    - reprs: batch of sequence representations (batch_size, length, dim)
    - positions: (batch_size, _, 2) which give a list of range (start, end) for each sample of the batch (indicating entire words for examples)
    - ids: (optional) (batch_size, _) ids to select positions to keep
    """
    batch_size = positions.shape[0]
    nb_range = positions.shape[1] if ids is None else ids.shape[1]
    dim = reprs.shape[2]

    res = torch.zeros((batch_size, nb_range, dim), device=reprs.device, dtype=reprs.dtype)

    for b in range(batch_size):
        for i in range(nb_range):
            range_id = i if ids is None else ids[b, i]
            res[b, i] = torch.sum(
                reprs[
                    b,
                    positions[b, range_id][0] : positions[b, range_id][1],
                ],
                0,
            )

    return res


def remove_batch_dimension(input: torch.Tensor, lengths: torch.Tensor):
    """
    Truncates batche inputs according to batch lengths and concatenate them, removing the batch dimension
    """
    return torch.cat((*[input[b][: lengths[b]] for b in range(input.shape[0])],))

def build_mask_from_lengths(lengths: torch.Tensor, max_length: int):
    """
    Builds a mask from lengths, where the mask is True for positions that are not padded
    - lengths: (batch_size,) tensor of lengths
    - max_length: maximum length of the sequences in the batch

    Returns:
    - mask: (batch_size, max_length) tensor of boolean values, where True indicates a valid position
    """
    mask = torch.arange(max_length, device=lengths.device).expand(lengths.shape[0], max_length) < lengths.unsqueeze(1)
    return mask