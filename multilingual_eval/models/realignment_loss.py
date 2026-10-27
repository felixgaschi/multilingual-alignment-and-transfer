from typing import List

import torch
import torch.nn.functional as F

from multilingual_eval.models.utils import remove_batch_dimension, sum_ranges_and_put_together, build_mask_from_lengths
from multilingual_eval.models.contexts import DumbContext


def compute_realignment_loss(
    encoder,
    realignment_transformation,
    realignment_layers: List[int],
    strong_alignment=False,
    realignment_temperature=0.1,
    realignment_coef=1.0,
    no_backward_for_source=False,
    regularization_lambda=1.0,
    initial_model=None,
    realignment_loss="contrastive",
    realignment_method="token", # "token" or "sentence"
    train_only_mapping=False,
    left_context=None,
    right_context=None,
    # sentences from left language
    left_input_ids=None,
    left_attention_mask=None,
    left_token_type_ids=None,
    left_position_ids=None,
    left_head_mask=None,
    left_inputs_embeds=None,
    left_lang_id=None,
    # sentences from right language
    right_input_ids=None,
    right_attention_mask=None,
    right_token_type_ids=None,
    right_position_ids=None,
    right_head_mask=None,
    right_inputs_embeds=None,
    right_lang_id=None,
    # alignment labels
    alignment_left_ids=None,  # [word_id] -> [[word_id | -1]]
    alignment_left_positions=None,  # [[batch, start, end]] -> [[[start, end]]]
    alignment_right_ids=None,  # [word_id]
    alignment_right_positions=None,  # [[batch, start, end]]
    alignment_nb=None,  # [[i]]
    alignment_left_length=None,  # [[i]]
    alignment_right_length=None,  # [[i]]
    # lang_probs
    lang_probs=None,
    lang_ids=None,
    meta_loss_type="micro",
):
    """
    Fonction for computing the realignment loss on a batch of realignment task

    Arguments:
    - encoder: the encoder of the model to build the representation of each sentence (typically model.bert)
    - realignment_tranformation: a transformation to apply to all transformation before realignment (typically some linear layers)
    - realignment_layers: the layers of the encoder on which to perform the realignemnt (typically [-1])
    - strong_alignment: whether to perform strong or weak realignment (only for contrastive method)
    - realignment_temperature: temperature in the softmax for contrastive learning
    - realignment_coef: a coefficient to apply to the loss
    - no_backward_for_source: whether to deactivate backward pass on the encoder applied on source (can work as a regularization method), default to False (because it doesn't really work)
    - regularization_lambda: coefficient to apply to the regularization term (if there is one)
    - initial_model: copy of the model to use in the regularization (if there is one)
    - realignment_loss: loss used, either contrastive or l2
    - train_only_mapping: default False, desactivates backpropagation through encoder (but not through mapping if there is one)
    - left_*: * for the left sentence
    - left_lang_id: id of the language of the left sentence
    - right_*: * for the right sentence
    - right_lang_id: id of the language of the right sentence
    - alignment_left_positions: position range of all words in the left sentence (in term of subword)
    - alignment_right_positions: same for the right sentence
    - alignment_left_ids: index of aligned word in alignment_left_positions
    - alignment_right_ids: index of corresponding aligned words in alignment_right_positions
    - alignment_nb: the number of aligned pair (usefull for truncation)
    - alignment_left_length: the number of word in alignment_left_positions (usefull for truncation)
    - alignment_right_length: the same for the right sentence
    """

    left_context_manager = left_context or (torch.no_grad() if no_backward_for_source else DumbContext())

    with left_context_manager:
        left_output = encoder(
            left_input_ids,
            attention_mask=left_attention_mask,
            head_mask=left_head_mask,
            inputs_embeds=left_inputs_embeds,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
            **(
                {"token_type_ids": left_token_type_ids} if left_token_type_ids is not None else {}
            ),  # This is because of distilbert
            **({"position_ids": left_position_ids} if left_position_ids is not None else {}),
            **(
                {
                    "lang_id": left_lang_id,
                    "train_only_mapping": train_only_mapping
                    and not no_backward_for_source,  # don't want nested torch.no_grad
                }
                if left_lang_id is not None or right_lang_id is not None
                else {}
            ),
        )

        left_hidden_states = left_output.hidden_states

    if initial_model is not None:
        initial_output = initial_model(
            left_input_ids,
            attention_mask=left_attention_mask,
            head_mask=left_head_mask,
            inputs_embeds=left_inputs_embeds,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
            **({"token_type_ids": left_token_type_ids} if left_token_type_ids is not None else {}),
            **({"position_ids": left_position_ids} if left_position_ids is not None else {}),
            **(
                {"lang_id": left_lang_id, "train_only_mapping": train_only_mapping}
                if left_lang_id is not None or right_lang_id is not None
                else {}
            ),
        )

        initial_hidden_states = initial_output.hidden_states
    else:
        initial_hidden_states = None

    right_context = right_context or DumbContext()

    with right_context:
        right_output = encoder(
            right_input_ids,
            attention_mask=right_attention_mask,
            head_mask=right_head_mask,
            inputs_embeds=right_inputs_embeds,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
            **({"token_type_ids": right_token_type_ids} if right_token_type_ids is not None else {}),
            **({"position_ids": right_position_ids} if right_position_ids is not None else {}),
            **(
                {"lang_id": right_lang_id, "train_only_mapping": train_only_mapping}
                if left_lang_id is not None or right_lang_id is not None
                else {}
            ),
        )

        right_hidden_states = right_output.hidden_states


    return compute_loss_from_representations(
        left_hidden_states, 
        right_hidden_states,
        realignment_transformation,
        realignment_layers,
        strong_alignment=strong_alignment,
        realignment_temperature=realignment_temperature,
        realignment_coef=realignment_coef,
        regularization_lambda=regularization_lambda,
        realignment_loss=realignment_loss,
        realignment_method=realignment_method,
        initial_model=initial_model,
        initial_hidden_states=initial_hidden_states,
        alignment_left_ids=alignment_left_ids,
        alignment_left_positions=alignment_left_positions,
        alignment_right_ids=alignment_right_ids,
        alignment_right_positions=alignment_right_positions,
        alignment_nb=alignment_nb,
        alignment_left_length=alignment_left_length,
        alignment_right_length=alignment_right_length,
        lang_probs=lang_probs,
        lang_ids=lang_ids,
        meta_loss_type=meta_loss_type
    )

def compute_loss_from_representations(
    left_hidden_states, 
    right_hidden_states,
    realignment_transformation,
    realignment_layers: List[int],
    strong_alignment=False,
    realignment_temperature=0.1,
    realignment_coef=1.0,
    regularization_lambda=1.0,
    realignment_loss="contrastive",
    realignment_method="token",
    initial_model=None,
    initial_hidden_states=None,
    # alignment labels
    alignment_left_ids=None, 
    alignment_left_positions=None,
    alignment_right_ids=None,
    alignment_right_positions=None, 
    alignment_nb=None,
    alignment_left_length=None, 
    alignment_right_length=None, 
    # lang_probs
    lang_probs=None,
    lang_ids=None,
    meta_loss_type="micro",
):
    the_device = left_hidden_states[0].device
    total_loss, meta_loss, per_lang_losses = None, None, None

    for layer in realignment_layers:
        # Inspired by https://github.com/shijie-wu/crosslingual-nlp/blob/780f738df2b75f653aaaf11b9f513850fe11ba36/src/model/aligner.py#L139

        if realignment_method == "sentence":
            aligned_left_repr = left_hidden_states[layer] * build_mask_from_lengths(
                alignment_left_length, left_hidden_states[layer].shape[1]
            ).to(the_device).unsqueeze(-1)
            aligned_right_repr = right_hidden_states[layer] * build_mask_from_lengths(
                alignment_right_length, right_hidden_states[layer].shape[1]
            ).to(the_device).unsqueeze(-1)
            aligned_left_repr = aligned_left_repr.sum(dim=1) / alignment_left_length.unsqueeze(-1)
            aligned_right_repr = aligned_right_repr.sum(dim=1) / alignment_right_length.unsqueeze(-1)
        elif realignment_method == "token":
            aligned_left_repr = sum_ranges_and_put_together(
                left_hidden_states[layer],
                alignment_left_positions,
                ids=alignment_left_ids,
            )
            aligned_right_repr = sum_ranges_and_put_together(
                right_hidden_states[layer],
                alignment_right_positions,
                ids=alignment_right_ids,
            )
            aligned_left_repr = remove_batch_dimension(aligned_left_repr, alignment_nb)
            aligned_right_repr = remove_batch_dimension(aligned_right_repr, alignment_nb)
        else:
            raise NotImplementedError(
                f"Unknown realignment_method for compute_realignment_loss: {realignment_method}"
            )

        if realignment_transformation is not None:
            aligned_left_repr = realignment_transformation(aligned_left_repr)
            aligned_right_repr = realignment_transformation(aligned_right_repr)

        if realignment_loss == "l2":
            loss = F.mse_loss(aligned_left_repr, aligned_right_repr)
            if lang_probs and lang_ids:
                pairwise_loss = F.mse_loss(aligned_left_repr, aligned_right_repr, reduction='none')
                example_loss = pairwise_loss.mean(dim=1)
                sample_weights = lang_probs[lang_ids]
                meta_loss = (example_loss * sample_weights).mean()
        elif realignment_loss == "contrastive":
            if realignment_method == "sentence":
                all_left_repr = aligned_left_repr
                all_right_repr = aligned_right_repr

                N = aligned_left_repr.shape[0]
                left_goal = torch.arange(
                    0, N, 1, device=the_device
                ) + N
                right_goal = torch.arange(
                    0, N, 1, device=the_device
                )
            elif realignment_method == "token":
                all_left_repr = sum_ranges_and_put_together(
                    left_hidden_states[layer], alignment_left_positions
                )
                all_right_repr = sum_ranges_and_put_together(
                    right_hidden_states[layer],
                    alignment_right_positions,
                )

                if realignment_transformation is not None:
                    all_left_repr = realignment_transformation(all_left_repr)
                    all_right_repr = realignment_transformation(all_right_repr)

                all_left_repr = remove_batch_dimension(all_left_repr, alignment_left_length)
                all_right_repr = remove_batch_dimension(all_right_repr, alignment_right_length)

                right_cumul_length = torch.cat(
                    (
                        torch.tensor([0], dtype=torch.long, device=the_device),
                        torch.cumsum(alignment_right_length, 0),
                    )
                )
                left_cumul_length = torch.cat(
                    (
                        torch.tensor([0], dtype=torch.long, device=the_device),
                        torch.cumsum(alignment_left_length, 0),
                    )
                )

                left_goal = torch.cat(
                    (
                        *[
                            all_left_repr.shape[0]
                            + right_cumul_length[b]
                            + alignment_right_ids[b][: alignment_nb[b]]
                            for b in range(alignment_left_ids.shape[0])
                        ],
                    )
                )
                right_goal = torch.cat(
                    (
                        *[
                            left_cumul_length[b] + alignment_left_ids[b][: alignment_nb[b]]
                            for b in range(alignment_right_ids.shape[0])
                        ],
                    )
                )
            else:
                raise NotImplementedError(
                    f"Unknown realignment_method for compute_realignment_loss: {realignment_method}"
                )

            

            aligned_reprs = torch.cat((aligned_left_repr, aligned_right_repr))
            all_reprs = torch.cat((all_left_repr, all_right_repr))
            sim = torch.matmul(aligned_reprs, all_reprs.transpose(0, 1))
            aligned_norms = aligned_reprs.norm(dim=1, keepdim=True)
            all_norms = all_reprs.norm(dim=1, keepdim=True)

            sim /= aligned_norms
            sim /= all_norms.transpose(0, 1)
            sim /= realignment_temperature

            if not strong_alignment:
                # remove same-language similarities
                sim[: aligned_left_repr.shape[0], : all_left_repr.shape[0]] -= 1e6
                sim[aligned_left_repr.shape[0] :, all_left_repr.shape[0] :] -= 1e6
            else:
                # remove (x,x) similarities
                sim[
                    torch.arange(0, aligned_left_repr.shape[0], 1, device=the_device),
                    right_goal,
                ] -= 1e6
                sim[
                    torch.arange(
                        aligned_right_repr.shape[0],
                        2 * aligned_right_repr.shape[0],
                        1,
                        device=the_device,
                    ),
                    left_goal,
                ] -= 1e6

            logits = F.log_softmax(sim, dim=-1)
            goal = torch.cat((left_goal, right_goal))

            loss = F.nll_loss(logits, goal)
            if lang_ids is not None:
                ## detach variable to seperate from model param
                logits_meta, goal_meta = logits.detach().clone(), goal.detach().clone()
                per_example_loss = F.nll_loss(logits_meta, goal_meta, reduction='none')

                # DEBUG: similar grad issue
                # print("lang_probs:", lang_probs.detach().cpu().numpy())
                # print("unique lang_ids in batch:", torch.unique(lang_ids, return_counts=True))
                # # lang_ids correspond to right-side examples
                # per_loss_right = per_example_loss[aligned_left_repr.shape[0]:]  # right half
                # for lang in torch.unique(lang_ids):
                #     mask_lang = (lang_ids == lang).to(per_loss_right.device)
                #     if mask_lang.sum() > 0:
                #         avg = per_loss_right[mask_lang].mean().item()
                #         print(f"lang {lang.item()} avg right-loss:", avg)

                # Weight for source lang samples in a batch will be similar to weights for right lang samples
                full_lang_ids = torch.cat([lang_ids, lang_ids], dim=0)

                # Compute per-language average losses (used by UCB)
                per_lang_losses = {}
                for lid in torch.unique(full_lang_ids):
                    mask = (full_lang_ids == lid)
                    per_lang_losses[lid.item()] = per_example_loss[mask].mean().item()

                # Compute meta_loss for gradient-based meta-learning (only when lang_probs provided)
                if lang_probs is not None:
                    sample_weights = lang_probs[full_lang_ids]
                    # right_weights = lang_probs[lang_ids]
                    # sample_weights = torch.cat([right_weights, right_weights], dim=0)

                    # Macro avg
                    if meta_loss_type == "macro":
                        lang_counts = torch.bincount(full_lang_ids)
                        ## weight of each example is divieded by the count of its language
                        sample_weights = sample_weights / lang_counts[full_lang_ids].float().clamp(min=1.0)
                    elif meta_loss_type == "micro_log":
                        sample_weights = torch.log(lang_probs[full_lang_ids] + 1e-9)

                    meta_loss = (per_example_loss * sample_weights).sum() / sample_weights.sum().clamp(min=1e-12)

        else:
            raise NotImplementedError(
                f"Unknown realignment_loss for compute_realignment_loss: {realignment_loss}"
            )

        if initial_model is not None:
            loss += regularization_lambda * F.mse_loss(
                left_hidden_states[layer], initial_hidden_states[layer]
            )

        if total_loss is None:
            total_loss = (realignment_coef / len(realignment_layers)) * loss
        else:
            total_loss += (realignment_coef / len(realignment_layers)) * loss

    return total_loss, meta_loss, per_lang_losses
