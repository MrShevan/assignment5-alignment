import torch
from torch import Tensor

from transformers import PreTrainedTokenizerBase


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    entropy = -torch.sum(torch.exp(log_probs) * log_probs, dim=-1)
    return entropy


def pad(seq, pad_id, max_length):
    """Pad the sequence untill the maximum length."""
    return seq + [pad_id] * (max_length - len(seq))


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    full_tokens_list = []
    response_mask_list = []
    max_len = 0

    for prompt, output in zip(prompt_strs, output_strs):
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        output_tokens = tokenizer(output, add_special_tokens=False)["input_ids"]

        full_tokens = prompt_tokens + output_tokens

        response_mask = [0] * (len(prompt_tokens)) + [1] * (len(output_tokens))
        max_len = max(max_len, len(full_tokens))

        full_tokens_list.append(full_tokens)
        response_mask_list.append(response_mask)

    return {
        "input_ids": torch.tensor([
            pad(input_ids, tokenizer.pad_token_id, max_len)[:-1] for input_ids in full_tokens_list
        ]),
        "labels": torch.tensor([
            pad(input_ids, tokenizer.pad_token_id, max_len)[1:] for input_ids in full_tokens_list
        ]),
        "response_mask": torch.tensor([
            pad(response_mask, 0, max_len)[1:] for response_mask in response_mask_list
        ]),
    }


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    logits = model(input_ids).logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
    
    result = {"log_probs": log_probs}
    if return_token_entropy:
        entropy = compute_entropy(logits)
        result["token_entropy"] = entropy
    return result


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.
    """
    response_token_count = response_mask.sum().detach()
    total_log_probs = masked_normalize(
        policy_log_probs, 
        response_mask, 
        normalize_constant=normalize_constant, 
        dim=None
    )
    loss = -total_log_probs
    scaled_loss = loss / gradient_accumulation_steps
    scaled_loss.backward()
    
    unnormalized_loss = - (total_log_probs * normalize_constant).detach()
    metadata = {
        "unnormalized_loss": unnormalized_loss,
        "response_token_count": response_token_count
    }
    return scaled_loss.detach(), metadata


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    mask = mask.to(tensor.dtype)
    masked_tensor = tensor * mask
    if dim is not None:
        sum_tensor = masked_tensor.sum(dim=dim)
    else:
        sum_tensor = masked_tensor.sum()
    return sum_tensor / normalize_constant


def sft_train_loop():
    pass
