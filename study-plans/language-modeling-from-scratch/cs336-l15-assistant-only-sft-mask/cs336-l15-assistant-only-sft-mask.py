import torch

def assistant_only_sft_mask(input_ids, role_ids, attention_mask,
                            assistant_role=1):
    """
    Returns: dictionary containing shifted labels and assistant loss mask
    """
    labels = torch.full_like(input_ids, -100)
    loss_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
    if input_ids.shape[1] > 1:
        active = (
            attention_mask[:, :-1]
            & attention_mask[:, 1:]
            & (role_ids[:, 1:] == assistant_role)
        )
        labels[:, :-1] = torch.where(
            active, input_ids[:, 1:], torch.full_like(input_ids[:, 1:], -100)
        )
        loss_mask[:, :-1] = active
    return {"labels": labels, "loss_mask": loss_mask}