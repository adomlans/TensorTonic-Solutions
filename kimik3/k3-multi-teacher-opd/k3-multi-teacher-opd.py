import torch

def multi_teacher_opd_reward(student_logits, teacher_logits, domain_indices, effort_indices, sampled_tokens, clip_threshold):
    """
    Returns: clipped token rewards and selected teacher token log probabilities.
    """
    batch_size = student_logits.shape[0]
    batch_indices = torch.arange(batch_size, device=student_logits.device)
    selected_teacher_logits = teacher_logits[domain_indices, effort_indices, batch_indices]
    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    teacher_log_probs = torch.log_softmax(selected_teacher_logits, dim=-1)
    student_token_log_probs = student_log_probs.gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
    teacher_token_log_probs = teacher_log_probs.gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
    reward = (teacher_token_log_probs - student_token_log_probs).detach()
    reward = torch.clamp(reward, -clip_threshold, clip_threshold)
    return reward, teacher_token_log_probs
