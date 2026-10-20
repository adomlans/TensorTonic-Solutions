def pipeline_bubble_utilization(num_stages, num_microbatches, include_backward):
    """
    Returns: dictionary containing the schedule and utilization metrics
    """
    wave_slots = num_microbatches + num_stages - 1
    schedule = []
    for time_slot in range(wave_slots):
        schedule.append([
            f"F{time_slot - stage}" if 0 <= time_slot - stage < num_microbatches else None
            for stage in range(num_stages)
        ])

    if include_backward:
        for offset in range(wave_slots):
            slot = []
            for stage in range(num_stages):
                microbatch = num_microbatches - 1 - (offset - (num_stages - 1 - stage))
                slot.append(f"B{microbatch}" if 0 <= microbatch < num_microbatches else None)
            schedule.append(slot)

    total_slots = len(schedule) * num_stages
    bubble_slots = sum(cell is None for slot in schedule for cell in slot)
    return {
        "schedule": schedule,
        "bubble_slots": bubble_slots,
        "total_slots": total_slots,
        "utilization": float((total_slots - bubble_slots) / total_slots),
    }
