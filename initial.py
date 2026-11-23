import math

# EVOLVE-BLOCK-START
def compute_lr(step: int, total_steps: int):
    """
    CURRENT STATE:
    - Phase 1 (warmup): LR increases from max_lr/div_factor to max_lr
    - Phase 2 (annealing): LR decreases from max_lr to max_lr/(div_factor*final_div_factor)
    - Uses cosine annealing
    Args:
        step: Current training step (0-based, from 0 to total_steps-1).
        total_steps: Total number of training steps (corresponds to number of epochs).
    Additional parameters:
        max_lr: Maximum learning rate (peak of the cycle)
        pct_start: Fraction of total steps for warmup phase
    Returns:
        float: Learning rate for this step
    """
    max_lr = 0.01
    pct_start = 0.3
    div_factor = 25.0
    final_div_factor = 1e4
    initial_lr = max_lr / div_factor
    min_lr = initial_lr / final_div_factor

    phase1_end_step = float(pct_start * total_steps) - 1

    def anneal(start, end, pct):
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    if step <= phase1_end_step:
        start_step = 0
        end_step = phase1_end_step
        start_lr = initial_lr
        end_lr = max_lr
    else:
        start_step = phase1_end_step + 1
        end_step = total_steps - 1
        start_lr = max_lr
        end_lr = min_lr

    pct = (step - start_step) / (end_step - start_step)
    pct = max(0.0, min(1.0, pct))

    return anneal(start_lr, end_lr, pct)
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """
    Main function called by evaluate.py.
    Runs the scheduler for all steps and returns the LR schedule.

    This function is flexible and can pass any parameters to compute_lr.
    The LLM can add new parameters to compute_lr, and they will be passed here.

    Args:
        **kwargs: All parameters for compute_lr, including:
            - total_steps (int): Total number of training steps correspond to number of training epochs

    Returns:
        List[float]: LR values for each step
    """

    # Extract required parameters
    total_steps = kwargs.get('total_steps', 50)

    # Generate LR schedule
    lr_schedule = []
    for step in range(total_steps):
        lr = compute_lr(step=step, total_steps=total_steps, **{
            k: v for k, v in kwargs.items()
            if k not in ['total_steps']
        })
        lr_schedule.append(float(lr))

    return lr_schedule


