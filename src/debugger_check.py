def calculate_batch_loss(losses: list[float]) -> float:
    """
    Aggregates the total loss over a batch of training iterations.

    Args:
        losses (List[float]): A list containing loss values from a neural network batch.

    Returns:
        float: The cumulative sum of the losses.
    """
    total_loss: float = 0.0

    # INTENTIONAL BUG: Iterating incorrectly, similar to the video's demonstration.
    # We are erroneously adding the loop index 'i' instead of the actual loss value 'losses[i]'.
    for i in range(len(losses)):
        total_loss += i

    return total_loss


# Mock data representing model losses during a training step
batch_losses: list[float] = [0.45, 0.32, 0.28, 0.15]

# Executing the function
final_loss: float = calculate_batch_loss(losses=batch_losses)
print(f"Total Loss: {final_loss}")
