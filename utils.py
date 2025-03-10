import boto3
from boto3.s3.transfer import TransferConfig
from tqdm import tqdm
import os

def upload_file_to_s3(file_path, bucket_name, s3_prefix):
    

    class ProgressPercentage(object):
        def __init__(self, filename):
            self._filename = filename
            self._size = float(os.path.getsize(filename))
            self._seen_so_far = 0
            self._pbar = tqdm(total=self._size, unit='B', unit_scale=True, desc=f"Uploading {os.path.basename(filename)}")

        def __call__(self, bytes_amount):
            self._seen_so_far += bytes_amount
            self._pbar.update(bytes_amount)

    s3_client = boto3.client('s3')
    file_name = os.path.basename(file_path)
    s3_path = f"{s3_prefix}/{file_name}"
    
    # Configure multipart upload
    config = TransferConfig(
        multipart_threshold=1024 * 25,  # 25MB
        max_concurrency=10,
        multipart_chunksize=1024 * 25,  # 25MB
        use_threads=True
    )
    
    try:
        s3_client.upload_file(
            file_path, 
            bucket_name, 
            s3_path,
            Config=config,
            Callback=ProgressPercentage(file_path)
        )
        return f"s3://{bucket_name}/{s3_path}"
    except Exception as e:
        print(f"Failed to upload {file_path} to S3: {str(e)}")
        return None
    
max_lr = 1e-3
warmup_steps = 10
max_steps = 25000
import math
def get_lr_lambda(current_step, warmup_steps, max_steps, max_lr):
    """
    Learning rate scheduler with:
    1. Linear warmup
    2. Cosine decay
    3. Minimum learning rate of 10% of max_lr
    """
    min_lr = max_lr * 0.1  # Minimum learning rate (10% of max_lr)
    
    if current_step < warmup_steps:
        # Linear warmup
        return max_lr * (current_step + 1) / warmup_steps
    elif current_step > max_steps:
        # After max_steps, return minimum learning rate
        return min_lr
    else:
        # Cosine decay between warmup_steps and max_steps
        decay_ratio = (current_step - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)


def plot_lr_schedule():
    """
    Helper function to visualize the learning rate schedule
    """
    import matplotlib.pyplot as plt
    steps = list(range(0, max_steps + 100))
    lrs = [get_lr_lambda(step, warmup_steps, max_steps, max_lr) for step in steps]
    
    plt.figure(figsize=(10, 5))
    plt.plot(steps, lrs)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Steps')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.show()

def plot_training_loss(log_file_path, output_path=None):
    """
    Parse a training log file and plot the running average loss against batch steps.
    Also adds a trend line to visualize the overall training progress.
    
    Args:
        log_file_path (str): Path to the training log file
        output_path (str, optional): Path to save the plot as PNG. If None, displays the plot instead.
    """
    import re
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import curve_fit
    
    # Regular expression to extract batch number and loss
    pattern = r"Batch (\d+), Running Avg Loss: ([0-9.]+)"
    
    steps = []
    losses = []
    
    # Read and parse the log file
    with open(log_file_path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                batch_num = int(match.group(1))
                loss = float(match.group(2))
                steps.append(batch_num)
                losses.append(loss)
    
    if not steps:
        print("No loss data found in the log file.")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(steps, losses, 'b-', alpha=0.5, label='Running Avg Loss')
    
    # Add trend line (using polynomial fit)
    def poly_func(x, a, b, c):
        return a * x**2 + b * x + c
    
    # Convert to numpy arrays for curve fitting
    x_array = np.array(steps)
    y_array = np.array(losses)
    
    # Fit the curve
    try:
        popt, _ = curve_fit(poly_func, x_array, y_array)
        x_line = np.linspace(min(steps), max(steps), 1000)
        y_line = poly_func(x_line, *popt)
        plt.plot(x_line, y_line, 'r-', label='Trend Line')
    except Exception as e:
        print(f"Could not fit trend line: {e}")
        # Fallback to simple moving average for trend
        window_size = min(len(steps) // 10, 100) if len(steps) > 100 else len(steps) // 2
        if window_size > 0:
            moving_avg = np.convolve(y_array, np.ones(window_size)/window_size, mode='valid')
            plt.plot(steps[window_size-1:], moving_avg, 'r-', label='Moving Average Trend')
    
    # Add labels and title
    plt.xlabel('Batch Number')
    plt.ylabel('Running Average Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True)
    plt.legend()
    
    # Add min and max loss annotations
    min_loss = min(losses)
    min_idx = losses.index(min_loss)
    max_loss = max(losses)
    max_idx = losses.index(max_loss)
    
    plt.annotate(f'Min: {min_loss:.5f}', 
                xy=(steps[min_idx], min_loss),
                xytext=(steps[min_idx], min_loss*1.05),
                arrowprops=dict(facecolor='green', shrink=0.05),
                fontsize=10)
    
    plt.annotate(f'Max: {max_loss:.5f}', 
                xy=(steps[max_idx], max_loss),
                xytext=(steps[max_idx], max_loss*0.95),
                arrowprops=dict(facecolor='red', shrink=0.05),
                fontsize=10)
    
    # Save or show the plot
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # plot_lr_schedule()
    plot_training_loss("training.log", "train_loss.png")