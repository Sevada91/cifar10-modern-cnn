import matplotlib.pyplot as plt
from pathlib import Path

def plot_history(history: dict, run_name=None, save_dir="../reports/figures"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Loss ----
    ax[0].plot(epochs, history["train_loss"], label="Train Loss")
    ax[0].plot(epochs, history["val_loss"], label="Val Loss")
    ax[0].set_title(f"Loss - {run_name}" if run_name else "Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[0].grid(True)

    # ---- Accuracy ----
    ax[1].plot(epochs, history["train_acc"], label="Train Acc")
    ax[1].plot(epochs, history["val_acc"], label="Val Acc")
    ax[1].set_title(f"Accuracy - {run_name}" if run_name else "Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()
    ax[1].grid(True)

    fig.tight_layout()

    if run_name:
        fig_path = save_dir / f"{run_name}_training_curves.png"
    else:
        fig_path = save_dir / "training_curves.png"

    plt.savefig(fig_path)
    plt.show()

    print(f"Saved plot to {fig_path}")
