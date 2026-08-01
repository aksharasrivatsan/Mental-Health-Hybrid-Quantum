import argparse

from hybrid_metrics import add_common_training_args, run_training

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune VGG16 block 5 with a PennyLane quantum layer on MFCC image folders."
    )
    return add_common_training_args(
        parser,
        default_epochs=30,
        default_learning_rate=0.00001,
        default_plot_path="vgg16_fine_tuned_quantum_accuracy.png",
    ).parse_args()


def main():
    args = parse_args()
    print("Training the Hybrid Quantum Fine-Tuned VGG16 model...")
    run_training(
        args,
        fine_tune_block5=True,
        title="Hybrid Quantum VGG16 Block 5 Fine-Tuning Results",
    )


if __name__ == "__main__":
    main()
