import argparse

from hybrid_quantum_tf import add_common_training_args, run_training


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a frozen-VGG16 hybrid quantum classifier on MFCC image folders."
    )
    return add_common_training_args(
        parser,
        default_epochs=30,
        default_learning_rate=0.0001,
        default_plot_path="vgg16_quantum_accuracy.png",
    ).parse_args()


def main():
    args = parse_args()
    print("Training the Hybrid Quantum VGG16 baseline...")
    run_training(
        args,
        fine_tune_block5=False,
        title="Hybrid Quantum VGG16 Baseline Results",
    )


if __name__ == "__main__":
    main()
