import argparse

from model import train, evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--final_evaluation', action="store_true", help='Perform final evaluation: compute final FID for a generator')
    parser.add_argument('-d', '--double', action="store_true", help='Do the model have a Double Generator Architecture?')
    parser.add_argument('-s', '--seed', default=2333, type=int, help='Do the model have a Double Generator Architecture?')
    parser.add_argument('-c', '--constant', default=64, type=int, help='Model Architecture constant (default 64)')
    parser.add_argument('-x', '--experience_dir', type=str, help='Experience directory to save the model and the vizu')
    parser.add_argument('-t', '--train_path', type=str, help='Path to training dataset')
    parser.add_argument('-v', '--valid_path', type=str, help='Path to validation dataset')
    parser.add_argument('-b', '--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=250000, help='Number of epochs to train the model for')
    parser.add_argument('-m', '--method', type=str, default='random', help='Method to generate hints: either "random" or "strokes" mode available')

    args = parser.parse_args()

    if not args.final_evaluation:
        train(
            double         = args.double,
            seed           = args.seed,
            constant       = args.constant,
            experience_dir = args.experience_dir,
            train_path     = args.train_path,
            valid_path     = args.valid_path,
            batch_size     = args.batch_size,
            epochs         = args.epochs,
            drift          = 1e-3,
            adwd           = 1e-4,
            method         = args.method
        )
    else:
        evaluate(
            seed           = args.seed,
            constant       = args.constant,
            experience_dir = args.experience_dir,
            valid_path     = args.valid_path,
            batch_size     = args.batch_size
        )
