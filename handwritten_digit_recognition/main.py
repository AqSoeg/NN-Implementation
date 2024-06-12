from src.train import train_model
from src.evaluate import evaluate_model


if __name__ == '__main__':
    train_model(epochs=10, batch_size=64, learning_rate=0.001)
    evaluate_model()
