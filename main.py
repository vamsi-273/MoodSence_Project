# from src.training.train import train_model
# from src.evaluation.evaluate import evaluate_model
# from src.training.train_edl import train_edl_model
# from src.evaluation.evaluate_edl import evaluate_edl_uncertainty
from src.evaluation.evaluate_edl_full import evaluate_edl


if __name__ == "__main__":
    # train_model(epochs=15, batch_size=16, lr=0.0003)
    # evaluate_model("best_edl_model.pth")
    # train_edl_model()
    # evaluate_edl_uncertainty()
    evaluate_edl()