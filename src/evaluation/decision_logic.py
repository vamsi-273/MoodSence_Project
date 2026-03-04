import torch

def decision_with_uncertainty(evidence, threshold=0.87, num_classes=7):

    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)

    probs = alpha / S
    uncertainty = num_classes / S

    confidence, predicted = torch.max(probs, 1)

    decision = []

    for i in range(len(predicted)):
        if uncertainty[i] > threshold:
            decision.append("UNCERTAIN - Activate Fallback")
        else:
            decision.append(f"Predicted Class: {predicted[i].item()}")

    return decision, uncertainty