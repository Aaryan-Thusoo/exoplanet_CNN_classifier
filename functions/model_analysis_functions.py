import numpy as np
import matplotlib.pyplot as plt
import random

import torch
import torch.nn.functional as F

def get_batch_information(model, test_loader):
    """
    Collects and returns test batch information

    :param model: trained model
    :param test_loader: DataLoader for test set

    :return: predictions, true values, confidence
    """

    model.eval()

    all_preds = []
    all_true = []
    all_conf = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:

            outputs = model(X_batch)
            probs = F.softmax(outputs, dim=1)

            confs, preds = torch.max(probs, dim=1)

            all_preds.append(preds.cpu())
            all_true.append(y_batch.cpu())
            all_conf.append(confs.cpu())

    all_preds = torch.cat(all_preds)
    all_true = torch.cat(all_true)
    all_conf = torch.cat(all_conf)

    return all_preds, all_true, all_conf

def get_all_incorrect(all_preds, all_true, lc_types):
    incorrect = []

    for i in range(0, len(all_preds)):
        if all_preds[i] != all_true[i]:
            incorrect.append((i, int(all_preds[i]), int(all_true[i])))

    incorrect_ordered = []

    for lc_type in lc_types:
        type_incorrect = []
        for i in range(0, len(incorrect)):
            if incorrect[i][2] == lc_type:
                type_incorrect.append(incorrect[i])

        incorrect_ordered.append(type_incorrect)

    return incorrect_ordered

from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(all_true, all_preds):
    names = ["normal", "transit", "eclipsing"]

    conf_matrix = confusion_matrix(all_true, all_preds, normalize='true')
    n_classes = len(names)

    plt.figure(figsize=(6, 5))
    plt.imshow(conf_matrix, cmap='Blues', vmin=0, vmax=1)

    for i in range(n_classes):
        for j in range(n_classes):
            num = conf_matrix[i, j] * 100
            text = f"{num:.2f}%"
            colour = 'white' if conf_matrix[i, j] > 0.6 else 'black'
            plt.text(j, i, text, ha="center", va="center", color=colour)

    accuracy = np.mean(np.array(all_true) == np.array(all_preds)) * 100

    plt.title(f"Accuracy: {accuracy}")
    plt.xticks(range(n_classes), names, rotation=45, ha='right')
    plt.yticks(range(n_classes), names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.colorbar(label="Fraction")
    plt.tight_layout()
    plt.show()


def plot_confidence_confusion_matrix(all_true, all_preds, all_conf):
    names = ["normal", "transit", "eclipsing"]
    n_classes = len(names)

    conf_sum = np.zeros((n_classes, n_classes), dtype=float)
    counts = np.zeros((n_classes, n_classes), dtype=int)

    for t, p, c in zip(all_true, all_preds, all_conf):
        conf_sum[t, p] += c
        counts[t, p] += 1

    avg_conf = np.divide(conf_sum, counts, out=np.zeros_like(conf_sum), where=counts > 0)

    plt.figure(figsize=(6, 5))
    plt.imshow(avg_conf, cmap='Greens', vmin=0, vmax=1)

    for i in range(n_classes):
        for j in range(n_classes):
            if counts[i, j] == 0:
                text = "--"
                colour = "black"
            else:
                num = avg_conf[i, j] * 100
                text = f"{num:.2f}%"
                colour = 'white' if avg_conf[i, j] > 0.6 else 'black'

            plt.text(j, i, text, ha="center", va="center", color=colour)

    accuracy = np.mean(np.array(all_true) == np.array(all_preds)) * 100

    plt.title(f"Accuracy: {accuracy:.2f}%")
    plt.xticks(range(n_classes), names, rotation=45, ha='right')
    plt.yticks(range(n_classes), names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.colorbar(label="Average confidence")
    plt.tight_layout()
    plt.show()


def plot_misclassified(X_test, incorrect_ordered, type, seed=None):
    """
    Plots some misclassified images
    :param incorrect_ordered: list(n, x_n) with each inner array corresponding to each lc_type with x_n number of incorrect points
    :param type: tuple, len==2, tells which type to check was misclassified and the class if was classified as
    :param seed int, preset index for which to check

    :return seeds used
    len(seeds) == len(num_plot)
    """
    class_names = ["Normal Star", "Transit", "Eclipsing Binary"]
    type_correct, type_incorrect = type[0], type[1]

    good_seed = False
    possible_seed_indexes = np.arange(len(incorrect_ordered[type_correct]))
    if seed is None:
        while not good_seed:
            seed = random.choice(possible_seed_indexes)

            if incorrect_ordered[type_correct][seed][1] != type_incorrect:
                possible_seed_indexes = possible_seed_indexes[possible_seed_indexes != seed]
            else:
                good_seed = True

    plt.figure(figsize=(12, 5))
    plt.plot(X_test[seed][0])
    plt.title(f"{class_names[type_correct]} Misclassified as {class_names[type_incorrect]} (Seed: {seed})")
    plt.show()

def plot_confidence_hist(all_conf, bins=20):
    plt.hist(all_conf.numpy(), bins=bins)
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title("Prediction Confidence Histogram")
    plt.show()

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

def plot_tsne_classifications(
    model,
    dataloader,
    feature_layer_name="fc2",
    class_names=None,
    color_by="pred",          # "pred", "true", or "correctness"
    perplexity=30,
    random_state=42,
    figsize=(8, 6)
):
    """
    Plot a t-SNE projection of model features.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model.
    dataloader : DataLoader
        DataLoader containing inputs and labels.
    feature_layer_name : str
        Name of the layer to hook for feature extraction.
    class_names : dict or list or None
        Mapping of class indices to class names.
        Example: {0: 'normal', 1: 'transit', 2: 'eclipsing'}
    color_by : str
        "pred"        -> color by predicted class
        "true"        -> color by true class
        "correctness" -> color by whether prediction was correct
    perplexity : int
        t-SNE perplexity.
    random_state : int
        Random seed for t-SNE.
    figsize : tuple
        Figure size.
    """

    model.eval()

    features = []
    true_labels = []
    pred_labels = []

    # Find the layer by name
    layer_dict = dict(model.named_modules())
    if feature_layer_name not in layer_dict:
        raise ValueError(f"Layer '{feature_layer_name}' not found in model. "
                         f"Available layers: {list(layer_dict.keys())}")

    def hook(module, input, output):
        features.append(output.detach().cpu())

    handle = layer_dict[feature_layer_name].register_forward_hook(hook)

    with torch.no_grad():
        for X, y in dataloader:
            outputs = model(X)
            preds = outputs.argmax(dim=1)

            true_labels.extend(y.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    handle.remove()

    # Combine hooked features
    all_features = torch.cat(features, dim=0)

    # Flatten in case features are not already 2D
    all_features = all_features.view(all_features.size(0), -1).numpy()

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    tsne_result = tsne.fit_transform(all_features)

    # Resolve class names
    if class_names is None:
        unique_classes = sorted(set(true_labels) | set(pred_labels))
        class_names = {i: f"Class {i}" for i in unique_classes}
    elif isinstance(class_names, list):
        class_names = {i: name for i, name in enumerate(class_names)}

    plt.figure(figsize=figsize)

    if color_by == "true":
        labels_for_plot = [class_names[i] for i in true_labels]
        sns.scatterplot(
            x=tsne_result[:, 0],
            y=tsne_result[:, 1],
            hue=labels_for_plot,
            alpha=0.8,
            s=60,
            edgecolor="black"
        )
        plt.title("t-SNE of Features Colored by True Class")

    elif color_by == "pred":
        labels_for_plot = [class_names[i] for i in pred_labels]
        sns.scatterplot(
            x=tsne_result[:, 0],
            y=tsne_result[:, 1],
            hue=labels_for_plot,
            alpha=0.8,
            s=60,
            edgecolor="black"
        )
        plt.title("t-SNE of Features Colored by Predicted Class")

    elif color_by == "correctness":
        correctness = np.where(pred_labels == true_labels, "Correct", "Misclassified")
        sns.scatterplot(
            x=tsne_result[:, 0],
            y=tsne_result[:, 1],
            hue=correctness,
            alpha=0.8,
            s=60,
            edgecolor="black"
        )
        plt.title("t-SNE of Features Colored by Classification Correctness")

    else:
        raise ValueError("color_by must be 'pred', 'true', or 'correctness'")

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Label")
    plt.tight_layout()
    plt.show()

