"""
File: LDA_model.py
Author: Ella Bitterman
Desc: This code is almost identical to the previous program buildLDAToPredict;
      I only changed the directories and file paths.
"""

import os
import json
import pprint

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import label_binarize
import seaborn as sns  # If seaborn is not available, you can use plt.imshow instead
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc


def create_emb_dct(input_path, file_ending):
    """ Given an input path to video embedding jsons, return a dictionary where
    key = video name and value = lists of the embeddings for given video.
    E.g. {class1: {class1_lift_1: [1.2, 3.4, 5.6], ...}, class2: {class2_kick_4: [embs]}}
    """
    emb_dct = {}
    class_names = list(set([file[:6] for file in os.listdir(input_path)]))

    for class_name in class_names:
        temp_dct = {}
        for file in os.listdir(input_path):
            if file.endswith(file_ending):
                if class_name == file[:6]:
                    vid_name = file.replace(file_ending, "")
                    with open(os.path.join(input_path, file), 'r', encoding='utf-8') as f:
                        emb = json.load(f)["embedding"][0]
                        temp_dct[vid_name] = emb
                    emb_dct[class_name] = temp_dct
    return emb_dct, class_names

def train_test(emb_dct):
    """ Given a video embedding dictionary, split the videos into train/test splits """
    train = {}
    test = {}
    for class_name, videos in emb_dct.items():
        video_names = list(videos.keys())
        embs = np.array(list(videos.values()))
        x_train, x_test, names_train, names_test = train_test_split(embs, video_names,
                                                                    test_size=0.2, random_state=42)
        train[class_name] = (x_train, names_train)
        test[class_name] = (x_test, names_test)
    return train, test

def pca_vis(train, output_path, method):
    """ Given a train dictionary for a certain set of video embeddings,
    create Gaussian distributions (PCA visualizations) for each class """

    for class_name, (x_train, _) in train.items():
        pca = PCA(n_components=2)
        x_train_2d = pca.fit_transform(x_train)
        plt.figure(figsize=(6,5))
        plt.scatter(x_train_2d[:,0], x_train_2d[:,1], c='r', s=40, label='train embeddings')
        plt.title(f'{class_name} {method} Embedding PCA 2D')
        plt.xlabel('PCA1')
        plt.ylabel('PCA2')
        plt.legend()
        plt.tight_layout()

        fig_path = os.path.join(output_path, f'{class_name}_{method}_pca2d.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()

    print(f"{method} PCA visualizations saved to {output_path}")

def run_lda(class_names, train, test, output_path, method):
    """ LDA classification and evaluation """
    x_train_all = [] # Prepare train sets
    y_train_all = [] # Prepare train sets

    for class_name in class_names:
        x_train, _ = train.get(class_name, ([], []))
        if len(x_train) == 0:
            continue
        x_train_all.append(x_train)
        y_train_all += [class_name] * len(x_train)
    x_train_all = np.vstack(x_train_all)
    y_train_all = np.array(y_train_all)

    x_test_all = [] # Prepare test sets
    y_test_all = [] # Prepare test sets
    video_names_all = []

    for class_name in class_names:
        x_test, names_test = test.get(class_name, ([], []))
        if len(x_test) == 0:
            continue
        x_test_all.append(x_test)
        y_test_all += [class_name] * len(x_test)
        video_names_all += names_test
    x_test_all = np.vstack(x_test_all)
    y_test_all = np.array(y_test_all)

    # Train LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(x_train_all, y_train_all)

    # Predict
    y_pred = lda.predict(x_test_all)
    y_proba = lda.predict_proba(x_test_all)  # shape: (n_samples, n_classes)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test_all)
    summary_lines = [f"LDA classification accuracy: {accuracy:.4f}"]

    # Save prediction results
    results = []
    for name, true_c, pred_c, proba in zip(video_names_all, y_test_all, y_pred, y_proba):
        prob_dct = {c: float(p) for c, p in zip(lda.classes_, proba)}
        results.append({
            "video": name,
            "true_class": true_c,
            "pred_class": pred_c,
            "probs": prob_dct
        })
    probs_path = os.path.join(output_path, f"lda_{method}_test_probs.json")
    with open(probs_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"{method} LDA model successfully ran")

    confusion_matrix(lda, y_test_all, y_proba, output_path, method)
    mean_tpr, mean_fpr = TPR_FPR(lda, y_test_all, y_pred, output_path, method)
    ROC_curve(lda, y_test_all, y_proba, output_path, method)

    summary_lines.append(f"LDA results: mean TPR={mean_tpr:.4f}, mean FPR={mean_fpr:.4f}")
    return summary_lines

def confusion_matrix(lda, y_test_all, y_proba, output_path, method):
    # Calculate confusion probability matrix
    class_labels = list(lda.classes_)
    n_classes = len(class_labels)
    confusion_prob_matrix = np.zeros((n_classes, n_classes))

    # Iterate over each true class
    for i, true_c in enumerate(class_labels):
        # Find all sample indices of this true class
        idx = np.where(y_test_all == true_c)[0]
        if len(idx) == 0:
            continue
        # Get the predicted probabilities for these samples
        probs = y_proba[idx]  # shape: (num_samples, n_classes)
        # For each predicted class, calculate the mean probability
        confusion_prob_matrix[i, :] = probs.mean(axis=0)

    # Draw heatmap
    plt.figure(figsize=(8, 6))
    # Swap axes: transpose confusion probability matrix and swap labels
    sns.heatmap(confusion_prob_matrix.T, annot=True, fmt=".2f", cmap="Reds",
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("True Class")
    plt.ylabel("Predicted Class")
    plt.title(f"Average Probability of Each Predicted Class Being from Each True Class (LDA with {method})")
    plt.tight_layout()

    heatmap_path = os.path.join(output_path, f"lda_{method}_test_confusion_prob_matrix.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"{method} confusion matrix heatmaps saved to {output_path}")

def TPR_FPR(lda, y_test_all, y_pred, output_path, method):
# Calculate mean TPR and FPR for all classes and plot
    tpr_lst = []
    fpr_lst = []
    for i, c in enumerate(lda.classes_):
        y_true = (y_test_all == c).astype(int)
        y_pred_bin = (y_pred == c).astype(int)
        tp = np.sum((y_true == 1) & (y_pred_bin == 1))
        fp = np.sum((y_true == 0) & (y_pred_bin == 1))
        fn = np.sum((y_true == 1) & (y_pred_bin == 0))
        tn = np.sum((y_true == 0) & (y_pred_bin == 0))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr_lst.append(tpr)
        fpr_lst.append(fpr)

    # Calculate mean TPR and FPR
    mean_tpr = np.mean(tpr_lst)
    mean_fpr = np.mean(fpr_lst)

    # Plot bar chart of mean TPR and FPR
    plt.figure(figsize=(6, 6))
    plt.bar(["Mean TPR", "Mean FPR"], [mean_tpr, mean_fpr], color=["skyblue", "salmon"])
    plt.ylabel("Rate")
    plt.title(f"Mean TPR and FPR of All Classes ({method})")
    plt.ylim(0, 0.6)
    plt.tight_layout()

    mean_tpr_fpr_fig_path = os.path.join(output_path, f"lda_{method}_mean_tpr_fpr_bar.png")
    plt.savefig(mean_tpr_fpr_fig_path, dpi=300)
    plt.close()
    print(f"{method} mean TPR and FPR bar graphs saved to {output_path}")

    return mean_tpr, mean_fpr

def ROC_curve(lda, y_test_all, y_proba, output_path, method):
    # Multiclass one-vs-rest -> binarize class labels
    classes = list(lda.classes_)
    y_test_bin = label_binarize(y_test_all, classes=classes)

    # Calculate ROC curve and AUC for each class
    fpr_dct = dict()
    tpr_dct = dict()
    roc_auc_dct = dict()
    for i, c in enumerate(classes):
        fpr_dct[c], tpr_dct[c], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc_dct[c] = auc(fpr_dct[c], tpr_dct[c])

    # Calculate micro-average ROC curve and AUC
    fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    # Only plot the micro-average ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_micro, tpr_micro, color='navy', lw=2, linestyle='-',
             label=f'all classes summary (mean AUC={roc_auc_micro:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve of All Classes ({method})")
    plt.legend(loc="lower right")
    plt.tight_layout()

    roc_fig_path = os.path.join(output_path, f"lda_{method}_all_classes_roc_curve.png")
    plt.savefig(roc_fig_path, dpi=300)
    plt.close()
    print(f"{method} ROC curves saved to {output_path}")

def embed_run(method, lda_path, emb_path):

    # Create directory for all results to go in
    lda_specific_dir = os.path.join(lda_path, f"LDA_{method}")
    os.makedirs(lda_specific_dir, exist_ok=True)

    # Create embedding dictionary
    specific_emb_path = os.path.join(emb_path, f"{method}_embs")
    specific_dct, class_names = create_emb_dct(specific_emb_path, f"_{method}_emb.json")

    # Get train/test splits
    specific_train, specific_test = train_test(specific_dct)

    # Visualize Gaussian distributions (PCA plots)
    pca_vis(specific_train, lda_specific_dir, method)

    # Run LDA model and visualize confusion matrix, mean TPR/FPR bar chart, and ROC curve
    summary_lines = run_lda(class_names, specific_train, specific_test, lda_specific_dir, method)

    # Save summary lines file
    specific_summary_path = os.path.join(lda_specific_dir, f"lda_{method}_predict_stats.txt")
    with open(specific_summary_path, 'w', encoding='utf-8') as f:
        for line in summary_lines:
            f.write(line + '\n')

    print(f"{method} full LDA analysis done.")


def main():
    # Set up paths to main directories
    current_dir = os.getcwd()
    verb_path = os.path.join(current_dir, "verb_classes")
    emb_path = os.path.join(verb_path, "embeddings")
    lda_path = os.path.join(verb_path, "LDA")
    os.makedirs(lda_path, exist_ok=True)

    # Test different embedding methods: 1. VideoMAE
    method = "VideoMAE"
    embed_run(method, lda_path, emb_path)


if __name__ == "__main__":
    main()
