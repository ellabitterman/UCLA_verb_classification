"""
File: LDA_model.py
Author: Ella Bitterman
Desc: This code is almost identical to the previous program buildLDAToPredict;
      I only changed the directories and file paths.


"""

import os
import json
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




def main():
    # Set up paths to main directories
    current_dir = os.getcwd()
    verb_path = os.path.join(current_dir, "verb_classes")
    emb_path = os.path.join(verb_path, "embeddings_seg") # change depending on emb type !!!!

    # Create directory for all results to go in
    ldat_dir = os.path.join(verb_path, "LDA_model_seg") # change depending on emb type !!!!
    os.makedirs(ldat_dir, exist_ok=True)


    # Save all embeddings to dictionary -> {vid_name: embedding}
    emb_dct = {} # e.g. {class1: {class1_lift_1: [1.2, 3.4, 5.6], ...}, class2: {class2_kick_4: [embs]}}
    class_names = list(set([file[:6] for file in os.listdir(emb_path)]))

    for class_name in class_names:
        temp_dct = {}
        for file in os.listdir(emb_path):
            if file.endswith("_embedding.json"):
                if class_name == file[:6]:
                    vid_name = file.replace("_embedding.json", "")
                    with open(os.path.join(emb_path, file), 'r', encoding='utf-8') as f:
                        emb = json.load(f)["embedding"][0]
                        temp_dct[vid_name] = emb
                    emb_dct[class_name] = temp_dct

    # 2. Split train and test sets
    train = {}
    test = {}
    for class_name, videos in emb_dct.items():
        video_names = list(videos.keys())
        embs = np.array(list(videos.values()))
        x_train, x_test, names_train, names_test = train_test_split(
            embs, video_names, test_size=0.2, random_state=42
        )
        train[class_name] = (x_train, names_train)
        test[class_name] = (x_test, names_test)
        print(f"Class {class_name} train set size: {len(x_train)}")

    # 3.5 Visualize Gaussian distribution for each class (keep PCA visualization)
    for class_name, (x_train, _) in train.items():
        pca = PCA(n_components=2)
        x_train_2d = pca.fit_transform(x_train)
        plt.figure(figsize=(6,5))
        plt.scatter(x_train_2d[:,0], x_train_2d[:,1], c='r', s=40, label='train embeddings')
        plt.title(f'{class_name} Embedding PCA 2D')
        plt.xlabel('PCA1')
        plt.ylabel('PCA2')
        plt.legend()
        plt.tight_layout()

        fig_path = os.path.join(ldat_dir, f'{class_name}_embedding_pca2d.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f'{class_name} PCA 2D plot saved to: {fig_path}')

    # LDA classification and evaluation
    # Prepare train set
    x_train_all = []
    y_train_all = []

    for class_name in class_names:
        x_train, _ = train.get(class_name, ([], []))
        if len(x_train) == 0:
            continue
        x_train_all.append(x_train)
        y_train_all += [class_name] * len(x_train)
    x_train_all = np.vstack(x_train_all)
    y_train_all = np.array(y_train_all)

    # Prepare test set
    x_test_all = []
    y_test_all = []
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

    # Save prediction results
    results = []
    for name, true_c, pred_c, proba in zip(video_names_all, y_test_all, y_pred, y_proba):
        prob_dct = {c: float(p) for c, p in zip(lda.classes_, proba)}
        results.append({
            'video': name,
            'true_class': true_c,
            'pred_class': pred_c,
            'probs': prob_dct
        })
    probs_path = os.path.join(ldat_dir, 'lda_model_test_probs.json')
    with open(probs_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"LDA test probabilities saved to: {probs_path}")

    # 1. Calculate confusion probability matrix
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

    # 2. Draw heatmap
    plt.figure(figsize=(8, 6))
    # Swap axes: transpose confusion probability matrix and swap labels
    sns.heatmap(confusion_prob_matrix.T, annot=True, fmt=".2f", cmap="Reds",
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("True Class")
    plt.ylabel("Predicted Class")
    plt.title("Average Probability of Each Predicted Class Being from Each True Class (LDA)")
    plt.tight_layout()

    heatmap_path = os.path.join(ldat_dir, "lda_test_confusion_prob_matrix.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Confusion probability matrix heatmap saved to: {heatmap_path}")

    summary_lines = [f"LDA classification accuracy: {accuracy:.4f}"]

    # Calculate mean TPR and FPR for all classes and plot
    # Calculate TPR and FPR for each class
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
    summary_lines.append(f"LDA results: mean TPR={mean_tpr:.4f}, mean FPR={mean_fpr:.4f}")

    # Plot bar chart of mean TPR and FPR
    plt.figure(figsize=(6, 6))
    plt.bar(['Mean TPR', 'Mean FPR'], [mean_tpr, mean_fpr], color=['skyblue', 'salmon'])
    plt.ylabel('Rate')
    plt.title('Mean TPR and FPR of All Classes')
    plt.ylim(0, 0.6)
    plt.tight_layout()

    mean_tpr_fpr_fig_path = os.path.join(ldat_dir, 'lda_model_mean_tpr_fpr_bar.png')
    plt.savefig(mean_tpr_fpr_fig_path, dpi=300)
    plt.close()
    print(f"Bar chart of mean TPR and FPR of all classes has been saved to: {mean_tpr_fpr_fig_path}")

    # ========================
    # Plot ROC curve for all classes
    # ========================

    # Multiclass one-vs-rest
    # Binarize class labels
    classes = list(lda.classes_)
    y_test_bin = label_binarize(y_test_all, classes=classes)
    # yProba shape: (n_samples, n_classes)

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
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve of All Classes')
    plt.legend(loc='lower right')
    plt.tight_layout()

    roc_fig_path = os.path.join(ldat_dir, 'lda_model_all_classes_roc_curve.png')
    plt.savefig(roc_fig_path, dpi=300)
    plt.close()
    print(f"Micro-average ROC curve of all classes has been saved to: {roc_fig_path}")

    # Save summary results
    summary_path = os.path.join(ldat_dir, 'lda_model_predict_stats.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        for line in summary_lines:
            f.write(line + '\n')
    print(f"Summary results saved to: {summary_path}")




if __name__ == "__main__":
    main()
