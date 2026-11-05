# 🧬 Support-Conditioned Prototype Transport for Few-Shot-based Medical Image Classification

## 🎯 Abstract

Few-shot learning (FSL) recognises unseen medical classes from a few labelled images, but most metric-based methods reduce each image and each class to a single vector, discarding local and heterogeneous lesion evidence. We propose SCPT (Support-Conditioned Prototype Transport), which matches sets of local query tokens to sets of class prototype atoms. SCPT encodes images as multi-scale local tokens, builds a diversity-adaptive number of atoms per class refined by optimal transport, and classifies queries by Sinkhorn distance. The framework employs a lightweight shared encoder as a backbone and is optimised end-to-end using an episodic cross-entropy objective. Extensive experiments on three benchmark datasets (HAM10000, Kvasir, and BreakHis) demonstrate that our proposed SCPT consistently outperforms state-of-the-art methods, achieving superior 2-way balanced accuracies across both 5-shot (84.88\%, 68.37\%, and 58.80\%) and 10-shot (86.43\%, 70.68\%, and 62.20\%) settings. Overall, this SCPT framework proves highly effective in extremely low-resource settings, marking a significant step toward practical, label-efficient medical image classification.

<img width="1561" height="422" alt="image" src="https://github.com/user-attachments/assets/a54b711f-7de3-4a7e-8b42-a780899bea64" />
**Figure 1:** Overview of the proposed SCPT framework. MER extracts multi-scale features as normalized tokens. SAPC constructs class-specific prototypes using support diversity, farthest-point sampling, and balanced matching. QPM performs Sinkhorn-based query-to-prototype matching for class prediction.

## Table 1. Comparison of methods in terms of 2-way balanced accuracy (\%), reported as mean $\pm$ standard deviation over five random seeds and 2,000 test episodes.
| Method | HAM10000 (5-shot) | HAM10000 (10-shot) | Kvasir (5-shot) | Kvasir (10-shot) | BreakHis (5-shot) | BreakHis (10-shot) |
|:---|---:|---:|---:|---:|---:|---:|
| SimpleShot| 63.85 ± 0.22 | 65.37 ± 0.21 | 58.04 ± 0.24 | 64.21 ± 0.17 | 53.67 ± 0.19 | 56.28 ± 0.19 |
| SetFeat (CVPR) | 86.77 ± 1.61 | 87.91 ± 1.68 | 60.91 ± 1.29 | 66.18 ± 0.24 | 60.70 ± 1.27 | 65.38 ± 1.29 |
| PADDLE (NeurIPS) | 71.52 ± 1.74 | 81.09 ± 1.47 | 68.39 ± 0.47 | 69.94 ± 0.84 | 55.32 ± 1.12 | 56.46 ± 1.38 |
| DeepBDC (CVPR) | 69.79 ± 0.52 | 75.20 ± 0.49 | 54.29 ± 0.48 | 54.70 ± 0.49 | 54.09 ± 0.43 | 54.60 ± 0.44 |
| MTUNet (AI) | 65.17 ± 0.45 | 77.08 ± 0.48 | 51.86 ± 0.53 | 53.46 ± 0.56 | 54.95 ± 0.59 | 58.35 ± 0.42 |
| CPEA (ICCV) | 84.75 ± 0.45 | 85.69 ± 0.41 | 68.84 ± 0.56 | 71.24 ± 0.45 | 58.22 ± 0.61 | 61.71 ± 0.63 |
| MTUNet++ (MTA) | 68.07 ± 0.61 | 79.85 ± 0.51 | 54.03 ± 0.61 | 55.01 ± 0.62 | 53.02 ± 0.64 | 57.71 ± 0.39 |
| MTUNet++ w/ GAN (MTA) | 84.35 ± 0.59 | 85.19 ± 0.53 | 63.49 ± 0.49 | 69.28 ± 0.53 | - | - |
| UNEM (CVPR) | 66.07 ± 1.24 | 79.20 ± 1.02 | 67.04 ± 0.54 | 68.55 ± 0.45 | 55.19 ± 0.84 | 56.49 ± 1.04 |
| **SCPT (Ours)** | 84.88 ± 1.49 | 86.43 ± 2.71 | 68.37 ± 2.18 | 70.68 ± 2.04 | 58.80 ± 1.28 | 62.20 ± 2.57 |

## Table 2. Intra-module ablation results (2-way balanced accuracy on 2000 test episodes, mean $\pm$ std over five seeds).
| ID | Variant | HAM10000 (5-shot) | HAM10000 (10-shot) | Kvasir (5-shot) | Kvasir (10-shot) |
|:---|:---|---:|---:|---:|---:|
| **Module 1: Feature Selection** | | | | | |
| M1-A | $S_2$ only | 71.42 ± 1.71 | 71.68 ± 1.84 | 59.67 ± 1.01 | 67.29 ± 1.74 |
| M1-B | $S_3$ only | 82.71 ± 1.39 | 84.04 ± 1.94 | 59.64 ± 1.64 | 63.01 ± 1.88 |
| M1-C | $S_B$ only | 82.23 ± 1.62 | 81.63 ± 1.54 | 67.64 ± 1.57 | **73.23 ± 1.73** |
| **M1-D** | **$S_2+S_3+S_B$ (Ours)** | **84.88 ± 1.49** | **86.43 ± 2.71** | **68.37 ± 2.18** | 70.68 ± 2.04 |
| | | | | | |
| **Module 2: Adaptive Prototype Selection** | | | | | |
| M2-A | Fixed capacity ($M=K$) | 65.94 ± 1.30 | 63.73 ± 3.20 | 53.34 ± 1.86 | 54.93 ± 3.36 |
| M2-B | Adaptive FPS (without OT) | 62.13 ± 2.22 | 60.02 ± 4.45 | 54.32 ± 0.62 | 55.36 ± 1.83 |
| **M2-C** | **Adaptive FPS + OT refinement (Ours)** | **84.88 ± 1.49** | **86.43 ± 2.71** | **68.37 ± 2.18** | **70.68 ± 2.04** |
| | | | | | |
| **Module 3: Assignment Strategy** | | | | | |
| M3-A | Euclidean assignment | 81.04 ± 2.49 | 77.62 ± 4.58 | 62.02 ± 1.84 | 63.34 ± 2.07 |
| M3-B | Soft cosine assignment | 83.63 ± 1.23 | 83.74 ± 4.07 | 62.88 ± 1.63 | 64.63 ± 1.92 |
| M3-C | Alternative M3 configuration | 82.73 ± 4.66 | 79.28 ± 5.30 | 64.36 ± 2.13 | 66.29 ± 2.59 |
| **M3-D** | **Balanced OT (Ours)** | **84.88 ± 1.49** | **86.43 ± 2.71** | **68.37 ± 2.18** | 70.68 ± 2.04 |

## Table 3. Cross-dataset transfer (balanced accuracy \%), no target fine-tuning. Mean $\pm$ std over five seeds.
| Source → Target | HAM10000 (5-shot) | HAM10000 (10-shot) | Kvasir (5-shot) | Kvasir (10-shot) | BreakHis (5-shot) | BreakHis (10-shot) |
|:---|---:|---:|---:|---:|---:|---:|
| **HAM10000 →** | 84.88 ± 1.49** | **86.43 ± 2.71** | 76.99 ± 1.98 | 78.91 ± 1.45 | 67.88 ± 1.98 | 71.43 ± 0.93 |
| **Kvasir →** | 59.15 ± 1.46 | 61.93 ± 1.41 | 70.68 ± 2.04 | 73.94 ± 2.74 | 57.39 ± 1.79 | 63.60 ± 2.51 |
| **BreakHis →** | 61.49 ± 1.08 | 65.28 ± 2.76 | 76.44 ± 2.28 | 76.20 ± 2.23 | 58.80 ± 1.28 | 62.20 ± 2.57 |


<img width="564" height="202" alt="image" src="https://github.com/user-attachments/assets/e2fede5b-667c-4827-905c-813075111636" />

**Figure 2:** Qualitative visualization of SCPT-based optimal transport matching for a dermatological query image. The query is compared against the true-class and predicted-class support prototypes, with the corresponding transport plans \(T^*\) illustrating query-token-to-atom assignments. The identical raw OT distances (\(D=0.231\)) for both support configurations indicate comparable transport costs, highlighting the ambiguity in prototype-level matching.

<img width="552" height="184" alt="image" src="https://github.com/user-attachments/assets/ecd13664-12ea-49ed-b968-79f11d269aa2" />

**Figure 3:** Qualitative visualization of optimal transport-based prototype matching for a dermoscopic query image. The query is compared against the true-class (nevus, NV) and predicted-class (vascular lesion, VASC) support prototypes. The corresponding transport plans \(T^*\) visualize query-token-to-atom assignments. The lower raw OT distance for the predicted class (\(D=0.250\)) compared with the true class (\(D=0.488\)) indicates stronger prototype affinity toward VASC, explaining the model's misclassification.

<img width="546" height="182" alt="image" src="https://github.com/user-attachments/assets/fe20820a-ba7d-416d-95e1-909c5cd49188" />

**Figure 4:** Qualitative visualization of optimal transport-based prototype matching for a gastrointestinal endoscopic query image. The query is compared against the true-class (polyps) and predicted-class (polyps) support prototypes. The corresponding transport plans \(T^*\) illustrate query-token-to-atom assignments. Identical raw OT distances (\(D=0.225\)) for both configurations indicate consistent prototype matching, supporting the correct classification of the query as a polyp.

<img width="544" height="177" alt="image" src="https://github.com/user-attachments/assets/b0a667bb-1bd0-42e9-944c-78ac0d0f190d" />

**Figure 5:** Qualitative visualization of optimal transport-based prototype assignment. Given a query image and two support examples (true class: ulcerative colitis; predicted class: polyps), the corresponding transport plans \(T^*\) illustrate the query-token-to-atom correspondence. The reported raw OT distances (\(D=0.313\) for the true class and \(D=0.223\) for the predicted class) indicate stronger transport affinity toward the predicted polyp prototype, consistent with the model's classification decision.







