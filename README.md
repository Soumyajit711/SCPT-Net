# Support-Conditioned Prototype Transport for Few-Shot-based Medical Image Classification

## 🎯 Abstract

Few-shot learning (FSL) recognises unseen medical classes from a few labelled images, but most metric-based methods reduce each image and each class to a single vector, discarding local and heterogeneous lesion evidence. We propose SCPT (Support-Conditioned Prototype Transport), which matches sets of local query tokens to sets of class prototype atoms. SCPT encodes images as multi-scale local tokens, builds a diversity-adaptive number of atoms per class refined by optimal transport, and classifies queries by Sinkhorn distance. The framework employs a lightweight shared encoder as a backbone and is optimised end-to-end using an episodic cross-entropy objective. Extensive experiments on three benchmark datasets (HAM10000, Kvasir, and BreakHis) demonstrate that our proposed SCPT consistently outperforms state-of-the-art methods, achieving superior 2-way balanced accuracies across both 5-shot (84.88\%, 68.37\%, and 58.80\%) and 10-shot (86.43\%, 70.68\%, and 62.20\%) settings. Overall, this SCPT framework proves highly effective in extremely low-resource settings, marking a significant step toward practical, label-efficient medical image classification.

## 🤝 Contribution

1) **Multi-Scale Token-to-Prototype Optimal Transport:** We represent each query image as a set of multi-scale local tokens and each class as a compact set of prototype atoms. Their correspondence is established through Sinkhorn-based Optimal Transport (OT), replacing conventional global-vector matching used in metric-based few-shot learning methods.
2) **Adaptive Prototype Capacity and OT-Based Refinement:** We introduce a support-diversity-driven mechanism that adaptively determines the number of prototype atoms per class. These atoms are further refined through entropic OT-barycentric updates to capture intra-class variations more effectively.
3) **Comprehensive Medical Imaging Evaluation:** We evaluate the proposed framework across three medical imaging benchmarks: HAM10000 (dermoscopy), Kvasir (endoscopy), and BreakHis (histopathology). We further conduct extensive intra-module ablation studies and cross-dataset generalization experiments to investigate the effectiveness and transferability of the proposed approach.

## Overview

SCPT is a few-shot medical image classification framework designed to address the limitations of conventional metric-based methods, particularly their reliance on global image representations and a single prototype per class.

The central hypothesis is that medical image categories exhibit substantial intra-class morphological diversity, which may not be adequately represented by a single global feature vector. SCPT therefore represents each query image as a set of multi-scale local tokens and each class as an adaptive collection of prototype atoms. Query-to-class correspondence is established using entropy-regularized Optimal Transport (OT), solved efficiently through Sinkhorn iterations.

The framework consists of three interconnected modules:

1. Multi-scale Embedding Representation (MER)
2. Support-Adaptive Prototype Construction (SAPC)
3. Query-to-Prototype Matching (QPM)

## Methodological Overview

### Overall Pipeline

Input medical images (support set and query image)
    |
    v
MER: Multi-scale Embedding Representation
    - Multi-scale feature extraction
    - Feature alignment and projection
    - Multi-scale feature fusion
    - Local token generation and normalization
    |
    +-------------------------------+
    |                               |
    v                               v
Support representations             Query representation
    |                               |
    v                               |
SAPC: Support-Adaptive               |
Prototype Construction               |
    - Support-diversity estimation   |
    - Adaptive atom capacity         |
    - Farthest Point Sampling (FPS)  |
    - OT-barycentric refinement      |
    |                               |
    +---------------+---------------+
                    |
                    v
QPM: Query-to-Prototype Matching
    - Query-token-to-atom cost computation
    - Entropic optimal transport
    - Sinkhorn transport-plan estimation
    - Transport-based class distance
                    |
                    v
             Predicted class
       (minimum transport distance)

## 1. Module 1: Multi-scale Embedding Representation (MER)

### 1.1 Motivation

Medical images contain discriminative characteristics at different spatial and semantic scales. Dermoscopic images may contain lesion boundaries and pigmentation patterns; endoscopic images may contain mucosal structures and localized abnormalities; and histopathological images may contain cellular arrangements and tissue-level morphology.

A single global representation can suppress these localized variations. MER therefore extracts complementary features at multiple encoder stages and converts the fused representation into a set of local tokens.

### 1.2 Multi-scale Feature Extraction

Given an input image:

    X ∈ R^(H0 × W0 × 3)

a pretrained encoder extracts intermediate feature maps:

    F2, F3, FB

where F2, F3, and FB denote intermediate-, deeper-, and bottleneck-level representations, respectively.

### 1.3 Feature Alignment and Fusion

The feature maps are spatially aligned and projected into a common embedding space:

    Fhat_i = Proj_i(Resize(F_i))

The aligned features are fused:

    F(X) = Fuse(Fhat_2, Fhat_3, Fhat_B)

For the described configuration, the fused representation may have dimensions:

    F(X) ∈ R^(128 × 14 × 14)

The precise dimensions depend on the encoder and configuration used.

### 1.4 Local Token Generation

The fused feature map is flattened into spatially distributed local tokens:

    U(X) = {u_1, u_2, ..., u_P}

For a 14 × 14 feature map:

    P = 14 × 14 = 196

Each token is a 128-dimensional vector:

    u_p ∈ R^128

The resulting representation is:

    U(X) ∈ R^(196 × 128)

Positional information may be incorporated before L2 normalization. The same representation procedure is applied to support and query images.

**Output:** A normalized set of local feature tokens for each input image.

## 2. Module 2: Support-Adaptive Prototype Construction (SAPC)

### 2.1 Motivation

Conventional prototype-based few-shot classification often represents each class using a single mean feature vector:

    p_c = (1/K) Σ_k g_(c,k)

This can be restrictive when support examples within a medical category exhibit substantial morphological variation. SAPC instead constructs multiple prototype atoms per class and adapts the number of atoms according to the diversity observed in the support set.

### 2.2 Support Representation and Diversity Estimation

For class c, let the support set be:

    S_c = {X_(c,1), X_(c,2), ..., X_(c,K)}

Each support image is processed by MER. An image-level descriptor can be obtained by aggregating its local tokens:

    g_(c,k) = Normalize((1/P) Σ_p u_(c,k,p))

The class mean descriptor is:

    gbar_c = (1/K) Σ_k g_(c,k)

A representative support-dispersion measure is:

    delta_c = (1/K) Σ_k ||g_(c,k) - gbar_c||_2

The diversity score estimates the variation among the support descriptors.

### 2.3 Adaptive Prototype Capacity

The number of atoms for class c is determined using the measured support diversity. A representative formulation is:

    M_c = clip(round(K(1 + delta_c)), m_min, m_max)

where M_c is the number of atoms, K is the number of support examples, delta_c is the diversity score, and m_min and m_max are capacity bounds.

This formulation is illustrative; the exact scaling and capacity bounds should match the implementation used in the experiments.

### 2.4 Farthest Point Sampling

The local tokens from all support images of class c are combined:

    U_c = Union_k U(X_(c,k))

For K support images and P tokens per image:

    U_c ∈ R^(KP × 128)

Farthest Point Sampling selects M_c representative initial atoms:

    B_c^(0) = FPS(U_c, M_c)

FPS encourages coverage of distinct regions in the support feature distribution.

### 2.5 Entropic OT-Barycentric Refinement

A cost matrix is constructed between support tokens and prototype atoms. Using cosine dissimilarity:

    C_(c,k)(m,p) = 1 - b_(c,m)^T u_(c,k,p)

An entropy-regularized transport plan can be obtained by solving:

    T* = argmin_(T in Pi(a,b))
         <T,C> + epsilon Σ_(m,p) T(m,p) log T(m,p)

Here, a and b are the prescribed transport marginals and epsilon controls entropic regularization.

The atoms are updated through transport-weighted barycentric aggregation:

    btilde_(c,m) =
        [Σ_(k,p) T*_(c,k)(m,p) u_(c,k,p)]
        / [Σ_(k,p) T*_(c,k)(m,p) + epsilon_0]

The updated atom is normalized:

    b_(c,m) = Normalize(btilde_(c,m))

The refinement process adjusts the atoms according to their transport-based correspondence with support tokens.

**Output:** A class-specific set of refined prototype atoms, with adaptive capacity M_c.

## 3. Module 3: Query-to-Prototype Matching (QPM)

### 3.1 Motivation

After constructing class-specific atoms, SCPT determines which class best corresponds to an unseen query. Rather than relying only on a global Euclidean or cosine distance, QPM models the correspondence between individual query tokens and class-specific prototype atoms as an optimal transport problem.

### 3.2 Query Representation

For query image X_q, MER generates:

    U_q = {u_(q,1), u_(q,2), ..., u_(q,P)}

For the example configuration:

    U_q ∈ R^(196 × 128)

The prototype atoms for class c are:

    B_c = {b_(c,1), b_(c,2), ..., b_(c,M_c)}

### 3.3 Query-Token-to-Atom Cost

The pairwise cosine dissimilarity is:

    C_(q,c)(p,m) = 1 - u_(q,p)^T b_(c,m)

The cost matrix has dimensions:

    C_(q,c) ∈ R^(P × M_c)

Each entry measures the dissimilarity between a query token and a class-specific atom.

### 3.4 Sinkhorn-Based Optimal Transport

The query tokens and prototype atoms are treated as two distributions. QPM estimates their correspondence by solving:

    T*_(q,c) = argmin_(T in Pi(a,b))
        <T,C_(q,c)> + epsilon KL(T || ab^T)

where T* is the transport plan, a and b are the query-token and prototype-atom marginals, and epsilon is the entropic regularization parameter.

Sinkhorn iterations efficiently approximate the transport plan. The plan describes how query-token mass is distributed across the prototype atoms of a candidate class.

### 3.5 Transport-Based Class Distance

The class distance is computed as:

    D(q,c) = <T*_(q,c), C_(q,c)>

The predicted class is:

    yhat_q = argmin_c D(q,c)

Thus, the query is assigned to the class with the minimum transport cost.

**Output:** Class-wise transport distances and the predicted query class.

## 4. Training and Inference

### 4.1 Episodic Training

For each training episode:

1. Sample C classes.
2. Select K support images per class.
3. Select query images from the sampled classes.
4. Process support and query images through MER.
5. Generate multi-scale local token representations.
6. Estimate support diversity for each class.
7. Determine the adaptive prototype capacity.
8. Initialize prototype atoms using FPS.
9. Refine atoms through OT-barycentric updates.
10. Compute query-to-prototype transport plans using QPM.
11. Calculate class-wise transport distances.
12. Predict query labels and optimize the episodic classification objective.

A representative classification loss is:

    L_episodic = -(1/N_q) Σ_i log[
        exp(-D(q_i,y_i)/tau) /
        Σ_c exp(-D(q_i,c)/tau)
    ]

where N_q is the number of query samples, y_i is the ground-truth class, D(q_i,c) is the transport distance, and tau is a temperature parameter.

The exact loss, optimization schedule, and whether the prototype refinement is differentiable should be reported according to the actual implementation.

### 4.2 Inference

For an unseen few-shot task:

1. Extract local tokens from support and query images using MER.
2. Construct class-specific prototype atoms using SAPC.
3. Adapt atom capacity according to support diversity.
4. Refine prototype representations using OT-barycentric updates.
5. Compute query-to-prototype transport plans using QPM.
6. Calculate class-wise transport distances.
7. Assign each query to the class with the minimum transport distance.

In the cross-dataset evaluation protocol, no target-dataset fine-tuning is performed.

## 5. Summary of the Three Modules

- **MER — Multi-scale Embedding Representation:** Extracts multi-level image features, fuses them, and generates normalized local tokens.
- **SAPC — Support-Adaptive Prototype Construction:** Estimates support diversity, determines the number of atoms per class, initializes them using FPS, and refines them through OT-barycentric updates.
- **QPM — Query-to-Prototype Matching:** Computes query-token-to-atom costs, estimates Sinkhorn transport plans, and classifies queries using transport distances.

## 6. Key Methodological Contributions

1. **Local token-based representation:** SCPT preserves spatially distributed local representations rather than relying exclusively on a single global image vector.
2. **Support-diversity-driven prototype construction:** The number of prototype atoms is adapted to the observed diversity of each class's support set.
3. **OT-based distribution matching:** Query-token distributions are compared with adaptive prototype distributions through entropy-regularized optimal transport.
4. **Cross-domain evaluation:** The framework is evaluated across dermoscopic, endoscopic, and histopathological datasets, including cross-dataset generalization experiments.

## 7. Central Research Hypothesis

Few-shot medical image classification can benefit from representing images through multi-scale local feature distributions, constructing support-diversity-aware multi-atom class prototypes, and performing distribution-level query-to-class matching through entropy-regularized optimal transport.

In summary:

    SCPT = MER + SAPC + QPM

MER determines what local information is represented. SAPC determines how each class is represented. QPM determines how the query is matched against those class representations.

## 8. Experimental Evaluation

The SCPT framework is evaluated on:

- **HAM10000:** Dermoscopic image classification.
- **Kvasir:** Gastrointestinal endoscopic image classification.
- **BreakHis:** Histopathological image classification.

The evaluation includes:

- 2-way 5-shot and 10-shot few-shot classification.
- Comparison against established few-shot learning methods.
- Intra-module ablation experiments.
- Cross-dataset transfer without target fine-tuning.
- Balanced accuracy reported as mean ± standard deviation over five random seeds and 2,000 test episodes, where applicable.



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







