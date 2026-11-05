# 🧬 Support-Conditioned Prototype Transport for Few-Shot-based Medical Image Classification

## 🎯 Abstract

Few-shot learning (FSL) recognises unseen medical classes from a few labelled images, but most metric-based methods reduce each image and each class to a single vector, discarding local and heterogeneous lesion evidence. We propose SCPT (Support-Conditioned Prototype Transport), which matches sets of local query tokens to sets of class prototype atoms. SCPT encodes images as multi-scale local tokens, builds a diversity-adaptive number of atoms per class refined by optimal transport, and classifies queries by Sinkhorn distance. The framework employs a lightweight shared encoder as a backbone and is optimised end-to-end using an episodic cross-entropy objective. Extensive experiments on three benchmark datasets (HAM10000, Kvasir, and BreakHis) demonstrate that our proposed SCPT consistently outperforms state-of-the-art methods, achieving superior 2-way balanced accuracies across both 5-shot (84.88\%, 68.37\%, and 58.80\%) and 10-shot (86.43\%, 70.68\%, and 62.20\%) settings. Overall, this SCPT framework proves highly effective in extremely low-resource settings, marking a significant step toward practical, label-efficient medical image classification.

<img width="1561" height="422" alt="image" src="https://github.com/user-attachments/assets/a54b711f-7de3-4a7e-8b42-a780899bea64" />
Figure 1: Overview of the proposed SCPT framework. MER extracts multi-scale features as normalized tokens. SAPC constructs class-specific prototypes using support diversity, farthest-point sampling, and balanced matching. QPM performs Sinkhorn-based query-to-prototype matching for class prediction.

