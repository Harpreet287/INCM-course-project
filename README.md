# Neural Dynamics & Perceptual Discrimination in RNNs

This repository contains a suite of Jupyter notebooks for training, analyzing, and perturbing Recurrent Neural Networks (RNNs) on a 2-Alternative Forced Choice (2AFC) Perceptual Discrimination task. The projects explores biological constraints (Dale's Law), novel training objectives (Adaptive Contrastive Gating), and causal mechanisms (Lesioning/Clamping).

## Dependencies

The code relies primarily on **PyTorch**, with one notebook demonstrating the **PsychRNN** (TensorFlow-based) library.

* Python 3.7+
* PyTorch (`torch`)
* NumPy (`numpy`)
* Matplotlib (`matplotlib`)
* Scikit-Learn (`sklearn`) - for PCA
* [PsychRNN](https://github.com/murraylab/PsychRNN) (Only for `PsychRNN_..._noise_comparison.ipynb`)

## Notebooks Overview

### 1. Core Models & Biological Constraints
**File:** `Dales_Law (1).ipynb`
* **Description:** Implements an RNN with **Dale's Law** (distinct excitatory and inhibitory populations) and multiplicative gating.
* **Model:** `RNNWithGateDale`
    * **Input:** 2 channels (Evidence Left/Right)
    * **Hidden:** 128 units (80% Excitatory, 20% Inhibitory)
    * **Output:** 2 units (Logits)
* **Experiments:** Compares three training regimes:
    1.  **Baseline:** Standard training.
    2.  **Noisy:** Injects recurrent noise (`rec_noise=0.1`) during training.
    3.  **Curriculum:** Linearly decreases stimulus coherence as accuracy improves.
* **Reproduce:** Run all cells. Outputs loss/psychometric plots to `out_full_experiment/`.

### 2. Adaptive Contrastive Gating (ACG)
**File:** `psychrnn_ACG_full_(2).ipynb`
* **Description:** Introduces a custom loss function to structure the latent space by separating class centroids.
* **Model:** `RNN_ACG` (PyTorch)
* **Loss Function:** $L = L_{CE} + \lambda \cdot \text{ReLU}(margin - ||\mu_0 - \mu_1||)$
* **Key Analyses:**
    * **State Space:** PCA trajectories of hidden activity.
    * **Dynamics:** Fixed point analysis using numerical optimization.
    * **Robustness:** Performance degradation under random neuron lesions.
* **Reproduce:** Run all cells. Generates `pca_trajectories.png` and `lesion_robustness.png`.

**File:** `psychrnn_novel_ACG.ipynb`
* **Description:** A lighter/demo version comparing Baseline, Gating-only, and ACG models. Good for quick verification of the ACG mechanism.

### 3. Causal Perturbations & Interpretability
**File:** `dynamics_causal_perturbations.ipynb`
* **Description:** Focuses on mechanistic interpretability of a trained RNN.
* **Model:** `RNNCausal` (Vanilla RNN, 80 hidden units).
* **Techniques:**
    * **Saliency:** Identifies top-k important neurons via gradient analysis.
    * **Lesioning:** Silences specific neurons to test causal necessity.
    * **Clamping:** Fixes neuron activity to constant values to test causal sufficiency.
* **Reproduce:** Run cells to train a baseline model, identify salient neurons, and generate `psych_lesion.png` (Psychometric curve pre/post lesion).

### 4. Curriculum Learning Demo
**File:** `PsychRNN_CurriculumLearning_works.ipynb`
* **Description:** A self-contained PyTorch implementation of a curriculum learning loop.
* **Algorithm:** Starts with high coherence (easy task). Evaluates validation accuracy every 200 iterations. If accuracy > 80%, coherence is reduced by 0.1.
* **Reproduce:** Run to see console output tracking the "lowered coherence" steps and final validation accuracy.

### 5. PsychRNN Library Comparison
**File:** `PsychRNN_PerceptualDiscrimination_noise_comparison_works.ipynb`
* **Description:** Reference implementation using the `psychrnn` library (TensorFlow backend).
* **Comparison:** Trains two models (Baseline vs. Noisy) to visualize how recurrent noise smooths PCA trajectories.
* **Note:** Requires `pip install git+https://github.com/murraylab/PsychRNN`.

## Data Generation
All PyTorch notebooks use a synthetic data generator `generate_batch` that produces:
* **Inputs:** Shape `(Batch, Time, 2)`. Gaussian noise + Coherence signal added to one channel.
* **Targets:** Binary label `0` or `1`.

## Quick Start
To reproduce the main ACG results:
1.  Open `psychrnn_ACG_full_(2).ipynb`.
2.  Ensure `BEST` hyperparameters are set (default in notebook).
3.  Run All Cells.
4.  Check the `acg_out/` directory for `all_results.json` and `pca_trajectories.png`.
