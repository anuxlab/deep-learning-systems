# deep-learning-systems

Deep Learning Notes — **now with a hands-on practical track.**

The [wiki](https://github.com/anuxlab/deep-learning-systems/wiki) holds the 24-module theory course. This repository turns it into
**25 executable notebooks** (one per module + an environment notebook) that run on **free GitHub-hosted runners (CPU only, no GPU, no
dataset downloads)**, each with its **own manually-triggered workflow** that executes the notebook, shows every output and log, and
**commits the results back into the notebook's folder**.

> *Practice over reading:* every notebook re-derives the key results of its wiki module in code, **verifies them numerically**
> (`check(...)` assertions: gradient checks, match-against-PyTorch tests, theorem bounds, ablations) and trains small models end-to-end.
> About **418 checks** across **94 figures** in total.

## Repository layout

```
notebooks/
  _common/dlsys.py              shared helpers: seeding, env report, show() figure saver, check(), timer(), metrics
  NN-<slug>/NN_<slug>.ipynb     the notebook (after a CI run it contains all outputs)
  NN-<slug>/NN_<slug>.html      standalone rendered copy            ┐
  NN-<slug>/run.log             per-cell timing + all stdout/stderr ├─ created & committed by the workflow
  NN-<slug>/figures/*.png       every figure                        │
  NN-<slug>/metrics.json        machine-readable results + env info ┘
scripts/run_notebook.py         executes one notebook, writes log/HTML even on failure
scripts/job_summary.py          builds the Markdown summary shown on the Actions run page
.github/workflows/
  _run-notebook.yml             reusable workflow (install CPU PyTorch → execute → upload artifact → commit back)
  nb-NN-<slug>.yml              25 thin, manually-triggered workflows (one per notebook)
requirements.txt                CPU dependencies (PyTorch is installed from the CPU wheel index)
```

## Running a notebook in CI (one topic at a time)

1. **Settings → Actions → General → Workflow permissions → “Read and write permissions”** (needed so the job can commit results).
2. Open the **Actions** tab, choose a workflow such as **“NB 07 · Feedforward Neural Networks and Backpropagation”**, click **Run workflow**.
   (or: `gh workflow run nb-07-mlp-backprop.yml`; add `-f commit_results=false` for a dry run).
3. The job installs CPU PyTorch + dependencies, runs the notebook top-to-bottom, and then:
   - shows a **job summary** (status, number of passed checks, timed sections, last 60 log lines),
   - uploads the folder as a **workflow artifact**,
   - **commits** the executed `.ipynb` (with all outputs embedded), `.html`, `run.log`, `figures/` and `metrics.json` back to the branch.
   If a cell fails, the partial notebook and the log are still committed/uploaded and the job is marked failed.

Workflows for different notebooks can run at the same time; the commit step rebases and retries on push races.

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/run_notebook.py notebooks/07-mlp-backprop/07_mlp_backprop.ipynb     # same as CI
# or interactively: jupyter lab   (open any notebook; run from its own folder)
```

## The notebooks

Timings and check counts were measured when this track was validated on a **single-vCPU sandbox** — the 4-vCPU free runner is typically
as fast or faster (add ~1–2 min for the one-off dependency install). Total for all 25: ~19 min of compute.

| # | Notebook | Wiki | What you do | Checks | CPU time | CI |
|---|---|---|---|---|---|---|
| 00 | [Environment, Reproducibility and Course Map](notebooks/00-environment/) | [Roadmap](https://github.com/anuxlab/deep-learning-systems/wiki/Notebook-Roadmap) | Environment record, seeds & reproducibility, CPU throughput, memory budgeting | 6 | ~46s | [run](../../actions/workflows/nb-00-environment.yml) |
| 01 | [Linear Algebra, Vector Spaces and Tensor Mathematics](notebooks/01-linear-algebra/) | [Module 01](https://github.com/anuxlab/deep-learning-systems/wiki/Module-01) | Broadcasting, matmul views, projections/QR, rank & null space, SVD geometry, Eckart–Young, PSD, einsum, strides | 36 | ~9s | [run](../../actions/workflows/nb-01-linear-algebra.yml) |
| 02 | [Calculus, Multivariable Calculus and Automatic Differentiation](notebooks/02-calculus-autodiff/) | [Module 02](https://github.com/anuxlab/deep-learning-systems/wiki/Module-02) | Finite differences, gradients & level sets, Jacobian/Hessian, Taylor, **autodiff engine from scratch**, VJP/JVP | 13 | ~10s | [run](../../actions/workflows/nb-02-calculus-autodiff.yml) |
| 03 | [Probability, Statistics and Information Theory](notebooks/03-probability-information/) | [Module 03](https://github.com/anuxlab/deep-learning-systems/wiki/Module-03) | Bayes, distributions, covariance vs independence, MLE, Monte Carlo, entropy/KL/MI, stable cross-entropy | 19 | ~10s | [run](../../actions/workflows/nb-03-probability-information.yml) |
| 04 | [Biological Neurons, McCulloch–Pitts Neurons and Perceptrons](notebooks/04-neurons-perceptrons/) | [Module 04](https://github.com/anuxlab/deep-learning-systems/wiki/Module-04) | McCulloch–Pitts gates, perceptron + Novikoff bound, XOR impossibility by search, 2-layer XOR | 9 | ~7s | [run](../../actions/workflows/nb-04-neurons-perceptrons.yml) |
| 05 | [Activation Functions, Neural Network Units and Expressivity](notebooks/05-activations-expressivity/) | [Module 05](https://github.com/anuxlab/deep-learning-systems/wiki/Module-05) | Activation zoo + derivative checks, saturation, softmax/temperature, universal approximation, depth ⇒ 2^d pieces | 24 | ~9s | [run](../../actions/workflows/nb-05-activations-expressivity.yml) |
| 06 | [Loss Functions, Gradient Descent and Empirical Risk Minimization](notebooks/06-losses-gradient-descent/) | [Module 06](https://github.com/anuxlab/deep-learning-systems/wiki/Module-06) | Surrogate losses, robust regression, CE gradients, batch/SGD/mini-batch, 2/L threshold, conditioning, ERM gap | 16 | ~10s | [run](../../actions/workflows/nb-06-losses-gradient-descent.yml) |
| 07 | [Feedforward Neural Networks and Backpropagation](notebooks/07-mlp-backprop/) | [Module 07](https://github.com/anuxlab/deep-learning-systems/wiki/Module-07) | NumPy MLP + backprop with gradient checks, XOR/moons/regression, PyTorch cross-check, overfit-16 test | 19 | ~20s | [run](../../actions/workflows/nb-07-mlp-backprop.yml) |
| 08 | [Deep Network Optimization and Loss Landscapes](notebooks/08-optimization-landscapes/) | [Module 08](https://github.com/anuxlab/deep-learning-systems/wiki/Module-08) | SGD/momentum/Nesterov/AdaGrad/RMSProp/Adam/AdamW from scratch vs `torch.optim`, landscapes, saddles, schedules, clipping | 13 | ~18s | [run](../../actions/workflows/nb-08-optimization-landscapes.yml) |
| 09 | [Weight Initialization, Normalization and Gradient Flow](notebooks/09-init-normalization/) | [Module 09](https://github.com/anuxlab/deep-learning-systems/wiki/Module-09) | Symmetry, Xavier/He variance over 50 layers, BatchNorm/LN/GN/RMSNorm vs torch, residual gradient flow, deep MLP training | 24 | ~21s | [run](../../actions/workflows/nb-09-init-normalization.yml) |
| 10 | [Regularization, Generalization and Learning Theory](notebooks/10-regularization-generalization/) | [Module 10](https://github.com/anuxlab/deep-learning-systems/wiki/Module-10) | Leakage demo, bias–variance, L1/L2/weight decay, dropout/label smoothing/mixup, early stopping, ensembles, double descent | 14 | ~40s | [run](../../actions/workflows/nb-10-regularization-generalization.yml) |
| 11 | [Convolutional Neural Networks and Advanced CNN Architectures](notebooks/11-cnn/) | [Module 11](https://github.com/anuxlab/deep-learning-systems/wiki/Module-11) | Conv from scratch vs torch, RF & params, pooling/conv backward, equivariance, CNN vs MLP, ResNet block, depthwise-separable | 21 | ~13s | [run](../../actions/workflows/nb-11-cnn.yml) |
| 12 | [Recurrent Neural Networks, LSTM and GRU](notebooks/12-rnn-lstm-gru/) | [Module 12](https://github.com/anuxlab/deep-learning-systems/wiki/Module-12) | BPTT + gradient check, gradient flow through time, LSTM/GRU cells vs torch, memory task, char-LM with sampling | 10 | ~125s | [run](../../actions/workflows/nb-12-rnn-lstm-gru.yml) |
| 13 | [Sequence-to-Sequence Learning and Attention Mechanisms](notebooks/13-seq2seq-attention/) | [Module 13](https://github.com/anuxlab/deep-learning-systems/wiki/Module-13) | Attention by hand, GRU seq2seq (none/additive/dot), length generalisation, alignment matrix, beam search | 10 | ~161s | [run](../../actions/workflows/nb-13-seq2seq-attention.yml) |
| 14 | [Transformer Architecture and Self-Attention](notebooks/14-transformer/) | [Module 14](https://github.com/anuxlab/deep-learning-systems/wiki/Module-14) | Scaled dot-product, √d_k, masks, permutation equivariance, MHA vs `nn.MultiheadAttention`, O(T²), tiny GPT | 20 | ~60s | [run](../../actions/workflows/nb-14-transformer.yml) |
| 15 | [Positional Encoding and Modern Attention Variants](notebooks/15-positional-encoding/) | [Module 15](https://github.com/anuxlab/deep-learning-systems/wiki/Module-15) | Sinusoidal, learned, T5-bias, **RoPE** (relative property proof), ALiBi, length extrapolation experiment, MQA/GQA | 20 | ~79s | [run](../../actions/workflows/nb-15-positional-encoding.yml) |
| 16 | [Autoencoders and Representation Learning](notebooks/16-autoencoders/) | [Module 16](https://github.com/anuxlab/deep-learning-systems/wiki/Module-16) | Linear AE = PCA, bottleneck sweep, sparse/denoising/contractive AEs, latent geometry, linear-probe/kNN | 9 | ~43s | [run](../../actions/workflows/nb-16-autoencoders.yml) |
| 17 | [Self-Supervised, Contrastive and Metric Learning](notebooks/17-contrastive-learning/) | [Module 17](https://github.com/anuxlab/deep-learning-systems/wiki/Module-17) | Triplet, InfoNCE/NT-Xent, SimCLR on digits, temperature, collapse vs VICReg, MoCo queue, CLIP loss | 14 | ~34s | [run](../../actions/workflows/nb-17-contrastive-learning.yml) |
| 18 | [Variational Autoencoders and Probabilistic Generative Modeling](notebooks/18-vae/) | [Module 18](https://github.com/anuxlab/deep-learning-systems/wiki/Module-18) | ELBO identity on a closed-form model, reparameterisation, VAE on digits, β-VAE, posterior collapse + free bits | 13 | ~29s | [run](../../actions/workflows/nb-18-vae.yml) |
| 19 | [Generative Adversarial Networks](notebooks/19-gan/) | [Module 19](https://github.com/anuxlab/deep-learning-systems/wiki/Module-19) | Optimal discriminator & JS, saturating vs non-saturating, 8-Gaussians ring, WGAN-GP, mode coverage/MMD, seed variance | 8 | ~87s | [run](../../actions/workflows/nb-19-gan.yml) |
| 20 | [Diffusion Models and Score-Based Generation](notebooks/20-diffusion/) | [Module 20](https://github.com/anuxlab/deep-learning-systems/wiki/Module-20) | Closed-form forward process, posterior, DDPM training, ancestral vs DDIM sampling, sliced-W2, classifier-free guidance | 12 | ~33s | [run](../../actions/workflows/nb-20-diffusion.yml) |
| 21 | [Graph Neural Networks and Geometric Deep Learning](notebooks/21-gnn/) | [Module 21](https://github.com/anuxlab/deep-learning-systems/wiki/Module-21) | Laplacian spectra, spectral clustering, GCN/SAGE/GAT/GIN from scratch, equivariance, semi-supervised karate club, expressiveness, over-smoothing | 18 | ~13s | [run](../../actions/workflows/nb-21-gnn.yml) |
| 22 | [Transfer Learning, Parameter-Efficient Fine-Tuning and Foundation Models](notebooks/22-transfer-foundation/) | [Module 22](https://github.com/anuxlab/deep-learning-systems/wiki/Module-22) | Scratch vs supervised vs self-supervised transfer (incl. **negative transfer**), forgetting + EWC/replay, LoRA from scratch, few-shot prototypes | 16 | ~37s | [run](../../actions/workflows/nb-22-transfer-foundation.yml) |
| 23 | [Large Language Models, Alignment, Retrieval and Multimodal Models](notebooks/23-llm-multimodal/) | [Module 23](https://github.com/anuxlab/deep-learning-systems/wiki/Module-23) | BPE from scratch, GPT + perplexity, mini scaling law, **KV cache**, top-k/top-p, reward model + DPO, TF-IDF RAG, ViT patches | 26 | ~245s | [run](../../actions/workflows/nb-23-llm-multimodal.yml) |
| 24 | [Evaluation, Interpretability, Robustness, Efficiency and Responsible AI](notebooks/24-evaluation-robustness-efficiency/) | [Module 24](https://github.com/anuxlab/deep-learning-systems/wiki/Module-24) | Imbalanced metrics, bootstrap CIs, calibration + temperature scaling, IG & sanity checks, FGSM + adv. training, int8/pruning/bf16, fairness, model card | 28 | ~34s | [run](../../actions/workflows/nb-24-evaluation-robustness-efficiency.yml) |

## Design principles

- **Free-runner friendly.** Tiny models, scikit-learn *digits* (bundled), synthetic data, `networkx` karate club, Python stdlib docstrings as an offline text corpus — no downloads, no GPU, every notebook finishes in minutes.
- **Verify, don't trust.** From-scratch implementations are tested against finite differences, closed forms, or PyTorch (`check(...)` prints ✅ and raises if it fails — a silent CI pass really means the maths held).
- **Honest results.** Where an experiment did *not* confirm the textbook story (e.g. negative transfer in notebook 22, tiny-budget GAN/WGAN behaviour in 19, mixed RNN-vs-LSTM outcomes in 12) the notebook says so rather than hiding it.
- **Reproducible.** Fixed seeds, an environment report in every run (`metrics.json → environment`), and a note that fixed seeds do not guarantee bit-identical results across hardware.

## License

See [LICENSE](LICENSE).
