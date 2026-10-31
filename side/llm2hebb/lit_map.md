## Literature map (CITED; about one page)

**Which local rules reproduce back-propagation, and when.**

| family | the local rule | relation to the BP gradient | conditions / costs | what it means for a *pretrained* LLM |
|---|---|---|---|---|
| delta rule (Widrow-Hoff) | dW = eta (y - p) x^T | **exact** gradient of one output layer (CE / squared error) | none | the read-out of any pretrained net can learn locally, exactly as BP would |
| predictive coding / inference learning (Whittington & Bogacz 2017; Song et al. 2020; Millidge, Tschantz & Buckley 2022; Millidge et al. 2023; Song et al. 2024) | every layer holds a prediction error; activities relax to equilibrium; dW_l = eps_l x_{l-1}^T (block-local) | approaches BP as the output nudge goes to 0 at equilibrium; **exactly** BP with the fixed-prediction assumption and step 1 (Z-IL) | needs the transposed Jacobian of the next layer (weight transport / symmetric feedback) and T >= L relaxation steps per update (about T x the cost of BP); works on arbitrary graphs incl. transformers | the forward pass is identical, so the weights carry over **exactly**; but the update then follows the gradient, so continued learning inherits BP's interference. "Prospective configuration" (relaxing activities first) is reported to reduce interference in small networks |
| equilibrium propagation (Scellier & Bengio 2017; Laborieux et al. 2021; Laborieux & Zenke 2022) | free and weakly nudged phases of an energy-based net; contrastive Hebbian dW ~ (x x^T)_nudged - (x x^T)_free | gradient as nudge beta -> 0 | needs an energy function and symmetric weights; reaches CIFAR/ImageNet-32 scale | a transformer is not an energy-based model, so it cannot be converted directly; it would have to be re-trained as one (attention is one step of a modern Hopfield net: Ramsauer et al. 2021; energy transformer: Hoover et al. 2023) |
| feedback alignment / DFA (Lillicrap et al. 2016; Nokland 2016; Launay et al. 2020) | error sent back through fixed random matrices | not the gradient; the forward weights *learn* to align with the random feedback during training | works from scratch, DFA even for transformers, with a gap to BP | a pretrained net was never aligned with any random feedback, so its updates start almost orthogonal to the gradient |
| target propagation (Bengio 2014; Lee et al. 2015; Meulemans et al. 2020; Ernoult et al. 2022) | each layer regresses to a target made by learned approximate inverses | Gauss-Newton-like, not the gradient | needs a trained inverse per layer; scales poorly (Bartunov et al. 2018) | the inverses would have to be trained (by some rule) on top of the pretrained net |

**Fast weights: the part of an LLM that already is Hebbian.** Linear attention is a fast-weight memory S += v k^T
(Schlag, Irie & Schmidhuber 2021; Ba et al. 2016). DeltaNet / Gated DeltaNet (Yang et al. 2024, 2025) and RWKV-7
(Peng et al. 2025) replace the sum by the delta rule S += beta (v - S k) k^T with decay, i.e. a local error-correcting
Hebbian update at every token, at inference time, with no back-propagation. Titans (Behrouz et al. 2025) and TTT layers
(Sun et al. 2024) push this to a small MLP updated by a local gradient at test time. Softmax transformers can be
linearised by a short fine-tune (T2R, Kasai et al. 2021; SUPRA, Mercat et al. 2024; LoLCATs, Zhang et al. 2024). Limit:
the state has a fixed size and decays; associative-recall studies find linear-state models fall behind softmax
attention on recall (Arora et al. 2024).

**Writing facts into the slow weights without BP.** MLPs act as key -> value memories (Geva et al. 2021). ROME/MEMIT
(Meng et al. 2022, 2023) write a fact with a rank-one update of delta-rule form; the target vector v* is found by
gradient descent (BP) at write time. Sequential editing degrades gradually and then collapses: ROME after about
100–1,000 edits, MEMIT after about 1,400 (Gupta, Rao & Anumanchipalli 2024).

**Episodic memory and complementary learning systems.** Catastrophic interference comes from dense, overlapping codes
trained by a slow learner on a non-stationary stream (McCloskey & Cohen 1989; French 1999). The brain's answer is a fast,
sparse hippocampus plus slow, interleaved cortical consolidation (McClelland, McNaughton & O'Reilly 1995; Kumaran,
Hassabis & McClelland 2016). LLM analogues: kNN-LM (Khandelwal et al. 2020), Memorizing Transformers (Wu et al. 2022),
Larimar's one-shot episodic memory (Das et al. 2024). Fine-tuning learns new facts slowly and raises hallucination
(Gekhman et al. 2024); retrieval beats fine-tuning for knowledge injection (Ovadia et al. 2024).
