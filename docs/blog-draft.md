# I Fine-Tuned a 2B Model for $0.40 and Spent the Rest of the Weekend on Calibration

I built a form abandonment classifier over a weekend. Gemma 2B, LoRA fine-tuned on 884 synthetic examples, deployed to both Cloudflare Workers AI at the edge and Modal with vLLM for calibrated inference. The model hit 0.916 macro-F1 against a zero-shot baseline of 0.063. That part was straightforward. The part that ate most of the weekend was making the model's confidence numbers mean something.

## The problem: six reasons a form gets abandoned

FormRecap captures interaction events as users fill out web forms. Focus, blur, input, scroll, exit. When someone abandons a form, the event trace tells you *what* happened, but not *why*. A binary "abandoned / completed" flag is nearly useless for recovery. If someone left because of a validation error, you show a tooltip. If they're comparison shopping, you send a follow-up email. If it's a bot, you ignore it.

Six classes: `validation_error`, `distraction`, `comparison_shopping`, `accidental_exit`, `bot`, `committed_leave`. Each maps to a different product response. A validation error means your form UX is broken and you need to fix it now. Comparison shopping means a well-timed discount code might convert. A bot means do nothing. The taxonomy was designed around actionability, not completeness.

The classifier takes a raw event trace and outputs one of the six classes plus a confidence score. The confidence score matters as much as the classification itself. A low-confidence classification should fall through to a human review queue or a more expensive model, not trigger an automated action on a real user. Getting the class right 92% of the time is good. Knowing *which* 8% you got wrong is better.

## Why fine-tune a small model

You could throw this at Claude or GPT-4 and get decent zero-shot results. For a product that processes every form abandonment event in real time, that's a non-starter. At scale you're looking at hundreds of milliseconds of latency and cents per call, compounding into real money fast.

A fine-tuned 2B parameter model runs on commodity hardware. On Cloudflare Workers AI, inference is sub-200ms from Australia. The training cost was $0.40 on a single Modal L4 GPU, about 30 minutes. Total project spend including all the data generation, failed experiments, and eval runs: under $15.

There's also a tunability argument. Zero-shot Gemma 2B scored 0.063 macro-F1 on this task. After fine-tuning: 0.916. That 14x improvement is the case for LoRA on small models. The base model has no idea what these event traces mean. After 884 examples, it does.

## Data pipeline

I generated 1,100 synthetic training examples using Claude Sonnet. Each prompt included a randomized form context (e-commerce checkout, insurance quote, job application, newsletter signup) and class-specific behavioral hints that told the generator *why* the user was abandoning. The form fields, timing between events, and interaction patterns all varied per call. Without this variation, the first generation run produced roughly 50% exact duplicates. LLMs generating synthetic data will happily produce the same 15-second checkout abandonment trace fifty times if you let them. The fix was per-call randomization, realistic timing jitter, and in-run dedup rejection at generation time. After exact-hash deduplication, 884 unique examples survived.

The test set is a separate concern entirely: 52 hand-labeled real examples. Not synthetic. These came from actual form interaction traces that I manually classified, deliberately including ambiguous edge cases. The synthetic/real split is important. If you evaluate a model trained on synthetic data against a synthetic test set, you're measuring how well it learned the generator's biases, not how well it handles the real world. 52 is thin, and I'd want at least 200 for production confidence intervals, but it was enough to establish generalization beyond the training distribution.

## The confidence detour

This is the part that justified the weekend. Classification accuracy gets you most of the way to a useful system. Calibration gets you the rest.

When you ask an LLM to output a confidence score, it gives you a number. "Confidence: 0.85." The problem is that number is largely made up. The model has learned that humans like seeing high confidence, and it obliges. Verbalized confidence from the fine-tuned Gemma 2B model had an Expected Calibration Error (ECE) of 0.145. When the model said 90% confident, it was right maybe 75% of the time.

Logprobs are better. Instead of asking the model to verbalize a number, you look at the actual probability the model assigned to the output token. The training format uses a leading digit (1-6) for each class, and each digit is a single token in the vocabulary. That gives you a clean scalar probability from the softmax, no parsing required. Raw logprob ECE: 0.103.

But raw logprobs are still miscalibrated. Neural networks are systematically overconfident, and fine-tuned models especially so. Temperature scaling is the simplest fix that actually works. You fit a single scalar T on a validation set by minimizing negative log-likelihood, then divide the logits by T before softmax. One number, one line of code, dramatic improvement. For Gemma 2B, the optimal temperature was 0.500, meaning the model's raw probabilities needed to be "spread out" significantly. After scaling, ECE dropped to 0.056. That means when the calibrated model says 80% confident, it's right about 80% of the time, give or take 5.6 percentage points. Good enough to make automated decisions on.

The leading-digit design was load-bearing for all of this. If the classification output were a multi-token string like "comparison_shopping", you'd need to aggregate probabilities across tokens and deal with tokenization ambiguity. A single digit means one token, one probability, one clean signal for calibration.

## The Cloudflare constraint that improved the architecture

Day one discovery: Cloudflare Workers AI BYO-LoRA does not expose logprobs. You upload your adapter, you get text completions, but no token-level probabilities. I confirmed this before writing any deployment code. This could have been a blocker. Instead it turned out to be a forcing function for a better design.

The constraint forced a dual-deployment architecture that turned out better than a single path would have been. Cloudflare Workers AI handles edge inference with verbalized confidence only, optimized for latency. Modal runs vLLM with full logprob extraction and temperature-scaled calibration for high-stakes paths where confidence accuracy matters. The gap between the two confidence methods is itself a measurable result: 0.145 ECE (verbalized, edge) vs 0.056 ECE (calibrated logprobs, Modal). For a low-stakes "show a tooltip" action, verbalized confidence is fine. For "automatically send this user a recovery email," you want the calibrated path.

## Results

| Model | Macro-F1 | ECE (verbalized) | ECE (logprob raw) | ECE (calibrated) |
|---|---|---|---|---|
| Zero-shot Gemma 2B | 0.063 | -- | -- | -- |
| Zero-shot Mistral 7B | 0.095 | -- | -- | -- |
| Mistral 7B CF LoRA | 0.760 | 0.245 | 0.071 | 0.075 |
| **Gemma 2B Full LoRA** | **0.916** | **0.145** | **0.103** | **0.056** |

Per-class F1 for the best model (Gemma 2B Full LoRA):

| Class | F1 |
|---|---|
| validation_error | 0.957 |
| distraction | 1.000 |
| comparison_shopping | 0.900 |
| accidental_exit | 1.000 |
| bot | 0.889 |
| committed_leave | 0.750 |

The weakest class is `committed_leave` at 0.750. This makes sense: a user who deliberately decides to leave looks a lot like several other abandonment patterns in the event trace. More real training data for this class would likely close the gap.

Gemma 2B outperforming Mistral 7B was a surprise. The Mistral adapter was trained under Cloudflare's LoRA constraints (specific quantization format, limited rank), which may have constrained it. The Gemma 2B full LoRA had no such restrictions.

Training details: QLoRA NF4, LoRA r=16, alpha=32, DoRA enabled. Three epochs, 30 minutes on a Modal L4 GPU. I used HuggingFace PEFT + TRL for the training loop. I started with Unsloth for its speed optimizations but dropped it for the plain HF stack after hitting compatibility issues with the export pipeline. Lesson reinforced: for a weekend project, boring tools that work beat fast tools that surprise you. The HF stack is slower to train but has zero surprises when it comes time to export adapters for two different deployment targets.

## What I'd do differently

52 test examples is the biggest weakness. Confidence intervals on 0.916 macro-F1 are wide (95% CI: 0.813-0.981). With 200+ real test examples, I'd trust the per-class numbers enough to make product decisions. Semantic deduplication would also help. I used exact-hash dedup on the event strings, which caught the obvious duplicates but not the near-duplicates that synthetic generation loves to produce. And a hyperparameter sweep over LoRA rank and learning rate would be worth the extra $5 in compute, especially since each run is only $0.40.

The other open question is whether CF Workers AI will add logprob support for BYO-LoRA. If it does, the dual-deploy architecture collapses into a single edge path with calibrated confidence, which is a simpler system.

## Links

- **Demo:** [lab.formrecap.com](https://lab.formrecap.com)
- **Source:** [github.com/jasonm4130/formrecap-lora-classifier](https://github.com/jasonm4130/formrecap-lora-classifier) (Apache 2.0)
- **Reproduce:** clone the repo, run `op run --env-file .env.op -- modal run training/modal_app.py`, trained model in 30 minutes
