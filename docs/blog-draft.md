# I Fine-Tuned Once, Deployed Twice, and Learned What Calibration Actually Looks Like

*Target: dev.to, 1500-2000 words*

---

## Hook

One-paragraph opener: built a form abandonment classifier over a weekend — LoRA fine-tuned Llama 3.2 3B on synthetic data, deployed to both Modal (logprobs + calibration) and Cloudflare Workers AI (edge). The interesting part wasn't the model — it was what happened when I tried to make it trustworthy.

---

## 1. The Problem: Why Forms Get Abandoned

- FormRecap tracks form interaction events (focus, blur, input, scroll, exit)
- When a user abandons a form, the event trace tells a story — but what story?
- 6-class taxonomy: validation_error, distraction, comparison_shopping, accidental_exit, bot, committed_leave
- Why classification matters: different abandonment reasons → different recovery strategies

## 2. Why Fine-Tune a Small Model?

- Zero-shot LLMs can do this — but at what cost and latency?
- The case for Llama 3.2 3B: smallest model with strong "tunability" (biggest delta from fine-tuning)
- LoRA makes it cheap: ~$0.40 per training run on a single L4 GPU
- Edge deployment via CF Workers AI BYO-LoRA: sub-200ms p50 from Australia

## 3. Synthetic Data: Teaching a Student with a Teacher

- 1100 synthetic examples generated with Claude Sonnet 4.5
- The diversity problem: first run had 50% exact duplicates (!)
- Fix: per-call form context randomization, behavioral hints per class, temperature=1.0, in-run dedup rejection
- 52 hand-crafted real test examples as the only eval that matters
- [Show: class distribution table]

## 4. Training: The Boring Part (That Should Be Boring)

- QLoRA NF4 + LoRA r=16 + DoRA + Unsloth on Modal
- 3 epochs, ~30 minutes, ~$0.40
- The setup is the product: reproducible, config-driven, one command
- [Show: training config table]

## 5. The Confidence Detour (The Interesting Part)

This is the section that makes the post worth reading.

- Initial assumption: model says "confidence: 0.85" → use that number
- Reality: verbalized confidence is poorly calibrated (LLMs are confidently wrong)
- The logprobs approach: extract probability of the leading digit token (1-6)
  - Each digit is a single token in Llama's vocabulary → clean signal
- Temperature scaling: one scalar T fit on validation set, minimizes ECE
- [Show: ECE comparison table — verbalized vs raw logprob vs calibrated logprob]
- [Show: reliability diagram before/after temperature scaling]

### The Cloudflare Constraint

- CF Workers AI BYO-LoRA does NOT expose logprobs (confirmed day 1)
- This forced the dual-deployment architecture:
  - **Edge (CF):** fast inference with verbalized confidence only
  - **Modal:** full logprobs + temperature-scaled calibration for high-stakes paths
- The gap between the two confidence paths is itself a measurable, interesting result

## 6. Results

- [Show: 6-row baseline comparison table]
  - Majority class baseline
  - Zero-shot Llama 3.2 3B
  - 5-shot Llama 3.2 3B
  - Claude Haiku
  - LoRA fine-tune (our model)
  - (Stretch: LoRA 1B)
- Key numbers: macro-F1, per-class F1, ECE before/after calibration
- What the model gets right, where it struggles (likely: distraction vs committed_leave confusion)

## 7. The Protection Stack (Brief)

- Weekend project ≠ unprotected endpoint
- Turnstile + rate limiting + HMAC tokens + daily neuron budget + KV kill switch
- One paragraph, link to code

## 8. What I'd Do Differently

- Semantic dedup (dropped for simplicity — exact hash was sufficient for synthetic data, wouldn't be for real)
- More real test data (52 is thin)
- Hyperparameter sweep (explicitly out of scope, but one knob I'd turn: LoRA rank)
- CF logprobs gap: hoping this gets addressed — would simplify to single deployment

## 9. Try It / Build It

- Live demo: lab.formrecap.com
- Full source: github.com/jasonm4130/formrecap-lora-classifier (Apache 2.0)
- Reproduce: clone → `op run --env-file .env.op -- modal run training/modal_app.py` → trained model in 30 min

---

## Visual Assets Needed

- [ ] Architecture diagram (train once, deploy twice)
- [ ] Class distribution bar chart
- [ ] Baseline comparison table (the centrepiece)
- [ ] ECE before/after table
- [ ] Reliability diagram (calibration plot)
- [ ] Cost breakdown (total spend < $15)

## LinkedIn Post Draft (Separate)

~300 words, Builder bucket. Key beats:
- Fine-tuned Llama 3.2 3B for form abandonment classification
- Train: 1100 synthetic examples, $0.40 on Modal L4
- Deploy: same adapter to Modal (logprobs) + CF Workers AI (edge)
- Headline number: ECE from X → Y after temperature scaling
- The CF constraint that made it interesting
- Link to demo + blog + repo
- No "excited to share" — numbers first, then context
