# LinkedIn Post Draft

Weekend project: fine-tuned a Gemma 2B LoRA to classify form abandonment from event traces. 884 synthetic training examples, $0.40 on a Modal L4 GPU, Macro-F1 = 0.916 on 52 hand-labeled real test examples.

Three things that mattered more than the model choice:

→ Calibration via logprobs. Verbalized confidence from the model ("confidence: 0.85") had an ECE of 0.145. Extracting the token logprob of the leading class digit and applying temperature scaling dropped that to 0.056. The model's text claims are unreliable. Its probability distribution is not.

→ Cloudflare Workers AI doesn't expose logprobs on BYO-LoRA endpoints. This forced a dual deployment: CF for edge latency (sub-200ms from Australia), Modal for the calibration-critical path. Better architecture than if the API had "just worked."

→ Cost controls on CF Workers AI. The paid tier has no native hard spend cap on AI inference. Built one in the Worker: KV-tracked daily neuron counter, HMAC demo tokens with 10-minute TTL, Turnstile verification, and a KV kill-switch. Total protection stack: 6 layers, zero monthly cost.

The zero-shot baseline (same model, no fine-tuning) scored F1 = 0.063. The fine-tuned adapter: 0.916. That's the LoRA delta on a 2B parameter model with 884 training examples.

Full writeup: [blog link]
Live demo: [demo link]
Repo: https://github.com/jasonm4130/formrecap-lora-classifier
Adapters on HuggingFace: [HF link]
