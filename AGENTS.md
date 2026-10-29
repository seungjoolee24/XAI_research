# Latent Truth XAI: Milestone 1

- Scope: inference, neutral/false/correct opinion baselines, candidate scoring, hidden-state inspection only. Do not add training, probes, logit lens, patching, steering, or ablation without a new user request.
- Keep experiment logic in `src/truthgate`; notebooks are execution interfaces.
- Preserve the exact same chat-template prompt for candidate scoring and generation. Never silently truncate a prompt or replace an invalid generated answer with a candidate prediction.
- Distinguish raw next-token logits from summed sequence conditional log probabilities. Candidate-normalized probabilities are not calibrated factual confidence.
- Pair flips by `(question_id, option_order)` and distinguish harmful flips, beneficial flips, invalid responses, and candidate-only changes.
- Use `eval()` and `torch.inference_mode()`, batch size one, and explicit dtype/device selection. Never silently offload GPU experiments to CPU.
- Save only small summaries and last-position state vectors in memory. Do not commit model weights, large activations, or experiment outputs.
- Tests must work offline with small deterministic fixtures or a randomly initialized tiny Qwen3 configuration. Never report test outputs as scientific model results.
- Run `python -m pytest` after relevant changes. For reproducibility, retain effective config, model/tokenizer revision, dataset digest, seed, dependency versions, and runtime metadata.
- Real Qwen3-4B GPU execution must be reported separately from local fixture/integration validation.
- Before every research inference run (including smoke inference, baseline, and hidden-state inspection), generate and deliver a pre-experiment PDF describing the purpose, exact dataset/subset and conditions, core pseudocode, effective settings, planned measures, and interpretation limits. Use the same effective configuration for the PDF and the run; do not start research inference until the user has been supplied this PDF. Generating a PDF during model loading without supplying it first does not satisfy this requirement.
- After every research run, deliver a post-experiment PDF with the actual run status, statistics and denominators, invalid-response counts, harmful/beneficial and candidate-only changes where applicable, noteworthy observations, key conclusions, and limitations. A failed or interrupted run must be reported honestly, with partial results clearly identified; never invent missing results or report fixture outputs as model findings.
- Keep pre/post reports linked to reproducibility evidence: effective config, dataset digest, model/tokenizer revisions, code/source hashes, seed, dependency versions, device/dtype and runtime metadata. Store generated PDFs with experiment outputs, outside Git; commit only report-generation code and required small assets. Offline unit/integration tests are software verification and do not constitute a scientific experiment.
