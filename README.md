# Catalog Review RL Lab: TRL GRPO on Snowflake Cortex Training

Train a small open model to review marketplace listings with reinforcement learning, on governed, PII-redacted data in Snowflake. For each listing it returns the category, the brand, and the policy violation, if any. You'll compare it with the untrained model, a few-shot prompt, SFT, and a frontier model, and learn to tell a real result from a leaky one.

> Synthetic data, demo results. Nothing here is a measured business outcome.

## What's in here

```
catalog-rl-lab/
├── GRPO.md                        Visual walkthrough of one GRPO step on a real listing
└── lab/
    ├── requirements.txt           Pinned PyPI deps (trl, arctic-platform[cortex], ...)
    ├── fetch_recipe.sh            Downloads only the TRL-on-Cortex recipe folder from Arctic Platform
    ├── cortex_env.sh              Exports ARCTIC_CORTEX_* from your profile, never prints the PAT
    ├── 01_governed_dataset.sql    v1 data, "the trap": one phrasing per class, random row split
    ├── 02_harder_dataset.sql      v2 data, "the fix": 8 phrasings per class, test phrasings held out,
    │                              tricky clean listings, AI_COMPLETE paraphrase, PII re-checked
    ├── listing_task.py            Prompt + business-weighted reward (shared by training and eval)
    ├── run_listing_grpo_cortex.py TRL GRPO driver: reuses the Arctic recipe, swaps data + reward,
    │                              saves a weights-only checkpoint, fills the deploy spec
    ├── run_listing.sh             Launcher with the recipe's tested defaults
    ├── build_sft_data.py          Same data, as SFT conversations, for the SFT baseline
    ├── deploy_base.json           Model deployment spec for the untrained base model
    ├── deploy_tuned.json          Model deployment spec template for a trained checkpoint
    ├── evaluate.py                Any number of models + few-shot + frontier, saves raw outputs
    ├── analyze.py                 Confusion matrices + before/after examples from saved outputs
    ├── plots.py                   Reward curves and results chart (writes ./images/)
    └── roi.py                     Yearly cost worksheet from eval metrics + your assumptions
```

## Base model

**[`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B)**, trained with full parameters. It's already in the Cortex Training model catalog, so you pick it by model ID. It's the model the TRL-on-Cortex recipe was tested with on 4 training and 4 sampling GPUs, and it's small enough to make mistakes that RL can fix. To scale up, set `MODEL=Qwen/Qwen3-8B` in Step 4 and deploy the same model as the base in Step 5.

**What gets downloaded.** The ~4 GB of model weights never come to your machine; Cortex's training, sampling and deployment jobs load them. The driver only fetches the tokenizer and config from Hugging Face, about 16 MB:

- **The tokenizer** turns prompts into token IDs and decodes sampled answers so the reward can score them.
- **The config** builds a one-layer placeholder model with random weights, because TRL's trainer won't start without a model object. It's never trained or served.

New to GRPO? [GRPO.md](GRPO.md) walks through one training step with a real listing.

## Prerequisites

- **Snowflake account** with Cortex Training enabled and capacity for 8 H200 GPUs. Check with `cortex-training capacity`.
- **A role** that can create tables, stages and masking policies in one schema, and can call `AI_REDACT` and `AI_COMPLETE`.
- **A programmatic access token (PAT)** for that user. Never print or commit it.
- **Local tools:** Python 3.12, `uv`, `git`, and the [Snowflake CLI](https://docs.snowflake.com/en/developer-guide/snowflake-cli/index) (`snow`). No local GPU.

## Step 0: install and connect

```bash
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install cortex-training -r lab/requirements.txt
cortex-training login cortex-training   # creates a PAT-based connection profile
cortex-training capacity                # confirm 8 H200 GPUs are free
```

## Step 1: build the governed datasets

Replace `<DB>`, `<SCHEMA>`, `<WH>` and `<ANALYST_ROLE>` in both SQL files. Run them as `<ANALYST_ROLE>`, the only role that can see unmasked text, so it can do the redaction.

```bash
snow sql -f lab/01_governed_dataset.sql --role <ANALYST_ROLE>   # v1: the trap
snow sql -f lab/02_harder_dataset.sql   --role <ANALYST_ROLE>   # v2: the fix
mkdir -p data && snow stage copy @<DB>.<SCHEMA>.RL_DATA ./data/
```

What to check in the output:

- **Leak checks return 0** for emails, phones and names in both redacted tables.
- **v2 split:** about 3,650 train and 1,230 test rows, and test contains only phrasings 6 and 7.
- **Paraphrase QA:** rows where `AI_COMPLETE` dropped or invented contact details are removed (about 2%).
- **Masking:** as `CATALOG_RL_VIEWER`, `SELECT seller_name, description` returns `***MASKED***`. If you see raw text, read the masking item under Troubleshooting.

## Step 2: fetch the TRL-to-Cortex adapter

```bash
bash lab/fetch_recipe.sh     # writes ./recipe
RECIPE=$PWD/recipe
```

- **Why `trl==1.17.1`:** TRL 1.18 reorganized its experimental async GRPO module, and the Cortex adapter imports it from the old location. 1.17.1 is the latest release that works.
- **Why fetch a folder:** the TRL-to-Cortex glue (`arctic_trl_cortex/`, six files) isn't on PyPI. It lives in [Arctic Platform's recipe folder](https://github.com/Snowflake-AI-Research/Arctic-Platform/tree/main/recipes/rl/trl/gsm8k_cortex), and cortex-training's [TRL page](https://github.com/snowflakedb/cortex-training/tree/main/recipes/trl/gsm8k_grpo) points there. The script downloads just that folder at a pinned commit.

## Step 3: point the driver at Cortex

```bash
source lab/cortex_env.sh cortex-training
```

## Step 4: train with GRPO

Smoke-test 2 steps first. It checks the connection, data loading, reward and checkpoint save in about 5 minutes.

```bash
cd "$RECIPE"
DATA=$OLDPWD/data/listings_v2_train.jsonl MAX_STEPS=2 bash run_listing.sh
```

A healthy smoke test logs `dataset ready`, per-step `reward` values between -1 and 1, `[listing] saved ... on training job <JOB_ID>`, and `[listing] wrote deploy_tuned.json`.

Then the full run, about 30 minutes for 200 steps. Give each run its own checkpoint ID and deploy spec:

```bash
cp deploy_tuned.json deploy_v2_s42.json
DATA=$OLDPWD/data/listings_v2_train.jsonl SEED=42 CHECKPOINT_ID=listing-v2-s42 \
  DEPLOY_SPEC=deploy_v2_s42.json METRICS_OUT=metrics_v2_s42.json bash run_listing.sh > v2_s42.log 2>&1
```

- **Run a second seed.** Repeat with `SEED=7` and new names. Our two seeds scored 0.951 and 0.924 and made opposite wrong-type mistakes.
- **The v1 trap run:** use `DATA=.../listings_train.jsonl` instead.
- **Stopping a run:** use `kill -INT <pid>`, not `SIGTERM` or `kill -9`. That lets the driver save the checkpoint and cancel the Cortex job.
- **Tunable defaults:** `MODEL`, `MAX_STEPS`, `NUM_GEN`, `PER_DEVICE_BSZ`, `MAX_COMPLETION_LEN`.

## Step 4b (optional): SFT baseline on the same data

SFT uses the cortex-training repo's own SFT recipe. Register the dataset, then train for about 1 epoch (450 steps at batch size 8, on 8 GPUs, about 10 minutes):

```bash
git clone https://github.com/snowflakedb/cortex-training && cd cortex-training
uv pip install -e . 'tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly'
python ../lab/build_sft_data.py ../data/listings_v2_train.jsonl \
  recipes/sft_quickstart/conversational/data/listings_v2_sft.jsonl
# add to CHAT_DATASETS in recipes/sft_quickstart/conversational/task.py:
#   BuiltinJsonl(name="listings_v2_sft", path=_DATA_DIR / "listings_v2_sft.jsonl"),
python -m recipes.sft_quickstart.conversational.train model=Qwen/Qwen3-1.7B \
  dataset=listings_v2_sft max_steps=450 renderer_name=qwen3_disable_thinking \
  connection=cortex-training
```

At the end, the recipe prints a `source_job_id` and a `checkpoint_id`. Put them into a copy of `deploy_tuned.json`.

## Step 5: deploy

Training holds all 8 GPUs, so deploy after it finishes. Each deployment takes 1 GPU and is READY in about 3 minutes.

```bash
cortex-training model-deployment create --spec lab/deploy_base.json
cortex-training model-deployment create --spec $RECIPE/deploy_v2_s42.json
cortex-training model-deployment list        # wait for every one to show READY
```

All deployments share one URL; the deployment ID picks the model:

```bash
export URL=https://<account>.snowflakecomputing.com/api/v2/databases/CORTEX_TRAINING_DB/schemas/PUBLIC/cortex-training/inference/v1
```

## Step 6: evaluate, analyze, chart

```bash
cd lab
python evaluate.py --test ../data/listings_v2_test.jsonl --n 400 --tag v2 --url $URL \
  --model base=sf-ct-... --model rl_s42=sf-ct-... --model sft=sf-ct-... \
  --fewshot-for base --frontier claude-sonnet-4-5
python analyze.py --a v2/base --b v2/rl_s42 --n 4    # confusion matrices + before/after
python plots.py                                      # images/04-reward.png, 05-results.png
python roi.py --eval ../eval_results.json            # yearly cost worksheet, your assumptions
```

`evaluate.py` merges results into `eval_results.json` and appends every raw answer to `eval_completions.jsonl`. Metrics:

- `missed_violation_rate`: the expensive mistake, so read this first
- `false_alarm_rate`: clean listings wrongly flagged
- `recall_by_type`: per-violation accuracy. This is where wrong-type habits show up
- `reward`, `valid_json`, `category_acc`, `brand_acc`, `violation_acc`, `top_category_errors`

**Read the errors, not just the scores.** In our run, the frontier model's "false alarms" were mostly correct: it spotted brand-product mismatches in the synthetic data, like JBL webcams. The generator pairs brands with products they don't make, and the labels call those listings clean. That's a known label flaw in this demo.

## Step 7: clean up

Deployments hold GPUs until you drop them.

```bash
cortex-training model-deployment drop sf-ct-...
cortex-training model-deployment list     # nothing READY
cortex-training capacity                  # 8 available
```

## Results from our run

400 held-out v2 listings, phrasings no model saw in training:

```
MODEL                 REWARD  MISSED  FALSE ALARM  CATEGORY
--------------------  ------  ------  -----------  --------
Qwen3-1.7B base       0.681   23.9%    1.2%         53%
base + 4-shot prompt  0.664   55.6%    0.0%         94%
SFT                   0.739    1.4%   48.4%        100%
GRPO, seed 42         0.951    4.9%    0.0%        100%
GRPO, seed 7          0.924    0.7%    0.8%        100%
Claude Sonnet 4.5     0.814    0.7%   34.9%         99%
GRPO trained on v1    0.897    4.2%    5.8%         94%   (scored 1.000 on v1's own test set)
```

## Troubleshooting

These are the issues we hit running this lab:

- **`SSLCertVerificationError: Hostname mismatch`.** TLS certificates cover the dashed hostname. If your account name has an underscore (`my_acct`), the host must use dashes instead (`my-acct`). `cortex_env.sh` converts it for you.
- **The masking policy shows raw PII to other roles.** `IS_ROLE_IN_SESSION()` also counts secondary roles. With `DEFAULT_SECONDARY_ROLES = ('ALL')`, anyone who holds the owner role sees raw data in every session. Check `CURRENT_ROLE()` instead, as both SQL files do, and test as a separate role.
- **Frontier eval scores 0% valid JSON.** The Python connector returns `AI_COMPLETE` output as a JSON-encoded string. Decode it once before parsing; `evaluate.py` does.
- **Deployment rejects the job ID.** Use the parent job UUID, not `<uuid>:training:0`. The driver writes the parent ID.
- **HTTP 400 when passing `chat_template_kwargs`.** The deployment endpoint rejects it. Plain calls already return non-thinking output for these checkpoints.
- **SFT fails with `no recommended renderer for 'Qwen/Qwen3-1.7B'`.** Pass `renderer_name=qwen3_disable_thinking`.
- **`ModuleNotFoundError: trl.experimental.async_grpo.async_rollout_worker`.** You have TRL 1.18 or newer. Install `trl==1.17.1`.
- **`snow` can't authenticate with MFA in scripts.** Download stage files with the PAT profile instead: `snowflake.connector.connect(connection_name="cortex-training")` then `GET @stage/file file://./data/`.
- **The PAT can't switch roles.** PATs are usually restricted to one role. Run the masking test from a Snowsight worksheet or an interactive session.
- **Reward stuck near -1.** The model isn't producing valid JSON. Check `log_completions` in the driver log. If the base model never gets it right, add a small SFT warmup before RL.
- **Reward flat at 1.0.** GRPO has stopped learning, because every group scores the same. Your data is probably too easy or leaking. See v1 vs v2.
- **HTTP 403 / 503 from a deployment.** 403: inference isn't enabled for the account. 503: not READY yet.

## References

- [Cortex Training](https://github.com/snowflakedb/cortex-training)
- [Arctic Platform TRL Cortex recipe](https://github.com/Snowflake-AI-Research/Arctic-Platform/tree/main/recipes/rl/trl/gsm8k_cortex)
- [Model deployment recipe](https://github.com/snowflakedb/cortex-training/tree/main/recipes/model_deployment)
- [AI_REDACT](https://docs.snowflake.com/en/sql-reference/functions/ai_redact)
- [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300)
