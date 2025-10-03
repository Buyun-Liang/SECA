# Semantically Equivalent and Coherent Attacks (SECA)

This repository is the official implementation of the NeurIPS 2025 paper [*SECA: Semantically Equivalent and Coherent Attacks for Eliciting LLM Hallucinations*](https://neurips.cc/virtual/2025/poster/119640), by [Buyun Liang](https://buyunliang.org/), [Liangzu Peng](https://liangzu.github.io/), [Jinqi Luo](https://peterljq.github.io/), [Darshan Thaker](https://darshanthaker.github.io/), [Kwan Ho Ryan Chan](https://ryanchankh.github.io/), and [René Vidal](https://www.grasp.upenn.edu/people/rene-vidal/).

## ✨ Abstract

Large Language Models (LLMs) are increasingly deployed in high-risk domains. However, state-of-the-art LLMs often produce hallucinations, raising serious concerns about their reliability. Prior work has explored adversarial attacks for hallucination elicitation in LLMs, but they often produce unrealistic prompts&mdash;by either inserting gibberish tokens or altering the original meaning&mdash;thus offering limited insight into how hallucinations may occur in practice. While adversarial attacks in computer vision often involve realistic modifications to input images, the problem of finding realistic adversarial prompts for eliciting LLM hallucinations has remained largely underexplored. To address this gap, we propose **S**emantically **E**quivalent and **C**oherent **A**ttacks (SECA) to elicit hallucinations via realistic modifications to the prompt that preserve its meaning while maintaining semantic coherence. Our contributions are threefold: (i) we formulate realistic attacks for hallucination elicitation as a constrained optimization problem over the input prompt space under semantic equivalence and coherence constraints; (ii) we introduce a constraint-preserving zeroth-order method to effectively search for adversarial yet feasible prompts; and (iii) we demonstrate through experiments on open-ended multiple-choice question answering tasks that SECA achieves higher attack success rates while incurring almost no constraint violations compared to existing methods. SECA highlights the sensitivity of both open-source and commercial gradient-inaccessible LLMs to realistic and plausible prompt variations.




![SECA Attack Example](seca_ex.png)

**Illustration of a factuality hallucination induced by a SECA adversarial prompt.** *The top two green boxes show the full attack prompt based on the original MMLU question in elementary mathematics, followed by the faithful and factual response from the target LLM. The bottom two blue boxes present a SECA-generated adversarial variant of the original prompt, with edits highlighted in red, and the corresponding target LLM explanation, which includes red-highlighted hallucinated content. In this example, the model selects the incorrect choice ('B') and generates a hallucinated explanation, showcasing a factuality hallucination.*

## 📦 Requirements 

To install requirements:

```setup
pip install -r requirements.txt
```

## 🤖 LLM Info

| **LLM Name**     | **Source / API Version**                                                                 |
|------------------|-------------------------------------------------------------------------------------------|
| Llama-3-3B       | [https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| Llama-3-8B       | [https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| Llama-2-13B      | [https://huggingface.co/meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) |
| Qwen-2.5-7B      | [https://huggingface.co/Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| Qwen-2.5-14B     | [https://huggingface.co/Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) |
| GPT-4o-mini      | gpt-4o-mini-2024-07-18 (API)                                                              |
| GPT-4.1-nano     | gpt-4.1-nano-2025-04-14 (API)                                                             |
| GPT-4.1-mini     | gpt-4.1-mini-2025-04-14 (API)                                                             |
| GPT-4.1          | gpt-4.1-2025-04-14 (API)                                                                  |


## 📊 Dataset

Please download the MMLU dataset from https://huggingface.co/datasets/cais/mmlu. As mentioned in the paper, we create a filtered subset of MMLU, where each prompt is included if and only if all target LLMs assign the highest confidence to the correct answer token. Please refer to *data/filtered_MMLU_index.json* for the exact questions used in our paper. Notice that we merge the questions from 'abstract_algebra', 'college_mathematics', and 'formal_logic' as mathematics (MAT) in our paper.

## 🚀 Demo

This repository includes a demo notebook [demo.ipynb](./demo.ipynb) that demonstrates how SECA attacks Llama-3-8B. The notebook presents a minimal implementation to facilitate user understanding of the method. To reproduce our full results, please see the details in Section 4 (Experiments) of the paper.

## ⚙️ Hyperparameters

**Core Task Settings**

| Argument                    | Default              | Description                                                                        |
| --------------------------- | -------------------- | ---------------------------------------------------------------------------------- |
| `--mmlu_subject`            | `'machine_learning'` | Subject domain from the MMLU benchmark (e.g., `philosophy`, `clinical_knowledge`). |
| `--mmlu_question_idx`       | `0`                  | Index of the MMLU question to attack.                                              |
| `--mmlu_dataset_split_type` | `'test'`             | Dataset split to use (`train`, `dev`, or `test`).                                  |
| `--target_llm`              | `'llama3_8b'`        | Target language model to attack (choose from ['llama3_8b', 'llama3_3b', 'llama2_13b', 'qwen2_5_7b', 'qwen2_5_14b', 'gpt_4_1_nano', 'gpt_4o_mini']).                  |

**SECA Configuration**

| Argument                             | Default | Description                                                                                |
| ------------------------------------ | ------- | ------------------------------------------------------------------------------------------ |
| `--max_iteration`                    | `30`    | Maximum number of optimization steps for finding adversarial prompts.                      |
| `--candidate_size_M`                 | `3`     | Number of candidate rephrasings proposed per iteration. (See Line 6 in Algorithm 1.)       |
| `--top_N_most_adversarial`           | `3`     | Number of top adversarial candidates selected per iteration. (See Line 10 in Algorithm 1.) |
| `--num_attack_trials_K`              | `1`     | Number of trials for best-of-K attack (used to compute ASR\@K).                            |
| `--termination_confidence_threshold` | `1.0`   | Threshold to stop optimization if attack confidence exceeds this value.                    |


**Other Settings**

| Argument     | Default | Description                           |
| ------------ | ------- | ------------------------------------- |
| `--rng_seed` | `42`    | Random seed for reproducibility.      |
| `--verbose`  | `False` | Enables verbose logging if specified. |


## 📬 Contact

For questions or bug reports, please either:

* Open an issue in this [GitHub repository](https://github.com/Buyun-Liang/SECA/issues), or
* Email [Buyun Liang](https://buyunliang.org/) at: `byliang [at] seas [dot] upenn [dot] edu`

## 📄 License

The code is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
