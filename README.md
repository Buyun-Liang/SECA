# Semantically Equivalent and Coherent Attacks (SECA)

This repository is the official implementation of the NeurIPS 2025 paper [*SECA: Semantically Equivalent and Coherent Attacks for Eliciting LLM Hallucinations*](https://neurips.cc/virtual/2025/poster/119640).

Authors: [Buyun Liang](https://buyunliang.org/), [Liangzu Peng](https://liangzu.github.io/), [Jinqi Luo](https://peterljq.github.io/), [Darshan Thaker](https://darshanthaker.github.io/), [Kwan Ho Ryan Chan](https://ryanchankh.github.io/), and [René Vidal](https://www.grasp.upenn.edu/people/rene-vidal/).

[Project Website](https://huggingface.co/spaces/byliang/SECA) · [ArXiv](https://arxiv.org/abs/2510.04398) · [NeurIPS Page](https://neurips.cc/virtual/2025/poster/119640) · [Code](https://github.com/Buyun-Liang/SECA) · [Poster](https://buyunliang.org/images/posters/neurips_2025_seca_poster.png)

[![Website](https://img.shields.io/badge/Website-SECA_Project-6f42c1)](https://huggingface.co/spaces/byliang/SECA)
[![arXiv](https://img.shields.io/badge/arXiv-2510.04398-b31b1b)](https://arxiv.org/abs/2510.04398)
[![NeurIPS](https://img.shields.io/badge/NeurIPS-2025-blue)](https://neurips.cc/virtual/2025/poster/119640)
[![Code](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/Buyun-Liang/SECA)
[![Poster](https://img.shields.io/badge/Poster-PNG-ff69b4)](https://buyunliang.org/images/posters/neurips_2025_seca_poster.png)


## ✨ Abstract

<p align="center">
⚠️ <b>Warning: This method may be misused for malicious purposes.</b>
</p>

Large Language Models (LLMs) are increasingly deployed in high-risk domains. However, state-of-the-art LLMs often produce hallucinations, raising serious concerns about their reliability. Prior work has explored adversarial attacks for hallucination elicitation in LLMs, but it often produces unrealistic prompts, either by inserting gibberish tokens or by altering the original meaning. As a result, these approaches offer limited insight into how hallucinations may occur in practice. While adversarial attacks in computer vision often involve realistic modifications to input images, the problem of finding realistic adversarial prompts for eliciting LLM hallucinations has remained largely underexplored. To address this gap, we propose Semantically Equivalent and Coherent Attacks (SECA) to elicit hallucinations via realistic modifications to the prompt that preserve its meaning while maintaining semantic coherence. Our contributions are threefold: (i) we formulate finding realistic attacks for hallucination elicitation as a constrained optimization problem over the input prompt space under semantic equivalence and coherence constraints; (ii) we introduce a constraint-preserving zeroth-order method to effectively search for adversarial yet feasible prompts; and (iii) we demonstrate through experiments on open-ended multiple-choice question answering tasks that SECA achieves higher attack success rates while incurring almost no semantic equivalence or semantic coherence errors compared to existing methods. SECA highlights the sensitivity of both open-source and commercial gradient-inaccessible LLMs to realistic and plausible prompt variations. Code is available at https://github.com/Buyun-Liang/SECA.




![SECA Attack Example](seca_ex.png)

**Illustration of a factuality hallucination induced by a SECA adversarial prompt.** *The top two green boxes show the full attack prompt based on the original MMLU question in elementary mathematics, followed by the faithful and factual response from the target LLM. The bottom two blue boxes present a SECA-generated adversarial variant of the original prompt, with edits highlighted in red, and the corresponding target LLM explanation, which includes red-highlighted hallucinated content. In this example, the model selects the incorrect choice ('B') and generates a hallucinated explanation, showcasing a factuality hallucination.*

For more examples, please refer to our [project website](https://buyunliang.org/project_website/seca/).

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

## 📖 Citation

If you find our work useful, please consider citing our paper:

```bibtex
@inproceedings{liang2025seca,
  title={SECA: Semantically Equivalent and Coherent Attacks for Eliciting LLM Hallucinations},
  author={Liang, Buyun and Peng, Liangzu and Luo, Jinqi and Thaker, Darshan and Chan, Kwan Ho Ryan and Vidal, Ren{\'e}},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

For convenience, you may also cite the arXiv version:

```bibtex
@article{liang2025seca,
  title={SECA: Semantically Equivalent and Coherent Attacks for Eliciting LLM Hallucinations},
  author={Liang, Buyun and Peng, Liangzu and Luo, Jinqi and Thaker, Darshan and Chan, Kwan Ho Ryan and Vidal, Ren{\'e}},
  journal={arXiv preprint arXiv:2510.04398},
  year={2025}
}
```


## 📬 Contact

For questions or bug reports, please either:

* Open an issue in this [GitHub repository](https://github.com/Buyun-Liang/SECA/issues), or
* Email [Buyun Liang](https://buyunliang.org/) at: `byliang [at] seas [dot] upenn [dot] edu`

## 📄 License

The code is licensed under the MIT License. See [LICENSE](./LICENSE) for details.

## 🛡️ Disclaimer
The code and documentation in this repository are made available for research and educational purposes only, with no warranties or guarantees. Users are fully responsible for ensuring their work complies with applicable laws, regulations, and ethical standards. The authors disclaim any liability for misuse, damage, or harm resulting from the use of this material.
