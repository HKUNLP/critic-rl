<h1 align="center"> Teaching Language Models to Critique via Reinforcement Learning </h1>

<p align="center">
<a href="https://zhxie.site/">Zhihui Xie</a>*<sup>1</sup>, <a href="https://scholar.google.com.sg/citations?user=xrnhH-cAAAAJ&hl=zh-CN">Jie Chen</a>*<sup>2</sup>, <a href="https://lchenat.github.io/">Liyu Chen</a><sup>2</sup>, <a href="https://weichaomao.web.illinois.edu/">Weichao Mao</a><sup>2</sup>, <a href="https://jingjingxu.com/">Jingjing Xu</a><sup>2</sup>, <a href="https://ikekonglp.github.io/">Lingpeng Kong</a><sup>1</sup>
</p>

<p align="center">
<sup>1</sup>The University of Hong Kong,</span>
<sup>2</sup>Bytedance, Seed</span>
</p>

<p align="center">
<a href="https://arxiv.org/abs/2502.03492">📃 Paper</a>
•
<a href="https://critic-rl.github.io/" >🔭 Project Page</a>
•
<a href="https://huggingface.co/Zhihui/CTRL-32B" >🤗 Model</a>
</p>

We propose **CTRL**, a framework that trains LLMs to critique **without human supervision**, enabling them to **supervise stronger models** and **achieve test-time scaling** through iterative critique-revisions.

<p align="center">
  <img src="https://critic-rl.github.io/static/images/illustration_v2.png" width="100%" alt="teaser">

## 🎯 Key Results

<p align="center">
  <img alt="Light" src="https://critic-rl.github.io/static/images/scaling_v3.png" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="https://critic-rl.github.io/static/images/c2w.png" width="45%">
</p>

- **Test-time Scaling**: Qwen2.5-Coder-32B-Ins with the CTRL critic achieves 106.1% relative improvement in Pass@1 on CodeContests through multi-turn critique-revision, while maintaining low error rates across iterations

- **Model-Agnostic**: The CTRL critic improves performance across different models (23.5% improvement with GPT-4o) and tasks (CodeContests, LiveCodeBench, MBPP+)

- **Critics-as-RMs**: The CTRL critic achieves 64.3% accuracy on JudgeBench as a generative reward model, competitive with stronger models like Claude-3.5-Sonnet

See our [project page](https://critic-rl.github.io/) for detailed analysis and more results.


<!-- Installation -->
## 📦 Installation

You can install the dependencies with the following command:
```bash
pip install -r requirements.txt
```

For evaluating code correctness, we use [SandboxFusion](https://bytedance.github.io/SandboxFusion/docs/docs/get-started) to deploy the code sandbox.

<!-- sandbox deployment -->
```bash
docker run -it -p 8080:8080 vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20241204

export SANDBOX_FUSION_ENDPOINT="your_sandbox_endpoint"
```

<!-- Data Prepration -->
## 📁 Data Preparation
We use the [TACO](https://huggingface.co/datasets/BAAI/TACO) dataset for training. Preprocess the data using:
```bash
python scripts/data/preprocess_taco.py
```

## 🚀 Training
Our training process consists of two stages: (1) SFT on synthetic data guided by execution feedback and (2) RL with verifiable rewards and GRPO.

### Stage I: SFT
We start with generating synthetic data with the following command:
```bash
bash examples/gen_taco.sh
```

Then, fine-tune the model using the following command:
```bash
bash examples/train_sft.sh
```

### Stage II: RL
Train with GRPO using verifiable rewards from sandbox execution:
```bash
bash examples/train_rl.sh
```

## 📋 Evaluation
We evaluate the model with the following command (e.g., for CodeContests):
```bash
bash examples/eval_codecontests.sh
```

## 🔧 Development
Please install the pre-commit hooks with the following command, before committing your code:
```bash
pip install pre-commit black
pre-commit install
```

## 📚 Citation
If you find this project useful, please consider citing:
```bibtex
@article{xie2025teaching,
  title={Teaching Language Models to Critique via Reinforcement Learning},
  author={Xie, Zhihui and Chen, Liyu and Mao, Weichao and Xu, Jingjing and Kong, Lingpeng and others},
  journal={arXiv preprint arXiv:2502.03492},
  year={2025}
}
```

## 💐 Acknowledgement
This project builds upon several amazing open-source projects:
- [verl](https://github.com/volcengine/verl): RL training framework
- [deepseek-coder](https://github.com/deepseek-ai/deepseek-coder): SFT training scripts
- [SandboxFusion](https://bytedance.github.io/SandboxFusion/docs/docs/get-started): Code execution environment

## 📝 License
This project is licensed under the [Apache 2.0 license](LICENSE).
