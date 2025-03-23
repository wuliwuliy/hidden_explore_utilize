

<div align="center">

# Simple Reinforcement Learning for Reasoning

[![Notion](https://img.shields.io/badge/Notion-%23000000.svg?style=for-the-badge&logo=notion&logoColor=white)](https://hkust-nlp.notion.site/simplerl-reason) [![Hugging Face](https://img.shields.io/badge/SimpleRL-fcd022?style=for-the-badge&logo=Huggingface&logoColor=000)](https://huggingface.co/collections/hkust-nlp/simplerl-67b543892b2ec6908ffff710)

</div>


This repo contains a simple reinforcement learning recipe to improve models' reasoning abilities. It is simple because only rule-based reward and GSM8K/Math datasets are used. We have used this code to successfully train 10 diverse base models with limited data (8K examples), achieving surprisingly strong results -- the accuracy gains range from 10 to more than 20 absolute points. These models include Llama3 8B, Mistral 7B/24B, DeepSeekMath 7B, Qwen2.5 0.5B/1.5B/7B/14B/32B, and Qwen2.5-Math-7B. While we observe significant increase in both response length and accuracy, we note that different models exhibit distinct reasoning behaviors during training, and the increased response length does not necessarily correlate with emergence of certain cognitive behaviors such as self-verification. We share many findings and practices in our paper, and we release the code, model checkpoints, and analysis tools here. 

> You may find an old version of this repo [here](), with our early results and codebase using OpenRLHF and PPO.

<div align="center">
<img src="https://github.com/user-attachments/assets/bacd1680-ccb0-4921-a687-8a595ebf5896" width="700" alt="simplelr-reaoning-intro-figure_00">
</div>

> Training dynamics of our Qwen2.5-SimpleRL-Zero training starting from the Qwen2.5-Math-7B, without SFT or reward models.

## News
- **[2025/03/24]** We perform successful zero RL training starting from 10 diverse base models. We release all 10 models and the code, and share many findings and practices in our [paper]().   
- **[2025/02/19]** We release checkpoints of [Qwen-2.5-Math-7B-SimpleRL-Zero](https://huggingface.co/hkust-nlp/Qwen-2.5-Math-7B-SimpleRL-Zero) and [Qwen-2.5-Math-7B-SimpleRL](https://huggingface.co/hkust-nlp/Qwen-2.5-Math-7B-SimpleRL) to Huggingface. 
- **[2025/01/25]** We release the training/eval code and our blog. We are working on the paper and will release it very soon.

## Main Results


#### All results are in pass@1 accuracy


|                            | AIME 2024 | MATH 500 | AMC  | Minerva Math | OlympiadBench | Avg.  |
|---------------------------------|-----------|----------|------|--------------|---------------|-------|
| Qwen2.5-Math-7B-Base            | 16.7      | 52.4     | 52.5 | 12.9         | 16.4          | 30.2  |
| Qwen2.5-Math-7B-Base + 8K MATH SFT | 3.3       | 54.6     | 22.5 | 32.7         | 19.6          | 26.5  |
| Qwen-2.5-Math-7B-Instruct       | 13.3      | 79.8     | 50.6 | 34.6         | 40.7          | 43.8  |
| Llama-3.1-70B-Instruct          | 16.73     | 64.6     | 30.1 | 35.3         | 31.9          | 35.7  |
| rStar-Math-7B                   | 26.7      | 78.4     | 47.5 | -            | 47.1          | -     |
| Eurus-2-7B-PRIME                | 26.7      | 79.2     | 57.8 | 38.6         | 42.1          | 48.9  |
| Qwen2.5-7B-SimpleRL-Zero        | 33.3      | 77.2     | 62.5 | 33.5         | 37.6          | 48.8  |
| Qwen2.5-7B-SimpleRL             | 26.7      | 82.4     | 62.5 | 39.7         | 43.3          | 50.9  |

<!-- #### Increase of Response Length does not always correspond to the "aha moment"

#### Rigid Format Reward Harms Training of Some Base Models

#### Pass@K Improves Significantly

#### Traditional SFT as a Cold Start Harms RL -->


## Model Checkpoints
|Model|Link|
|-|-|
|Llama-3.1-8B-SimpleRL|[🤗]()|

## Quick Start

### Installation

Our code is implemented based on OpenRLHF. Please follow [OpenRLHF's guidance](https://github.com/OpenRLHF/OpenRLHF/tree/main?tab=readme-ov-file#installation) to configure required environments and install our version:

```bash
git clone https://github.com/hkust-nlp/simpleRL-reason.git
cd train
pip install -e .
```

### Reproducing SimpleRL-Zero
The minimum hardware requirement for training is 6 H/A100-80G GPUs (note: this configuration has not been tested yet). To accelerate our experiments, we used 4 nodes, each equipped with 8 H/A100-80G GPUs, to train on 8K MATH examples for 120 steps over approximately 1.5 days, achieving convergence. However, our results indicate that satisfactory performance can be achieved with around 60 steps, which requires less than one day of training using 4 nodes.

The training process leverages PPO with Ray and vLLM for acceleration. So firstly, you need to launch the ray cluster using the command below:
```bash
# launch the master node of ray in container
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

# if you want to launch ray on more nodes, use
ray start --address {MASTER-NODE-ADDRESS}:6379  --num-gpus 8
```

Next, submit the training job from the master node:

```bash
cd train
# For 4 nodes:
ray job submit --address="http://127.0.0.1:8265" \
        --runtime-env-json='{
        "pip": ["ray==2.12.0", "latex2sympy2", "timeout_decorator"]
    }' -- /bin/bash examples/script/train_ppo_qwen_base_math_lv35_new.sh

# For 1 node:
ray job submit --address="http://127.0.0.1:8265" \
        --runtime-env-json='{
        "pip": ["ray==2.12.0", "latex2sympy2", "timeout_decorator"]
    }' -- /bin/bash examples/script/train_ppo_qwen_base_math_lv35_1_node.sh

```


### Evaluate

We used [Qwen Math's codebase](https://github.com/QwenLM/Qwen2.5-Math/tree/main/evaluation) for evaluation, but for fairness considerations, we completely prohibited solving problems by calling code. Please follow the `/eval` instructions for evaluation


## Citation

If you find this blog or our code useful, we would appreciate it if you could cite our work:

```bibtex
@misc{zeng2025simplerl,
  title={7B Model and 8K Examples: Emerging Reasoning with Reinforcement Learning is Both Effective and Efficient},
  author={Weihao Zeng and Yuzhen Huang and Wei Liu and Keqing He and Qian Liu and Zejun Ma and Junxian He},
  year={2025},
  howpublished={\url{https://hkust-nlp.notion.site/simplerl-reason}},
  note={Notion Blog}
}
```


## Acknowledgement
We implement our reinforcement learning algorithm extending from [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF). We utilize [vLLM](https://github.com/vllm-project/vllm) for inference and develop evaluation scripts based on [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math/tree/main/evaluation). Particularly, we thank the developers of DeepSeek-R1 and Kimi-k1.5 for their innovation and contribution to the open-source community.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=hkust-nlp/simpleRL-reason&type=Date)](https://star-history.com/#hkust-nlp/simpleRL-reason&Date)


