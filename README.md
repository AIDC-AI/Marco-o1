<p align="center">
    <img src="https://avatars.githubusercontent.com/u/172576026?s=200&v=4" width="150" style="margin-bottom: 0.2;"/>
<p>
<h2 align="center"> <a href="https://github.com/AIDC-AI/Marco-o1/">Marco-o1</a></h2>
<h5 align="center"> If you appreciate our project, please consider giving us a star ⭐ on GitHub to stay updated with the latest developments.  </h2>

<h4 align="center">

🚀 Welcome to the repo of **Marco-o1**!

[![🤗Hugging Face](https://img.shields.io/badge/🤗Hugging_Face-Marco_o1-yellow)](https://huggingface.co/) [![Project Page](https://img.shields.io/badge/Project_Page-Marco_o1-blue)](https://github.com/AIDC-AI/Marco-o1/)

<div align="center">
<img src="https://img.shields.io/badge/Version-1.0.0-blue.svg" alt="Version"> 
<img src="https://img.shields.io/badge/License-CC%20BY%204.0-green.svg" alt="License">
<img src="https://img.shields.io/github/stars/AIDC-AI/Marco-o1?color=yellow" alt="Stars">
<img src="https://img.shields.io/github/issues/AIDC-AI/Marco-o1?color=red" alt="Issues">
<img src="https://img.shields.io/badge/python-3.8-purple.svg" alt="Python">

<br>
<br>
Yu Zhao, Huifeng Yin, Longyue Wang

</h4>

## 🔥 News

- [2024/10/27] 🔥 We released **Marco-o1**. This initial release includes our reasoning model, optimized for complex problem-solving and versatile applications across various domains.

## Introduction

Recently, OpenAI announced the launch of its groundbreaking product, OpenAI o1, formerly known by the codename "Strawberry." This next-generation generative AI model distinguishes itself from traditional counterparts through its enhanced reasoning capabilities, enabling it to tackle complex problems. OpenAI o1 has demonstrated remarkable performance across a variety of challenging benchmarks. Notably, it has excelled in the American Invitational Mathematics Examination (AIME), showcasing superior problem-solving skills that surpass those of the current leading model, GPT-4o. Additionally, o1 has achieved outstanding results on competitive programming platforms such as CodeForces and LeetCode Pro Max, further underscoring its advanced computational abilities. In several academic disciplines, OpenAI o1 has even outperformed individuals holding PhDs, highlighting its potential to revolutionize fields.

Acknowledging the significance of these advancements, we have replicated the technical roadmap of the OpenAI o1 model. Our replication effort aims to further explore and validate the state-of-the-art capabilities of this groundbreaking model, as well as to contribute to the broader understanding and development of AI technologies. Through this endeavor, we seek to push the boundaries of what AI can achieve in complex problem-solving and interdisciplinary applications. This reproduction effort aims to elucidate the architectural innovations and training strategies that contribute to o1's superior performance. We presents our replication process, the challenges encountered, and the insights gained, thereby contributing to the broader discourse on building more capable and reliable AI models.

**Our primary contributions are twofold:**

1. **Discussion and Analysis of o1's Technical Roadmap:** We provide an in-depth discussion and propose hypotheses regarding the technical strategies and innovations that underpin the OpenAI o1 model.
2. **Replication of the o1 Model:** We have made significant efforts to replicate the o1 model, achieving preliminary and stage-wise results that offer valuable insights into its methodologies and performance.

## Conjectures on the Implementation of o1

This section aims to speculate on the implementation of o1, drawing information primarily from insights shared by OpenAI researchers and various analyses.

Small models combined with infinitely long Chains-of-Thought (CoT) can solve any problem in the world. Therefore, o1 must be a model that enhances the ability to output correct answers by utilizing extremely long CoTs.

\fig

The discussion will be structured into the following parts:

Firstly, o1 is a model rather than a system. Thus, models like o1 require strong reasoning capabilities and robust conversational abilities to perform actions such as self-critique.

\fig

Considering that the o1 System Card repeatedly mentions that o1-mini has poor world knowledge, it is speculated that o1 is a small model trained from scratch. Therefore, the overall technical stack can be broken down into:

- **Pre-training Phase:**
  
  - Strengthen reasoning abilities.
- **Post-training Phase:**
  
  - **Supervised Fine-Tuning (SFT):**
    - Continue to enhance reasoning abilities.
  - **Reinforcement Learning (RL):**
    - Explore ways to expand the model's solution space.
    - Train a more effective reward model.
- **Inference Phase:**
  
  - Best of N sampling.

### Pre-Training Phase

In this phase, since the subsequent RL stage allows for significant flexibility, the relative importance of the base model might be lower. Due to the use of tree search later on, the model doesn't necessarily need to be extremely strong—since the solution space is vast, there's always a path that leads to the correct answer. However, a well-pretrained model can effectively reduce the computational cost in the subsequent RL phase.

Enhancing the model's reasoning abilities is crucial during pre-training. Known strategies include increasing the proportion of data related to code, mathematics, and academic papers while reducing data from sources like Common Crawl (cc) to improve the model's training Return on Investment (ROI).

### Supervised Fine-Tuning (SFT) Phase

This phase shares the same goal as the pre-training phase: to increase the proportion of CoT and conversational data, thereby enhancing the model's ability to produce lengthy CoTs and engage in dialogues.

### Reinforcement Learning Phase

OpenAI has invested heavily in reinforcement learning, and it has always been one of their strengths. OpenAI claims that RL can save up to 30 times the computational resources compared to improvements made during the pre-training phase.

Insights from interviews about o1 can be summarized into the following key points:

- **RL-generated CoTs are superior to human-generated ones:** Humans often prefer content that is easy to understand rather than content that is logically rigorous.
- **Challenges exist but RL is promising:** Despite challenges like reward design, RL remains a viable and ongoing pathway.
- **CoT combined with self-critique can solve any problem:** This approach enhances the model's problem-solving capabilities.

\fig

Specific Challenges for o1 Compared to Other Models:

- **How to Expand the Model's Solution Space?** In other words, how can the model be made to output longer and more sophisticated CoTs?
- **How to Determine the Quality of CoTs?**

##### Expanding the Solution Space

The model needs to broaden the conventional decoding space to obtain better long CoTs and results.

**Monte Carlo Tree Search (MCTS):**

MCTS is a strategy used in several recent works to expand the solution space and is the approach used by AlphaGo Zero.

**Searching CoT Paths in the Solution Space:**

By emulating AlphaZero, an MCTS tree is constructed, and each action is explored. Paths that have been explored are more likely to lead to correct results and are thus reinforced during training. During inference, these paths are more likely to be followed, yielding better results.

##### Is MCTS the Only Way?

While MCTS is used to expand the model's solution space, it's not the only method. Some argue against MCTS being the approach used in o1.

- **Proponents of MCTS** point out that many current papers involve LLMs combined with MCTS, and earlier work like FSBS shows elements of tree search.
- **Opponents** argue that since Noam (a key contributor to o1) doesn't specialize in MCTS but in Counterfactual Regret Minimization (CFR) and its variants, it's plausible that CFR methods are used instead. Moreover, MCTS excels in games like Go, while CFR is strong in poker; their effectiveness in LLMs remains uncertain.

Regardless of the method, the essence is to expand the model's solution space. As long as there's a path that leads to the correct answer, finding and reinforcing it is sufficient. This implies that the base model doesn't necessarily need to be extremely powerful.

> "Through training, the models learn to refine their thinking process, try different strategies, and recognize their mistakes." — OpenAI

##### Determining CoT Quality

It is believed that a Process Reward Model (PRM) is used. Some suggest using both Outcome Reward Model (ORM) and PRM, but this is debatable. Studies have shown that ORM is generally less effective than PRM, and ORM's training data can be considered a subset of PRM's data (PRM provides denser signals and higher data utilization). Therefore, it's unlikely that a subset would outperform the entire set.

If patch-level MCTS is indeed used, PRM (or its variants) is necessary to prune the tree and reduce the search space.

### Safety Alignment

A noteworthy aspect is o1's safety alignment strategy.

Compared to traditional RLHF (Reinforcement Learning from Human Feedback) for content safety, o1 likely adopts Anthropic's AI Constitution model for content safety. Combined with its strong reasoning abilities, o1 achieves enhanced safety.

\fig

### Inference Phase

How does o1 perform inference? It is speculated that o1 generates outputs token by token without using MCTS during inference. One reason is that o1 sometimes outputs incorrect guesses; if MCTS were used, incorrect nodes would be less likely. Additionally, experiments on the relationship between output token length and output latency show a linear trend, indicating that tokens are not being hidden, thus supporting a token-by-token output mechanism.

\fig

Given that o1, especially o1-mini, is priced at 20 times that of 4o-mini, it's possible that multiple models run in parallel online (though this doesn't contradict the conclusion that o1 is a single model). PRM might be used to select the best outputs from multiple samples (Best of N), with dynamic difficulty adjustment determining the value of N.

> "On the 2024 AIME exams, GPT-4o only solved on average 12% (1.8/15) of problems. o1 averaged 74% (11.1/15) with a single sample per problem, 83% (12.5/15) with consensus among 64 samples, and 93% (13.9/15) when re-ranking 1000 samples with a learned scoring function." — OpenAI

**Dynamic Compute Resource Selection:**

\fig

Since o1 outputs tokens sequentially, why is the model so expensive? The o1 series has higher costs not only for output but also for input. The input and output costs of o1-preview are 4 and 3 times that of 4o, respectively. For o1-mini, both input and output are 20 times that of 4o-mini.

The diversity of outputs is crucial for the feasibility of the inference-time scaling law. Without sufficient diversity, even the best PRM cannot select optimal outputs from multiple samples, failing to achieve the desired effect.

> "Ideally, test-time compute should modify the distribution so as to generate better outputs than naïvely sampling from the LLM itself would. In general, there are two knobs to induce modifications to an LLM’s distribution."

Even increasing the temperature in a single model doesn't guarantee diversity (e.g., in self-critique tasks, all critics often point out the same issue).

Therefore, the following conjectures are made:

- **Input Side:** Due to the use of an AI Constitution, the system prompt becomes longer. To increase the diversity of samples in Best of N, multiple prompts or even multiple models might be used to force diversity, resulting in non-shared KV caches and increased costs.
- **Output Side:** The increased cost is primarily due to multiple models (including summarization models). The motivation remains to enhance output diversity, as previously mentioned.

## 🎨 Case Show

We employed Monte Carlo Tree Search (MCTS) to construct a reasoning-based Chain-of-Thought (CoT) dataset. A classic example from this dataset is the question, "How many 'r's are in the word 'strawberry'?" Through multiple search steps, the correct answer was obtained.

\fig

## ⚡️ Install

The following instructions are for Linux installation.
We would like to recommend the requirements as follows.

* Python == 3.9.16
* CUDA Version >= 11.7

1. Clone this repository and navigate to the Uni-MoE folder

```bash
git clone https://github.com/AIDC-AI/Marco-o1.git
cd Marco-o1
```

2. Install Package

```Shell

```

## ⚡️ Available Models and Datasets

[Marco-o1](https://huggingface.co) (Our Reasoning Model)

[Marco Reasoning Dataset](https://huggingface.co) (Our Dataset)

## Experimental Results

| **Header 1** | **Header 2** | **Header 3** |
|--------------|--------------|--------------|
| | |  |
| | | |
| | | |

## 🌈 How to infer and deploy your demo

```bash
cd /path/to/Marco-o1
conda activate Marco
python demo/demo.py
```

## Citation

If you find Marco-o1 useful for your research and applications, please cite:

```
@misc{zhao-etal-2024-marco-o1,
author = {Yu Zhao, Huifeng Yin, Longyue Wang},
title = {Marco-o1},
year = {2024},
publisher = {GitHub},
journal = {GitHub repository},
howpublished = {\url{https://github.com/AIDC-AI/Marco-o1}}
}
```
