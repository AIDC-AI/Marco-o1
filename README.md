<p align="center">
    <img src="assets/logo.png" width="150" style="margin-bottom: 0.2;"/>
<p>

# 🍓 Marco-o1: An Open Large Reasoning Model for Real-World Solutions

<!-- <h2 align="center"> <a href="https://github.com/AIDC-AI/Marco-o1/">Marco-o1</a></h2> -->
<!-- <h5 align="center"> If you appreciate our project, please consider giving us a star ⭐ on GitHub to stay updated with the latest developments.  </h2> -->
 
<h4 align="center">

<!-- [![🤗Hugging Face](https://img.shields.io/badge/🤗Hugging_Face-Marco_o1-yellow)](https://huggingface.co/) [![Project Page](https://img.shields.io/badge/Project_Page-Marco_o1-blue)](https://github.com/AIDC-AI/Marco-o1/) -->


<div align="center">
<img src="https://img.shields.io/badge/Version-1.0.0-blue.svg" alt="Version"> 
<img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License">
<img src="https://img.shields.io/github/stars/AIDC-AI/Marco-o1?color=yellow" alt="Stars">
<img src="https://img.shields.io/github/issues/AIDC-AI/Marco-o1?color=red" alt="Issues">
<img src="https://img.shields.io/badge/python-3.8-purple.svg" alt="Python">

</h4>

<div align="center">

<!-- **Affiliations:** -->

⭐ _**MarcoPolo Team**_ ⭐

[_**AI Business, Alibaba International Digital Commerce**_](https://aidc-ai.com)

<!-- Yu Zhao, [Huifeng Yin](https://github.com/HuifengYin), [Longyue Wang](http://www.longyuewang.com/) -->

:octocat: [**Github**]()  🤗  [**Hugging Face**]() 📝  [**Paper**]() :technologist:  [**Model**]() 🗂️  [**Data**]() 📽️  [**Demo**]()

</div>

#

<div align="center">

🎯 **Marco-o1** Large Language Model (LLM) powered by _Chain-of-Thought (CoT) fine-tuning_, _Monte Carlo Tree Search (MCTS)_, _reflection mechanisms_, and _innovative reasoning strategies_—optimized for complex real-world problem-solving tasks. 

</div>

<div align="center">
  <img src="assets/strawberry_2.jpg" alt="Figure Description or Alt Text" width="80%">
  <p><strong>Figure 1: </strong> A classic 'strawberry' question reasoned by our Marco-o1 model: "How many 'r' are in strawberry"</p>
</div>

## 🚀 Highlights

Our main contributions are threefold: 
- 🍀**Fine-Tuning with CoT Data:** We develop <span style="color:blue">Marco-o1-CoT</span> by performing full-parameter fine-tuning on the base model using open-source CoT dataset combined with our self-developed synthetic data. 
- 🍀**Solution Space Expansion via MCTS:** We integrate LLMs with MCTS (_Marco-o1-MCTS_), using the model's output confidence to guide the search and expand the solution space. 
- 🍀**Reasoning Action Strategy:** We implement novel reasoning action strategies and a reflection mechanism (_Marco-o1-MCTS Mini-Step_), including exploring different action granularities within the MCTS framework and prompting the model to self-reflect, thereby significantly enhancing the model's ability to solve complex problems.
- 🍀**Application in Translation Tasks:** We are the first to apply Large Reasoning Models (LRM) to _Machine Translation task_, exploring inference time scaling laws in the multilingual and translation domain.

## 🔥 News

<!-- ## Coming Soon -->

<!-- This is our initial version, and we will continue to update and enhance the model's reasoning capabilities. -->

1. [Coming Soon] **Training Reward Models:** We are working on training reward models, including Outcome Reward Modeling (ORM) and Policy Reward Modeling (PRM), to provide a more accurate reward signal for MCTS. A more precise reward function will help reduce randomness in tree search results and improve overall performance.

2. [Coming Soon] **Reinforcement Learning Training:** We are conducting reinforcement learning training to develop better reasoning models. By utilizing RL techniques, we aim to refine the model's decision-making processes and further enhance its problem-solving abilities.

- [2024/11/13] 🔥 We released **Marco-o1**. This initial release includes our reasoning model, optimized for complex problem-solving and versatile applications across various domains.



## Introduction

OpenAI recently unveiled the revolutionary o1 model, which, with its exceptional reasoning capabilities, has excelled on platforms like AIME, CodeForces, and LeetCode Pro Max, surpassing other leading models. Inspired by this achievement and given our Marco team's specialization in multilingual and translation, we recognized the inherent challenges in translation—such as the need to accurately handle difficult terms, including internet slang, colloquial expressions, specialized terminology, and emerging new words. To address these complexities, we have replicated and extended the o1 model's technical roadmap with a stronger focus on multilingual and translation tasks, while also considering general reasoning capabilities. We are proud to introduce Marco-o1, which further enhances the model's reasoning performance through innovative methodologies designed to meet the unique demands of translation tasks.

<div align="center">
  <img src="assets/intro.jpg" alt="Figure Description or Alt Text" width="80%">
  <p><strong>Figure 1: </strong>Overview of the Marco-o1</p>
</div>

Our main contributions are threefold: 
- **Fine-Tuning with CoT Data:** We developed Marco-o1-CoT by performing full-parameter fine-tuning on the base model using open-source Chain-of-Thought (CoT) data combined with our self-developed synthetic data. 
- **Solution Space Expansion via MCTS:** We integrated Large Language Models (LLMs) with Monte Carlo Tree Search (MCTS), using the model's output confidence to guide the search and expand the solution space. 
- **Reasoning Action Strategy:** We implemented novel reasoning action strategies and a reflection mechanism, including exploring different action granularities within the MCTS framework and prompting the model to self-reflect, thereby significantly enhancing the model's ability to solve complex problems.
- **Application in Translation Tasks:** We are the first to apply Large Reasoning Models(LRM) to translation tasks, exploring inference time scaling laws in the multilingual and translation domain.


## 🎨 Case Show

This is a classic example from our Marco-o1 model, "How many 'r's are in the word 'strawberry'?" Through multiple search steps, the correct answer was obtained, as shown in Figure 2. Although we tested general reasoning capabilities, our primary focus is on tackling challenging translation problems. An example of this focus is shown in Figure 3, illustrating the translation of the sentence "This shoe has a comfortable sole and is highly recommended for purchase."

<div align="center">
  <img src="assets/strawberry_2.jpg" alt="Figure Description or Alt Text" width="80%">
  <p><strong>Figure 2: </strong>Reasoning example of "How many 'r' are in strawberry"</p>
</div>

<div align="center">
  <img src="assets/translation.jpg" alt="Figure Description or Alt Text" width="80%">
  <p><strong>Figure 3: </strong>Translation example of "This shoe has a comfortable sole and is highly recommended for purchase."</p>
</div>


## Datasets

To enhance the reasoning capabilities of the Marco-o1 model, we employed a supervised fine-tuning strategy using a variety of datasets.

**CoT Data:** We generated the Marco CoT Dataset using MCTS, which helped to formulate complex reasoning pathways. Additionally, we improved the Open-O1 project's CoT Dataset by applying a heuristic and quality filtering process, allowing the model to adopt structured reasoning patterns effectively.

**Other Data:** Recognizing the critical role of robust instruction-following capabilities in executing complex tasks, we used a set of instruction-following data. Furthermore, to ensure that the model retains some specific knowledge, we integrated some other SFT datasets. This combination ensures the model remains competent across a wide range of tasks, maintaining its general effectiveness while significantly boosting its reasoning flair.

## Solution Space Expansion via MCTS

We integrated LLMs with MCTS to enhance the reasoning capabilities of our model. Here's how the combination works:

- **Nodes as Reasoning States:** In the MCTS framework, each node represents a reasoning state of the problem-solving process.
- **Actions as LLM Outputs:** The possible actions from a node are the outputs generated by the LLM. These outputs represent potential steps or mini-steps in the reasoning chain.
- **Rollout and Reward Calculation:** During the rollout phase, the LLM continues the reasoning process to a terminal state. We obtain the value of this state by computing a confidence score using the following formulas:
1. **Confidence Score ($c_i$):**

   For each token $t_i$ generated during the rollout, we calculate its confidence score by applying the softmax function to its log probability and the log probabilities of the top 5 alternative tokens. This is given by:

   $$c_i = \frac{\exp(p(t_i))}{\sum_{k=1}^{5} \exp(p(t_k))}$$

   **Where:**

    - $c_i$ is the confidence score for the $i^{th}$ token in the rollout.
    - $p(t_i)$ is the log probability of the $i^{th}$ token generated by the LLM.
    - $p(t_k)$ for $k = 1$ to $5$ are the log probabilities of the top 5 predicted tokens at the $i^{th}$ step.
    - $n$ is the total number of tokens in the rollout sequence.

   This equation ensures that the confidence score reflects the relative probability of the chosen token compared to the top alternatives, effectively normalizing the scores between 0 and 1.
  
2. **Reward Score ($v$):**

   After obtaining the confidence scores for all tokens in the rollout sequence, we compute the average confidence score across all tokens to derive the overall reward score:

   $$v = \frac{1}{n} \sum_{i=1}^{n} c_i$$

   **Where:**

   - $v$ is the overall reward score for the rollout path.

   This average serves as the reward signal that evaluates the quality of the reasoning path taken during the rollout. A higher $v$ indicates a more confident and likely accurate reasoning path.

- **Guiding MCTS:** This reward score $R$ is used to evaluate and select promising paths within the MCTS, effectively guiding the search towards more confident and reliable reasoning chains.

By employing this method, we effectively expand the solution space, allowing the model to explore a vast array of reasoning paths and select the most probable ones based on calculated confidence scores.


## Reasoning Action Strategy

### Action Selection

To enhance our model's reasoning capabilities and effectively expand its solution space, we explored different granularity levels for actions within the Monte Carlo Tree Search (MCTS) framework. Specifically, we considered three approaches:

1. **Step as Action:** Each action corresponds to an entire thought or reasoning step.

2. **Token as Action:** At the other end of the spectrum, each action is a single token generated by the model.

3. **Mini-Step as Action:** Serving as a middle ground between the two, we introduced "Mini-Step" sequences of tokens (e.g., 32 or 64 tokens) as actions.

In our experiments, we employed both the **Step as Action** and **Mini-Step as Action** strategies within the MCTS framework:

- **Step as Action:** We allowed the model to generate complete steps as actions, enabling it to take significant reasoning steps during the search process.
- **Mini-Step as Action:** We used Mini-Step of 32 or 64 tokens as actions, allowing the model to explore reasoning paths at a finer granularity than whole steps but coarser than individual tokens.

By integrating these strategies, we expanded the solution space and improved the model's ability to navigate complex reasoning tasks. This combination allowed us to leverage the strengths of both approaches—efficient search and detailed exploration—leading to better performance in problem-solving scenarios.

### Reflection after Thinking

We introduced a reflection mechanism by adding the phrase **"Wait! Maybe I made some mistakes! I need to rethink from scratch."** at the end of each thought process. This prompts the model to self-reflect and reevaluate its reasoning steps. Implementing this reflection has yielded significant improvements, especially on difficult problems that the original model initially solved incorrectly. With the addition of reflection, approximately half of these challenging problems were answered correctly.

From the self-critic perspective, this approach allows the model to act as its own critic, identifying potential errors in its reasoning. By explicitly prompting the model to question its initial conclusions, we encourage it to re-express and refine its thought process. This self-critical mechanism leverages the model's capacity to detect inconsistencies or mistakes in its own output, leading to more accurate and reliable problem-solving. The reflection step serves as an internal feedback loop, enhancing the model's ability to self-correct without external intervention.

## Experimental Results

Based on Qwen2-7B-Instruct, we performed supervised fine-tuning using CoT data to create Marco-o1-CoT. Subsequently, we employed Marco-o1-CoT within the framework of MCTS tree search, differentiating by actions:

- **Using each inference step as an action (Step).**
- **Using a 64-token Mini-Step as an action (64 Tokens).**
- **Using a 32-token Mini-Step as an action (32 Tokens).**

During testing, each model utilized a CoT prompt to ensure consistency in reasoning processes.

We then tested these configurations on the English and Chinese subsets of the MGSM dataset, obtaining the following results:

| **Model**                | **MGSM-en (Acc.)** | **MGSM-zh (Acc.)** |
|--------------------------|--------------------|--------------------|
| Qwen2-7B-Instruct        | 84.23%             | 76.80%             |
| Marco-o1-CoT             | 85.60%             | 71.20%             |
| Marco-o1-MCTS (Step)     | 90.40%             | 80.00%             |
| Marco-o1-MCTS (Mini-Step of 64 Tokens) | 88.40%             | 80.40%             |
| Marco-o1-MCTS (Mini-Step of 32 Tokens) | 87.60%             | 82.40%             |

These results demonstrate:

1. Performance of Marco-o1-CoT vs. Qwen2-7B-Instruct:

   - In the MGSM-en dataset, Marco-o1-CoT shows an advantage over Qwen2-7B-Instruct, as shown in Figure 4, which is expected due to the fine-tuning with English CoT data.
   - In the MGSM-zh dataset, however, Marco-o1-CoT exhibits a decrease in performance compared to Qwen2-7B-Instruct. This decline is attributed to the fact that the CoT data used for fine-tuning was in English, which may not transfer effectively to the Chinese dataset.

2. Impact of MCTS Integration:

   - The three MCTS-enhanced models demonstrate improvements over Marco-o1-CoT, indicating that incorporating MCTS helps to expand the model's solution space and increase the probability of obtaining correct answers.
   - However, since we use the Confidence Score as the reward, the tree search results exhibit significant randomness. In MGSM-en, the "Step as Action" strategy performs the best, while in MGSM-zh, the "Mini-Step as Action (32)" strategy yields the highest accuracy.
   - Currently, as shown in Figure 5-6, we cannot draw definitive conclusions about which action strategy is superior. We believe that as the reward becomes more accurate, the larger solution space provided by MCTS will demonstrate greater potential.

<div align="center">
  <img src="assets/cot_step.jpg" alt="Figure Description or Alt Text" width="100%">
  <p><strong>Figure 4: </strong>Marco-o1-CoT got it wrong (left), Marco-o1-MCTS (Step) got it right (right) in the MGSM dataset</p>
</div>

<div align="center">
  <img src="assets/step_patch.jpg" alt="Figure Description or Alt Text" width="100%">
  <p><strong>Figure 5: </strong>Marco-o1-MCTS (Step) got it wrong (left), Marco-o1-MCTS (Mini-Step of 32 Tokens) got it right (right) in the MGSM dataset</p>
</div>

<div align="center">
  <img src="assets/patch_step.jpg" alt="Figure Description or Alt Text" width="100%">
  <p><strong>Figure 6: </strong>Marco-o1-MCTS (Mini-Step of 64 Tokens) got it wrong (left), Marco-o1-MCTS (Step) got it right (right) in the MGSM dataset</p>
</div>

These results demonstrate the effectiveness of our approach in enhancing the reasoning capabilities of the model across different languages and configurations.


## ⚡️ Available Models and Datasets

[Marco-o1](https://huggingface.co/AIDC-AI/Marco-o1) (Our Lastest Model)

[Marco Reasoning Dataset](https://github.com/AIDC-AI/Marco-o1/blob/main/data/CoT_demo.json) (Our Partial Dataset)

## Installation

To install Marco-o1, follow these steps:

```bash
# Clone the repository
git clone https://github.com/AIDC-AI/Marco-o1

# Change to the Macaw-LLM directory
cd Marco-o1

# Install required packages
pip install -r requirements.txt

```

## Usage 🚀

1. **Load Marco-o1-CoT model:** 
    ```
    # Load model directly
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("AIDC-AI/Marco-o1")
    model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Marco-o1")
    ```

2. **Inference:** 

    Execute the inference script (you can give any customized inputs inside):
    ```
    ./src/talk_with_model.py

    # Use vLLM
    ./src/talk_with_model_vllm.py

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


## LICENSE

This project is licensed under [Apache License Version 2](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) (SPDX-License-identifier: Apache-2.0).

## DISCLAIMER

We used compliance checking algorithms during the training process, to ensure the compliance of the trained model and dataset to the best of our ability. Due to complex data and the diversity of language model usage scenarios, we cannot guarantee that the model is completely free of copyright issues or improper content. If you believe anything infringes on your rights or generates improper content, please contact us, and we will promptly address the matter.
