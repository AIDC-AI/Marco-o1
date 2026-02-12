<p align="center">
    <img src="assets/logo_1.png" width="250" style="margin-bottom: 0.2;"/>
<p>

# 🍓 Marco-o1: Towards Efficient Reasoning Models

<!-- Broader Real-World Applications -->

<!-- # 🍓 Marco-o1: An Open Large Reasoning Model for Real-World Solutions -->

<!-- <h2 align="center"> <a href="https://github.com/AIDC-AI/Marco-o1/">Marco-o1</a></h2> -->
<!-- <h5 align="center"> If you appreciate our project, please consider giving us a star ⭐ on GitHub to stay updated with the latest developments.  </h2> -->
 
<h4 align="center">

<!-- [![🤗Hugging Face](https://img.shields.io/badge/🤗Hugging_Face-Marco_o1-yellow)](https://huggingface.co/) [![Project Page](https://img.shields.io/badge/Project_Page-Marco_o1-blue)](https://github.com/AIDC-AI/Marco-o1/) -->


<div align="center">
<img src="https://img.shields.io/badge/Version-2.0.0-blue.svg" alt="Version"> 
<img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License">
<img src="https://img.shields.io/github/stars/AIDC-AI/Marco-o1?color=yellow" alt="Stars">
<img src="https://img.shields.io/github/issues/AIDC-AI/Marco-o1?color=red" alt="Issues">
<img src="https://img.shields.io/badge/python-3.8-purple.svg" alt="Python">

</h4>

<div align="center">

<!-- **Affiliations:** -->

<!-- ⭐ _**MarcoPolo Team**_ ⭐ -->

⭐[_**Alibaba International Digital Commerce**_](https://aidc-ai.com)⭐

:octocat: [**Github**](https://github.com/AIDC-AI/Marco-o1)  🤗  [**Hugging Face**](https://huggingface.co/AIDC-AI/Marco-o1) 📝  [**Paper**](https://arxiv.org/abs/2411.14405) 🧑‍💻 [**Model**](https://huggingface.co/AIDC-AI/Marco-o1) 🗂️  [**Data**](https://github.com/AIDC-AI/Marco-o1/tree/main/data) 📽️  [**Demo**](https://huggingface.co/AIDC-AI/Marco-o1)

</div>

#

<div align="center">
  <img src="assets/timeline.png" alt="Figure Description or Alt Text" width="100%">
  <p><strong>Figure 1: </strong> Marco-o1 timeline.</p>
</div>

🎯 **Marco-o1** not only focuses on subjects with standard answers, such as mathematics, physics, and coding that are highly suitable for the use of Reinforcement Learning, but we also emphasize some open-ended solutions. Our goal is to build a general model applicable to agentic, incorporating comprehensive planning capabilities and function call abilities.

⚠️ **Limitations:** <ins>We would like to emphasize that this research work is inspired by OpenAI's o1 (from which the name is also derived). 
This work aims to explore potential approaches to shed light on the currently unclear technical roadmap for large reasoning models. 
Besides, our focus is on open-ended questions, and we have observed interesting phenomena in multilingual applications. 
However, we must acknowledge that the current model primarily exhibits o1-like reasoning characteristics and its performance still fall short of a fully realized "o1" model. 
This is not a one-time effort, and we remain committed to continuous optimization and ongoing improvement.</ins>





## 🔥 News

<!-- ## Coming Soon -->

<!-- This is our initial version, and we will continue to update and enhance the model's reasoning capabilities. -->

- [Coming Soon] 🏃 **Marco-o1 Agentic:** A more powerful agentic model is coming soon...

- [2025/02/09] 🔥 **[EDPO(Difficulty-Estimated Policy Optimization)](./src/DEPO/src):** We proposed an optimization algorithm based on an online data difficulty selector. To our knowledge, this is the first work on online data selection. Experiments show that compared with GRPO, we can better resist the noise interference caused by Zero Advantage, achieving an average performance improvement of 2.4%. At the same time, this online selector can also provide multi-scale routing based on prompt difficulty in large-scale online services.

- [2025/02/09] 🔥 Our paper **[A State-Transition Framework for Efficient LLM Reasoning](https://arxiv.org/abs/2602.01198)** has been accepted to the main conference of ICLR 2026.

- [2025/02/09] 🔥 **Marco-o1 v3:** We released **[Marco-o1 v3](./src/v3/src)**. By training a pluggable Linear component MAM (Mixed Attention Module) on the existing Dense model, we were able to dynamically compress the model to save context tokens. At the same time, we introduced TTT (Test-Time Training), and ultimately we achieved a 20% reduction in inference cost while obtaining an average performance improvement of 4.7%.

- [2025/05/15] 🔥 Our paper **[《Marco-o1 v2: Towards Widening The Distillation Bottleneck for Reasoning Models》](https://arxiv.org/abs/2503.01461)** has been accepted to the main conference of ACL 2025.


- [2025/02/14] 🔥 We released **[Marco-o1 v2](./README_v2.md)**. This version entirely relies on self-built data and has undergone DPO. It has been optimized more comprehensively for mathematical problem-solving、planning and instruction-following capabilities. 🍬 This time, our model's ability in counting letters is quite impressive! 😁


- [2024/11/13] 🔥 We released **[Marco-o1 v1](./README_v1.md)**: Towards Open Reasoning Models for Open-Ended Solutions. This initial release includes our reasoning model, optimized for complex problem-solving and versatile applications across various domains.



# ⚡️ Released Resources

## Codes and Models

📥 [Marco-o1 v1](https://huggingface.co/AIDC-AI/Marco-o1)

📥 [Marco-o1 v2](https://huggingface.co/AIDC-AI/Marco-o1)

💻 [Marco-o1 v3](./src/v3/src)

💻 [Marco-o1 DEPO](./src/DEPO/src)

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

## Usage

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
    ./src/output/talk_with_model.py

    # Use vLLM
    ./src/output/talk_with_model_vllm.py
    ```
3. **Deploy using FastAPI:**

    Check the README.md file in examples folder.


# 👨🏻‍💻 Acknowledgement

## Main Contributors
From MarcoPolo Team, AI Business, Alibaba International Digital Commerce:
- [Yu Zhao](https://github.com/Sniper970119)
- [Huifeng Yin](https://github.com/HuifengYin)
- [Liang Zhang](https://scholar.google.com/citations?user=MSCCJiMAAAAJ&hl=zh-CN)
- [Longyue Wang](http://www.longyuewang.com)


## Citation

If you find Marco-o1 useful for your research and applications, please cite:

```
@misc{zhao2024marcoo1openreasoningmodels,
      title={Marco-o1: Towards Open Reasoning Models for Open-Ended Solutions}, 
      author={Yu Zhao and Huifeng Yin and Bo Zeng and Hao Wang and Tianqi Shi and Chenyang Lyu and Longyue Wang and Weihua Luo and Kaifu Zhang},
      year={2024},
      eprint={2411.14405},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.14405}, 
}

@misc{yin2025wideningdistillationbottleneckreasoning,
      title={Marco o1 v2:Towards Widening The Distillation Bottleneck for Reasoning Models}, 
      author={Huifeng Yin and Yu Zhao and Minghao Wu and Xuanfan Ni and Bo Zeng and Hao Wang and Tianqi Shi and Liangying Shao and Chenyang Lyu and Longyue Wang and Weihua Luo and Kaifu Zhang},
      year={2025},
      eprint={2503.01461},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.01461}, 
}

@misc{zhang2026statetransitionframeworkefficientllm,
      title={A State-Transition Framework for Efficient LLM Reasoning}, 
      author={Liang Zhang and Yu Zhao and Longyue Wang and Tianqi Shi and Weihua Luo and Kaifu Zhang and Jinsong Su},
      year={2026},
      eprint={2602.01198},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.01198}, 
}

@misc{zhao2026difficultyestimatedpolicyoptimization,
      title={Difficulty-Estimated Policy Optimization}, 
      author={Yu Zhao and Fan Jiang and Tianle Liu and Bo Zeng and Yu Liu and Longyue Wang and Weihua Luo},
      year={2026},
      eprint={2602.06375},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.06375}, 
}
```

## LICENSE

This project is licensed under [Apache License Version 2](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) (SPDX-License-identifier: Apache-2.0).

## DISCLAIMER

We used compliance checking algorithms during the training process, to ensure the compliance of the trained model and dataset to the best of our ability. Due to complex data and the diversity of language model usage scenarios, we cannot guarantee that the model is completely free of copyright issues or improper content. If you believe anything infringes on your rights or generates improper content, please contact us, and we will promptly address the matter.
