# llm-local-finetuning
Python scripts for fine-tuning Language Models using different techniques.

Consider a scenario where you have a pre-trained model like BERT, that was trained on vast amounts of general text data (like Wikipedia, Books, Papers, etc.). This pre-trained model has learned general patterns of language, but it’s not yet specialized for any particular task like classifying whether a specific legal document is forged or not, or whether a movie review is positive or negative.

Fine-tuning is the process of taking a pre-trained model (like any Large Language Model) and then adjusting its parameters on a smaller, task-specific dataset to optimize it for a particular task. Thus, adapting the existing model to a specific domain. By fine-tuning, we leverage the knowledge the model already has while tailoring it to the specific task at hand.

## Full Fine-Tuning

During a full fine-tuning process, all the parameters of the pre-trained model are adjusted based on the new dataset and task. This approach usually requires a large dataset and significant computational resources. It is useful when a high degree of specialization is needed. The model is prone to overfitting if the new dataset is small.

## Parameter Efficient Fine-Tuning (PEFT)

PEFT is a fine-tuning strategy to adjust only a subset of the model’s parameters, often just a few layers or specific components, instead of the entire model. This reduces the computational complexity and the memory usage. PEFT allows models to adapt to new tasks with fewer resources, making fine-tuning feasible even with limited infrastructure or smaller datasets.

PEFT techniques provide a wide range of methods for efficiently fine-tuning large models, allowing adaptation to specific tasks with limited resources. Techniques like LoRA and Adapters are particularly useful for handling very large models, while methods like Prompt Tuning and BitFit are lightweight options for fast adaptation. The choice of fine-tuning method depends on the task, available resources, budget, and timeline.

Some key PEFT techniques:

### Adapter Layers

Adapters are small Neural Networks inserted between layers of a pre-trained model. During fine-tuning only these adapter layers are trained, while the rest of the model weights remain frozen. This drastically reduces the number of trainable parameters during fine-tuning.

This approach was introduced in the paper **"Parameter-Efficient Transfer Learning for NLP"** by Houlsby et al., published in 2019. It proposes the use of adapter layers to make fine-tuning pre-trained language models more parameter-efficient by adding small task-specific adapter modules between the layers of a pre-trained model while freezing the original parameters.

Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., De Laroussilhe, Q., Gesmundo, A., Attariyan, M., & Gelly, S. (2019). **Parameter-Efficient Transfer Learning for NLP.**

Paper Link: https://arxiv.org/abs/1902.00751

### LoRA (Low-Rank Adaptation)

LoRA introduces low-rank matrices in specific layers of a pre-trained model. During fine-tuning only these matrices are trained, while the rest of the model weights remain frozen. It modifies specific layers of the transformer model by learning rank-decomposed updates.

This approach was introduced in the paper **"LoRA: Low-Rank Adaptation of Large Language Models"** by Edward J. Hu, Yelong Shen, et al., published in 2021. It proposes the use of low-rank matrices in the transformer layers of large pre-trained models, typically in the attention mechanisms. LoRA is particularly useful in fine-tuning large models efficiently with less memory and computational cost. This approach allows efficient fine-tuning of large models like GPT-3 and BERT.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., & Chen, W. (2021). **LoRA: Low-Rank Adaptation of Large Language Models**.

Paper Link: https://arxiv.org/abs/2106.09685

### Prefix-Tuning

Prefix-Tuning introduces a set of learnable tokens or prefixes to the model’s input sequence. During fine-tuning only these prefix parameters are updated, not the model itself. These prefixes act like task specific prompts that guide the model’s behaviour. For encoder-only models (e.g., BERT), prefixes are added to the beginning of the input sequence. For decoder-only models (e.g., GPT), prefixes are added to both the beginning of the input and the beginning of each layer's activations. For encoder-decoder models (e.g., T5), separate prefixes are added to the encoder and decoder.

This approach was introduced in the paper **"Prefix-Tuning: Optimizing Continuous Prompts for Generation"** by Xiang Lisa Li and Percy Liang, published in 2021. It proposes a method for fine-tuning pre-trained language models by learning task-specific continuous prefixes that are prepended to the input, while freezing rest of the model. The prefixes are typically initialized randomly and then optimized using gradient descent to minimize the task-specific loss. It is particularly efficient for text generation tasks and dialogue systems.

Li, X. L., & Liang, P. (2021). **Prefix-Tuning: Optimizing Continuous Prompts for Generation**.

Paper Link: https://arxiv.org/abs/2101.00190

### Prompt Tuning

Prompt Tuning is similar to Prefix-Tuning and it optimizes a small set of task-specific “soft prompts” or continuous token embeddings that are prepended to the input sequence. It is useful in case of zero-shot or few-shot learning, where the number of task-specific examples are limited.

This approach was introduced in the paper **"The Power of Scale for Parameter-Efficient Prompt Tuning"** by Brian Lester, Rami Al-Rfou, and Noah Constant, published in 2021. It proposes the concept of tuning only a small set of soft prompt embeddings instead of tuning the entire model, for adapting the model to downstream tasks. The effectiveness of Prompt Tuning increases with model size, often matching full fine-tuning performance for models with billions of parameters. The paper explores how prompt tuning can be scaled to work effectively with large pre-trained models like T5 for various NLP tasks.

Lester, B., Al-Rfou, R., & Constant, N. (2021). **The Power of Scale for Parameter-Efficient Prompt Tuning**.

Paper Link: https://arxiv.org/abs/2104.08691

### P-Tuning

P-Tuning extends Prompt Tuning by learning continuous prompt embeddings rather than discrete tokens. It extends Prompt Tuning by optimizing prompt embeddings that are placed between token sequences in a flexible way. These embeddings are closely tied to the model’s layers, and the approach interacts more deeply with the model’s internal structures, like with the attention heads. It allows for more flexible and better-performing task adaptation.

This approach was introduced in the paper **"GPT Understands, Too"** by Xiao Liu, Yanan Zheng, Zhengxiao Du, Ming Ding, Yujie Qian, Zhilin Yang, and Jie Tang, published in 2021. It proposes a method that optimizes continuous prompt embeddings for language models. P-Tuning is particularly effective for bi-directional models like BERT, where the flexible placement of prompts can leverage the model's full context. P-Tuning enables pre-trained models to adapt to downstream tasks efficiently, especially in few-shot learning scenarios.

Liu, X., Zheng, Y., Du, Z., Ding, M., Qian, Y., Yang, Z., & Tang, J. (2021). **GPT Understands, Too: Enhancing the Generalization of Pre-trained Models via Knowledge-Aware Optimization**.

Paper Link: https://arxiv.org/abs/2103.10385

### BitFit (Bias Fine-Tuning)

BitFit updates only the bias terms of a model’s layers during the fine-tuning process, while keeping the other model weights unchanged. As the number of bias terms are fewer in number, this technique significantly reduces the number of trainable parameters.

This approach was introduced in the paper **"BitFit: Simple Parameter-Efficient Fine-tuning for Transformer-Based Masked Language-Models"** by Yftah Ziser and Roi Reichart, published in 2021. It proposes a method where only bias terms of a pre-trained model are fine-tuned. BitFit leverages the hypothesis that bias terms can capture task-specific adjustments without altering the core learned representations in weight matrices. The effectiveness of BitFit can vary across different layers of the model, with some layers’ biases potentially being more critical for adaptation than others. This results in a highly efficient fine-tuning process with a very small number of trainable parameters.

Ziser, Y., & Reichart, R. (2021). **BitFit: Simple Parameter-Efficient Fine-tuning for Transformer-Based Masked Language-Models**.

Paper Link: https://arxiv.org/abs/2106.10199