# Frequently Asked Questions (FAQ)

1. **Does it work for scanned PDF documents?**

    Yes, Vision Parse is specifically designed to handle scanned PDF documents effectively. It uses advanced Vision LLMs to extract text, tables, images, and LaTeX equations from both regular and scanned PDF documents with high precision.

2. **I am facing latency issues while running llama3.2-vision locally. How can I improve the performance of locally hosted vision models?**

    This is a known limitation with locally hosted Ollama models. Here are some solutions:

    - **Use API-based Models**: For better performance, consider using API-based models like OpenAI or Gemini, which are significantly faster and more accurate.
    - **Enable Concurrency**: Set `enable_concurrency` to `True` so that multiple pages are processed in parallel, thereby reducing latency. You can also increase the value of `OLLAMA_NUM_PARALLEL` to maximize the number of pages that can be processed in parallel.
    - **Disable Detailed Extraction**: Disable the `detailed_extraction` parameter for simpler PDF documents, which can improve latency.

3. **The llama3.2-vision:11b model was hallucinating and unable to extract content accurately from the PDF document. How can I improve the extraction accuracy of locally hosted vision models?**

    To improve extraction accuracy with the llama3.2-vision:11b model:
    
    - **Adjust Model Parameters**: Lower the `temperature` and `top_p` for more deterministic outputs and to reduce hallucinations.
    - **Define Custom Prompts**: By defining custom prompts according to your document structure, you can guide the model to extract content more accurately.
    - **Enable Detailed Extraction**: Enabling `detailed_extraction` will help the Vision LLM detect the presence of images, LaTeX equations, structured, and semi-structured tables, and then extract them with high accuracy.
    - **Consider Using Alternative Models**: Try API-based models like gpt-4o or gemini-1.5-pro for better accuracy and performance. Avoid using smaller models that are prone to hallucination.

4. **What are the recommended values for model parameters such as temperature, top_p, etc., to improve extraction accuracy?**

    Here are the recommended values for model parameters to improve extraction accuracy:
    - Set `temperature` to 0.7 and `top_p` to 0.5.
    - For Ollama models, increase `num_ctx` to 16384 and `num_predict` to 8092 (depending on the model size) and set `repeat_penalty` to 1.3.
    - For OpenAI models, increase `max_tokens` to 8192 (depending on the model size) and set `frequency_penalty` to 0.3.

    Note: The recommended values are generic and may need to be adjusted based on your document structure and the model's capabilities.
