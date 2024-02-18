from langchain_community.llms.llamacpp import LlamaCpp


def load_model(name, temp=0.3, max_tokens=512):
    llm = LlamaCpp(
        model_path='../models/' + name,
        temperature=temp,
        max_tokens=max_tokens,
        top_p=1,
        n_gpu_layers=-1,
        n_batch=512,
        n_ctx=2048,
        verbose=False
    )
    return llm
