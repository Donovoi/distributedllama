# distributedllama

My musings with distribution

## Introduction

1. Start ray
ray start --head  --metrics-export-port=8080

2. Start ray on the other nodes:

`ray start --address='ADDRESS OF FIRST NODE'`

3. the following python will invoke the llm and share the load:
    1.

   ```Python
   from langchain_community.llms import Ollama
    
    import ray
    # here we will use ray and ollama to distribute the computation
    
    ray.init()

    # create an ollama object

    ollama = Ollama(
        base_url='<http://localhost:11434>',
        model="llama3",
        )

    print(ollama.invoke("Text Prompt Here"))

```
