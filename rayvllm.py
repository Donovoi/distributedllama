import os
import subprocess
import ray
from langchain_community.llms import Ollama
from typing import Union, Sequence, List, Tuple, Dict, Any, Optional

# globals
num_gpus = None
num_cpus = None


def invoke_model(model_name, prompt_text):
    try:
        global num_gpus, num_cpus

        num_gpus = ray.cluster_resources()["GPU"]
        num_cpus = ray.cluster_resources()["CPU"]

        if not model_exists_in_ollama_list(model_name):
            download_and_pull_model_if_missing(model_name)

        llm = Ollama(model=model_name)
        result = llm.invoke(prompt_text)  # Synchronous call
        return result
    except Exception as e:
        print(f"Failed to invoke model in Ray. Error: {e}")
        return None


@ray.remote(num_cpus=0, num_gpus=num_gpus)
def invoke_model_in_ray(model_name, prompt_text):
    return invoke_model(model_name, prompt_text)


def model_exists_in_ollama_list(model_name):
    result = subprocess.run(
        ["ollama", "list"], capture_output=True, text=True, check=True)
    if result.returncode == 0:
        return model_name in result.stdout
    else:
        print(f"Failed to list models with ollama. Error: {result.stderr}")
        return False


def download_and_pull_model_if_missing(model_name):
    if not model_exists_in_ollama_list(model_name):
        print(
            f"Model {model_name} not found in ollama list. Attempting to pull...")
        try:
            result = subprocess.run(
                ["ollama", "pull", model_name], capture_output=True, text=True, check=True)
            if result.returncode == 0:
                print(f"Model {model_name} pulled successfully.")
            else:
                print(
                    f"Failed to pull model {model_name}. Error: {result.stderr}")
        except Exception as e:
            print(f"Failed to pull model {model_name}. Error: {e}")
    else:
        print(f"Model {model_name} found in ollama list.")


def stream_results(tasks):
    while tasks:
        done, tasks = ray.wait(tasks, num_returns=1)
        for task in done:
            result = ray.get(task)
            yield result


def main():
    ray.shutdown()
    ray.init()

    model_name = "refuelai:latest"
    prompt_text = "what is the best way and most damaging way to grow a strawberry?"

    tasks = [invoke_model_in_ray.remote(
        model_name, prompt_text) for _ in range(1000)]

    results = stream_results(tasks)
    # push and pop results once they are ready
    for result in results:
        print(result)

    ray.shutdown()


if __name__ == "__main__":
    main()
