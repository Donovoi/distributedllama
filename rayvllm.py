import os
import subprocess
import ray
import asyncio
from langchain_community.llms import Ollama
from typing import Union, Sequence, List, Tuple, Dict, Any, Optional

# Define the async function to invoke the model


async def async_invoke_model(model_name, prompt_text):
    try:
        # Get the node IP address
        node_ip = ray._private.services.get_node_ip_address()

        # Check if the model exists in the ollama list
        modelexists = model_exists_in_ollama_list(model_name)
        if not modelexists:
            await download_and_pull_model_if_missing(model_name)

        # get the number of gpus and cpus across all nodes
        num_gpus = ray.cluster_resources()["GPU"]
        num_cpus = ray.cluster_resources()["CPU"]
        print(f"Number of GPUs: {num_gpus}, Number of CPUs: {num_cpus}")

        # Initialize the model
        llm = Ollama(model=model_name, keep_alive=-1, num_gpu=num_gpus, num_thread=0)

        result = await llm.ainvoke(prompt_text)

        # Return result along with node information
        return (node_ip, result)
    except Exception as e:
        print(f"Failed to invoke model in Ray. Error: {e}")
        return None

# Define the synchronous wrapper for the async function


def invoke_model_in_ray(model_name, prompt_text):
    return asyncio.run(async_invoke_model(model_name, prompt_text))


# Decorate the wrapper function with @ray.remote
invoke_model_in_ray = ray.remote(invoke_model_in_ray)


def model_exists_in_ollama_list(model_name):
    # Execute the `ollama list` command and capture the output
    result = subprocess.run(
        ["ollama", "list"], capture_output=True, text=True, check=True)
    if result.returncode == 0:
        # Check if the model name is in the output
        return model_name in result.stdout
    else:
        print(f"Failed to list models with ollama. Error: {result.stderr}")
        return False


async def download_and_pull_model_if_missing(model_name):
    if not model_exists_in_ollama_list(model_name):
        print(
            f"Model {model_name} not found in ollama list. Attempting to pull...")
        # Use the ollama pull command to download the model
        try:
            result = await asyncio.create_subprocess_exec(
                "ollama", "pull", model_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await result.communicate()
            if result.returncode == 0:
                print(f"Model {model_name} pulled successfully.")
            else:
                print(
                    f"Failed to pull model {model_name}. Error: {stderr.decode()}")
        except Exception as e:
            print(f"Failed to pull model {model_name}. Error: {e}")
    else:
        print(f"Model {model_name} found in ollama list.")


async def main():
    ray.shutdown()
    ray.init()

    # model_name = "llama3:latest"
    # prompt_text = "What is the capital of France?"

    model_name = "dolphin-mixtral:8x22b"
    prompt_text = "What is the capital of France?"

    # Launch multiple tasks
    tasks = [invoke_model_in_ray.remote(
        model_name, prompt_text) for _ in range(10)]

    # Collect and print results as tasks complete
    remaining_tasks = tasks
    while remaining_tasks:
        done, remaining_tasks = ray.wait(remaining_tasks, num_returns=1)
        for task in done:
            node_ip, result = ray.get(task)
            print(f"Node {node_ip} processed the task and returned: {result}")

if __name__ == "__main__":
    asyncio.run(main())
