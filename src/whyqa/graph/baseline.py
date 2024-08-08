# pyright: basic
"""Answer a WhyQA dataset using causal graphs and OpenAI API.

It loads a dataset from a JSON file, builds causal graphs for each item,
combines the graphs, summarizes nodes, generates answers using the OpenAI API,
and evaluates the answers using sentence embeddings.

This loads a dataset from a JSON file, and for each item in the dataset:
- Extracts causal relationships from each supporting text
- Combines the extracted relationships into a single causal graph
- Summarises the combined nodes in the graph
- Generates an answer to the question using the causal graph
- Evaluates the generated answer using cosine similarity with the expected answer using
sentence embeddings from SentenceTransformer

The dataset is a JSON file containg a list of items, each with the following keys:
- query (str): The question to answer.
- texts (list[str]): A list of texts to extract causal relationships from.
- answer (str): The expected answer to the question.

The output directory contains a subdirectory for each run, with the following files:
- result.json: A JSON file containing the results.
- config.json: A JSON file containing the configuration used for the run.

The `result.json` file contains a list of items, each with the following keys:
- query (str): The question.
- texts (list[str]): The list of texts.
- expected_answer (str): The expected answer.
- generated_answer (str): The generated answer.
- similarity_score (float): The similarity score between the expected and generated
answers.
"""

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import networkx as nx
import openai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class DatasetItem:
    query: str
    texts: Sequence[str]
    answer: str


@dataclass
class OutputItem:
    query: str
    texts: Sequence[str]
    expected_answer: str
    generated_answer: str
    similarity_score: float


def load_dataset(file_path: Path) -> list[DatasetItem]:
    """Load the dataset from a JSON file."""
    data: list[dict[str, Any]] = json.loads(file_path.read_text())
    return [
        DatasetItem(query=item["query"], texts=item["texts"], answer=item["answer"])
        for item in data
    ]


class GPTClient:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def call_openai_api(self, prompt: str) -> str:
        """Call the OpenAI API with the given prompt."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=150,
            )
            if result := response.choices[0].message.content:
                return result
            else:
                return ""
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return ""


def build_causal_graph(client: GPTClient, text: str) -> nx.DiGraph:
    """Build a causal graph from the given text using the OpenAI API."""
    prompt = f"""Extract causal relationships from the following text and represent them as a list of (cause, effect) pairs.
Use the format 'cause -> effect' for each relationship, with one relationship per line.
If there are no clear causal relationships, return an empty list.

Text:
{text}

Causal relationships:"""

    response = client.call_openai_api(prompt)

    graph = nx.DiGraph()
    for line in response.split("\n"):
        if "->" in line:
            cause, effect = map(str.strip, line.split("->"))
            graph.add_edge(cause, effect)

    return graph


def combine_graphs(client: GPTClient, graphs: Sequence[nx.DiGraph]) -> nx.DiGraph:
    """Combine multiple graphs into a single graph, de-duplicating nodes."""
    combined_graph = nx.DiGraph()
    node_mapping = {}

    for graph in graphs:
        for node in graph.nodes():
            if node not in node_mapping:
                if similar_nodes := [
                    n
                    for n in combined_graph.nodes()
                    if are_nodes_similar(client, node, n)
                ]:
                    # Create a new node that combines all similar nodes
                    combined_node = " / ".join([node, *similar_nodes])
                    for similar_node in similar_nodes:
                        node_mapping[similar_node] = combined_node
                    node_mapping[node] = combined_node

                    # Update the combined graph
                    combined_graph.add_node(combined_node)
                    for similar_node in similar_nodes:
                        combined_graph = nx.relabel_nodes(
                            combined_graph, {similar_node: combined_node}
                        )
                else:
                    node_mapping[node] = node
                    combined_graph.add_node(node)

        for edge in graph.edges():
            combined_graph.add_edge(node_mapping[edge[0]], node_mapping[edge[1]])

    return combined_graph


def summarize_graph(client: GPTClient, graph: nx.DiGraph) -> nx.DiGraph:
    """Summarize nodes in the graph, especially composite nodes."""
    summarized_graph = nx.DiGraph()

    for node in graph.nodes():
        if " / " in node:
            summary = summarize_nodes(client, node.split(" / "))
        else:
            summary = node

        summarized_graph.add_node(summary)
        for predecessor in graph.predecessors(node):
            summarized_graph.add_edge(predecessor, summary)
        for successor in graph.successors(node):
            summarized_graph.add_edge(summary, successor)

    return summarized_graph


def summarize_nodes(client: GPTClient, nodes: Sequence[str]) -> str:
    """Summarize a set of similar nodes into a single description."""
    prompt = f"""Summarize the following related events into a single, concise description:

Events:
{"\n".join(f"- {node}" for node in nodes)}

Summary:"""

    return client.call_openai_api(prompt)


def are_nodes_similar(client: GPTClient, node1: str, node2: str) -> bool:
    """Determine if two nodes are similar enough to be considered the same."""
    prompt = f"Are these two events essentially the same? Answer with 'Yes' or 'No':\n1. {node1}\n2. {node2}\nAnswer:"
    response = client.call_openai_api(prompt)
    return response.lower() == "yes"


def answer_question(client: GPTClient, graph: nx.DiGraph, question: str) -> str:
    """Generate an answer to the question using the causal graph."""
    graph_repr = "\n".join([f"{edge[0]} -> {edge[1]}" for edge in graph.edges()])
    prompt = f"""Using the following causal graph, answer the question:

Graph:
{graph_repr}

Question:
{question}

Answer:"""
    response = client.call_openai_api(prompt)

    # Parse the answer from the response
    answer_prefix = "Answer:"
    if answer_prefix in response:
        return response.split(answer_prefix, 1)[1].strip()
    return response.strip()


def calculate_similarity(text1: str, text2: str, model: SentenceTransformer) -> float:
    """Calculate the cosine similarity between two texts using sentence embeddings."""
    embeddings = model.encode([text1, text2])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return (similarity + 1) / 2  # Normalize to [0, 1]


def main(
    dataset_path: Path,
    api_key: str | None,
    senttf_model_name: str,
    gpt_model_name: str,
    output_path: Path,
    run_name: str | None,
) -> None:
    """Process the dataset and evaluate answers."""
    if api_key is None:
        api_key = os.environ["OPENAI_API_KEY"]
    if not run_name:
        run_name = f"{gpt_model_name}-{datetime.now(UTC).isoformat()}"

    client = GPTClient(api_key, gpt_model_name)
    dataset = load_dataset(dataset_path)
    senttf_model = SentenceTransformer(senttf_model_name)

    config = {
        "dataset_path": str(dataset_path),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "senttf_model": senttf_model_name,
        "gpt_model": gpt_model_name,
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
    }

    output_items: list[OutputItem] = []

    for i, item in enumerate(dataset, 1):
        graphs = [build_causal_graph(client, text) for text in item.texts]
        combined_graph = combine_graphs(client, graphs)
        summarized_graph = summarize_graph(client, combined_graph)
        predicted_answer = answer_question(client, summarized_graph, item.query)
        similarity = calculate_similarity(predicted_answer, item.answer, senttf_model)

        output_item = OutputItem(
            query=item.query,
            texts=item.texts,
            expected_answer=item.answer,
            generated_answer=predicted_answer,
            similarity_score=similarity,
        )
        output_items.append(output_item)

        print(f"Item {i}/{len(dataset)}:")
        print(f"Query: {item.query}")
        print(f"Predicted Answer: {predicted_answer}")
        print(f"Expected Answer: {item.answer}")
        print(f"Similarity Score: {similarity:.4f}\n")
        print()

    avg_similarity = sum(item.similarity_score for item in output_items) / len(
        output_items
    )
    print(f"Average Similarity Score: {avg_similarity:.4f}")

    output_dir = output_path / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "result.json").write_text(
        json.dumps([asdict(item) for item in output_items], indent=2)
    )
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "dataset_path", type=Path, help="Path to the JSON file containing the dataset"
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key. Defaults to the OPENAI_API_KEY environment variable",
    )
    parser.add_argument(
        "--senttf-model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name (default: %(default)s)",
    )
    parser.add_argument(
        "--gpt-model",
        default="gpt-4o-mini",
        help="OpenAI GPT model name (default: %(default)s)",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run name to store outputs (default: {GPT model}-{ISO timestamp})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="output",
        help="Path to output directory (default: %(default)s)",
    )
    args = parser.parse_args()

    main(
        args.dataset_path,
        args.api_key,
        args.senttf_model,
        args.gpt_model,
        args.output_dir,
        args.run_name,
    )
