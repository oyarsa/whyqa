# pyright: basic
"""Answer a WhyQA dataset using causal graphs and OpenAI API.

It loads a dataset from a JSON file, builds causal graphs for each item,
combines the graphs, summarises nodes, generates answers using the OpenAI API,
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
- log.json: A JSON file containing logs of all interactions with the OpenAI API.

The `result.json` file contains a list of items, each with the following keys:
- query (str): The question.
- texts (list[str]): The list of texts.
- expected_answer (str): The expected answer.
- generated_answer (str): The generated answer.
- similarity_score (float): The similarity score between the expected and generated
answers.

The `log.json` file contains an object where each key is an item ID and the value is
a list of objects, each containing:
- role (str): Either "user" or "assistant".
- data (str): The prompt sent to the API or the API response.
"""

import argparse
import copy
import hashlib
import json
import os
import sys
import warnings
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import dotenv
import networkx as nx
import openai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class DatasetItem:
    id: str
    query: str
    texts: Sequence[str]
    answer: str


@dataclass
class OutputItem:
    id: str
    query: str
    texts: Sequence[str]
    expected_answer: str
    generated_answer: str
    similarity_score: float


@dataclass
class APIInteraction:
    role: str
    data: str


def load_dataset(file_path: Path) -> list[DatasetItem]:
    """Load the dataset from a JSON file."""
    data: list[dict[str, Any]] = json.loads(file_path.read_text())
    return [
        DatasetItem(
            id=item["id"],
            query=item["query"],
            texts=item["texts"],
            answer=item["answer"],
        )
        for item in data
    ]


class GPTClient:
    def __init__(self, api_key: str, model: str, seed: int) -> None:
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model
        self._log: dict[str, list[APIInteraction]] = defaultdict(list)
        self._request_counter = 0
        self._seed = seed
        self._input_tokens = 0
        self._output_tokens = 0

    def call_openai_api(self, item_id: str, prompt: str) -> str:
        """Call the OpenAI API with the given prompt. Returns the response.

        If there was an error calling the API, logs the error and returns an empty
        string. Otherwise, returns the response string with leading and trailing
        whitespace removed.

        Logs the interaction per item id (user prompt and assistant result) and returns
        the result. The log can be obtained from the `log` attribute.
        """
        try:
            self._request_counter += 1
            print(f"\t\t\tCalling OpenAI API ({self._request_counter})")

            self._log[item_id].append(APIInteraction(role="user", data=prompt))
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                seed=self._seed,
            )
            result = response.choices[0].message.content or "<empty>"

            if response.usage:
                self._input_tokens += response.usage.prompt_tokens
                self._output_tokens += response.usage.completion_tokens
        except (openai.OpenAIError, IndexError) as e:
            print(f"Error calling OpenAI API: {e}")
            self._log[item_id].append(
                APIInteraction(role="assistant", data=f"Error: {e}")
            )
            return ""
        else:
            self._log[item_id].append(APIInteraction(role="assistant", data=result))
            return result.strip()

    @property
    def log(self) -> Mapping[str, Sequence[APIInteraction]]:
        return copy.deepcopy(self._log)

    def calc_cost(self) -> float:
        """Calculate the cost spent based on the number of tokens used."""
        model_pricing = {
            "gpt-3.5-turbo": (1, 2),
            "gpt-3.5-turbo-0125": (0.5, 1.5),
            "gpt-4": (30, 60),
            "gpt-4-0613": (30, 60),
            "gpt-4-turbo": (10, 30),
            "gpt-4-turbo-2024-04-09": (10, 30),
            "gpt-4o": (5, 15),
            "gpt-4o-2024-05-13": (5, 15),
            "gpt-4o-2024-08-06": (5, 15),
            "gpt-4o-mini-2024-07-18": (0.15, 0.60),
            "gpt-4o-mini": (0.15, 0.60),
        }

        if self._model not in model_pricing:
            print(
                f"WARNING: Pricing for model {self._model} is not available",
                file=sys.stderr,
            )
            return float("nan")

        input_, output = model_pricing[self._model]
        input_cost = (self._input_tokens / 1e6) * input_
        output_cost = (self._output_tokens / 1e6) * output

        return input_cost + output_cost


def remove_prefix(string: str, prefix: str) -> str:
    """Remove a prefix from a string if it exists.

    If the string starts with the prefix, remove the prefix and any leading whitespace
    after it. If the prefix is not found, return the original string unchanged.
    """
    if string.startswith(prefix):
        return string[len(prefix) :].lstrip()
    return string


def build_causal_graph(client: GPTClient, item_id: str, text: str) -> nx.DiGraph:
    """Build a causal graph from the given text using the OpenAI API."""
    prompt = f"""Extract causal relationships from the following text and represent them as a list of (cause, effect) pairs.
Use the format 'cause -> effect' for each relationship, with one relationship per line.
If there are no clear causal relationships, return nothing.
Each line must be only "node1 -> node2" without any additional text or formatting.
Only proper nouns should be capitalised; anything else should be lowercase.

Text:
{text}

Causal relationships:"""

    response = client.call_openai_api(item_id, prompt)
    response = remove_prefix(response, "Causal relationships:")
    return parse_graph(response)


def parse_graph(graph_str: str) -> nx.DiGraph:
    """Parse a string of causal relationships into a DiGraph.

    Args:
        graph_str: A string containing causal relationships, one per line.
            Each line should be in the format "cause -> effect".
            Empty lines and lines without "->" are ignored.
            Leading and trailing whitespace in cause and effect is stripped.

    Returns:
        A NetworkX DiGraph representing the causal relationships.

    Example input:
        "
        event A -> consequence B
        factor X -> outcome Y
        cause 1 -> effect 1
        cause 1 -> effect 2
        "

    Note:
        - The function is case-sensitive; "Event A" and "event A" are treated as
          different nodes.
        - If the same causal relationship appears multiple times in the input,
          only one edge will be created in the graph.
        - The function does not validate the semantic correctness of the relationships;
          it only parses the syntactic structure.
    """
    graph = nx.DiGraph()
    for line in graph_str.split("\n"):
        if "->" in line:
            cause, effect = map(str.strip, line.split("->"))
            graph.add_edge(cause, effect)
    return graph


def combine_graphs(
    client: GPTClient, item_id: str, graphs: Iterable[nx.DiGraph]
) -> nx.DiGraph:
    """Combine multiple graphs into a single graph using the LLM."""
    if not graphs:
        raise ValueError("At least one graph must be provided.")

    graph_representations: list[str] = []
    for i, graph in enumerate(graphs, 1):
        edges = (f"{src} -> {dst}" for src, dst in graph.edges())
        graph_representations.append(f"Graph {i}:\n" + "\n".join(edges))

    prompt = f"""Given the following causal graphs, combine them into a single coherent graph.
Merge similar nodes and remove redundancies. Present the result as a list of causal \
relationships in the format 'cause -> effect', one per line.
Each line must be only "node1 -> node2" without any additional text or formatting.

{"\n\n".join(graph_representations)}

Combined graph:"""

    response = client.call_openai_api(item_id, prompt)
    response = remove_prefix(response, "Combined graph:")
    return parse_graph(response)


def answer_question(
    client: GPTClient, item_id: str, graph: nx.DiGraph, question: str
) -> str:
    """Generate an answer to the question using the causal graph."""
    graph_repr = "\n".join([f"{edge[0]} -> {edge[1]}" for edge in graph.edges()])
    prompt = f"""Using the following causal graph, answer the question:

Graph:
{graph_repr}

Question:
{question}

Answer:"""
    response = client.call_openai_api(item_id, prompt)
    answer = remove_prefix(response, "Answer:")

    if not answer:
        print(
            f"WARNING: Failed to generate answer for question: {question}",
            file=sys.stderr,
        )
        answer = "Unable to generate answer"

    return answer


def calculate_similarity(text1: str, text2: str, model: SentenceTransformer) -> float:
    """Calculate the cosine similarity between two texts using sentence embeddings."""
    embeddings = model.encode([text1, text2])
    similarity = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
    return (similarity + 1) / 2  # Normalise to [0, 1]


def report_graphs(graphs: Sequence[nx.DiGraph]) -> None:
    """Print the number of graphs, nodes in each graph and edges in each graph."""
    print(f"  Number of graphs: {len(graphs)}")
    for i, graph in enumerate(graphs, 1):
        print(f"    Graph {i}:")
        print(f"      Nodes: {len(graph.nodes())}")  # type: ignore
        for node in graph.nodes():
            print(f"        {node}")
        print(f"      Edges: {len(graph.edges())}")
        for edge in graph.edges():
            print(f"        {edge[0]} -> {edge[1]}")
        print()
    print()


def main(
    dataset_path: Path,
    api_key: str | None,
    senttf_model_name: str,
    gpt_model_name: str,
    output_path: Path,
    run_name: str | None,
    max_texts: int | None,
    seed: int,
    max_samples: int | None,
) -> None:
    # Suppress useless warnings from transformers
    warnings.filterwarnings(
        "ignore", category=FutureWarning, module="transformers.tokenization_utils_base"
    )

    dotenv.load_dotenv()
    if api_key is None:
        api_key = os.environ["OPENAI_API_KEY"]
    if not run_name:
        run_name = f"{gpt_model_name}-{datetime.now(UTC).isoformat()}"

    client = GPTClient(api_key, gpt_model_name, seed)
    dataset = load_dataset(dataset_path)
    senttf_model = SentenceTransformer(senttf_model_name)

    # Save the configuration for reproducibility
    config = {
        "dataset_path": str(dataset_path),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "senttf_model": senttf_model_name,
        "gpt_model": gpt_model_name,
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
        "max_texts": max_texts,
        "seed": seed,
        "max_samples": max_samples,
    }

    dataset = dataset[:max_samples]
    output_items: list[OutputItem] = []

    for i, item in enumerate(dataset, 1):
        print(f"Item {i}/{len(dataset)}:")

        texts = item.texts[:max_texts]
        print(f"  Building causal graphs. ({len(texts)} texts)")
        graphs = [build_causal_graph(client, item.id, text) for text in texts]
        report_graphs(graphs)

        print("  Combining causal graphs.")
        combined_graph = combine_graphs(client, item.id, graphs)
        report_graphs([combined_graph])

        print("  Answering question.")
        predicted_answer = answer_question(client, item.id, combined_graph, item.query)

        print("  Calculating similarity score.")
        similarity = calculate_similarity(predicted_answer, item.answer, senttf_model)

        output_item = OutputItem(
            id=item.id,
            query=item.query,
            texts=item.texts,
            expected_answer=item.answer,
            generated_answer=predicted_answer,
            similarity_score=similarity,
        )
        output_items.append(output_item)

        print()
        print(f"Query: {item.query}")
        print(f"Predicted Answer: {predicted_answer}")
        print(f"Expected Answer: {item.answer}")
        print(f"Similarity Score: {similarity:.4f}\n")
        print()

    avg_similarity = sum(item.similarity_score for item in output_items) / len(
        output_items
    )
    print(f"Average Similarity Score: {avg_similarity:.4f}")
    print(f"Total Cost: ${client.calc_cost()}")

    output_dir = output_path / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "result.json").write_text(
        json.dumps([asdict(item) for item in output_items], indent=2)
    )
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    log_serialisable = {
        item_id: [asdict(interaction) for interaction in interactions]
        for item_id, interactions in client.log.items()
    }
    (output_dir / "log.json").write_text(json.dumps(log_serialisable, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "dataset_path", type=Path, help="Path to the JSON file containing the dataset"
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key. Defaults to the OPENAI_API_KEY env var (including .env)",
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
    parser.add_argument(
        "--max-texts",
        type=int,
        default=None,
        help="Maximum number of texts to process per item (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the OpenAI API (default: %(default)s)",
    )
    parser.add_argument(
        "--max-samples", "-n", type=int, default=None, help="Maximum number of samples"
    )
    args = parser.parse_args()

    main(
        args.dataset_path,
        args.api_key,
        args.senttf_model,
        args.gpt_model,
        args.output_dir,
        args.run_name,
        args.max_texts,
        args.seed,
        args.max_samples,
    )
