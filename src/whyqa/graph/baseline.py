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

The dataset is a JSON file containing a list of items, each with the following keys:
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
import logging
import os
import sys
import warnings
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import dotenv
import openai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from whyqa import metrics as metrics_

log = logging.getLogger(__file__)


@dataclass(frozen=True)
class DatasetItem:
    id: str
    query: str
    texts: Sequence[str]
    answer: str


@dataclass(frozen=True)
class OutputItem:
    id: str
    query: str
    texts: Sequence[str]
    expected_answer: str
    generated_answer: str


@dataclass(frozen=True)
class Metrics:
    cosine_similarity: float
    em: float
    f1: float
    rouge_l_precision: float
    rouge_l_recall: float
    rouge_l_f1: float


@dataclass(frozen=True)
class ResultItem:
    output: OutputItem
    metrics: Metrics


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
        self._seed = seed
        self._input_tokens = 0
        self._output_tokens = 0

    def run(self, item_id: str, prompt: str) -> str:
        """Call the OpenAI API with the given prompt. Returns the response text.

        If the request is successful, returns the response string with leading and
        trailing whitespace removed. If there was an error calling the API, logs the
        error and returns an empty string.

        Logs the interaction per item id (both the user prompt and assistant result)
        on both successful and failed API calls. The log can be obtained from the `log`
        property.
        """
        try:
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
            log.exception("Error calling OpenAI API")
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
            log.warning(f"Pricing for model {self._model} is not available")
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


class Edge(NamedTuple):
    source: str
    relation: str
    destination: str


class Graph:
    """A directed graph with nodes and labelled edges.

    There's only one edge per unique relationship. Note that there may be multiple
    edges with the same nodes but different relation labels.
    """

    edges: Collection[Edge]
    nodes: Collection[str]

    def __init__(self, edges: Iterable[tuple[str, str, str] | Edge]) -> None:
        """Constructs a graph with the given edges. Nodes are derived from the edges.

        Edges containig self-loops and cycles with the same relation are removed.

        Args:
            edges: A sequence of tuples in the format (source, relation, destination).
        """
        self.edges = tuple(
            self._simplify_edges(Edge(src, rel, dst) for src, rel, dst in edges)
        )
        nodes: list[str] = []
        for src, _, dst in self.edges:
            nodes.extend((src, dst))
        self.nodes = frozenset(nodes)

    @classmethod
    def _simplify_edges(cls, edges: Iterable[Edge]) -> Iterable[Edge]:
        """Simplify the edges by removing self-loops and cycles with the same relation.

        Considers only self-loops like "(A, R, A)" and "(A, R, B), (B, R, A)" as cycles.
        I.e. cycles must have the same relation label or both sides.
        """
        cleaned_edges: set[Edge] = set()

        for edge in edges:
            if edge.source == edge.destination:
                continue

            reverse_key = Edge(edge.destination, edge.relation, edge.source)
            if reverse_key in cleaned_edges:
                continue

            cleaned_edges.add(edge)

        return cleaned_edges


def parse_graph(graph_str: str) -> Graph:
    """Parse a sequence of causal relationships into a `Graph`.

    Args:
        graph_str: A string containing causal relationships, one per line.
            Each line should be in the format "cause -> relation -> effect".
            Empty lines, lines without "->" or with less than 3 components are ignored.
            Leading and trailing whitespace in cause and effect is stripped.

    Returns:
        A `Graph` representing the causal relationships.

    Example input:
        "
        event A -> relation 1 -> consequence B
        factor X -> relation 2 -> outcome Y
        cause 1 -> relation 3 -> effect 1
        cause 1 -> relation 4 -> effect 2
        "

    Note:
        - The function is case-sensitive; "Event A" and "event A" are treated as
          different nodes. Similarly for relation labels.
        - The graph only contains one edge per unique causal relationship. Note that
          there may be multiple edges with the same nodes but different relation labels.
    """
    edges: list[tuple[str, str, str]] = []
    for line in graph_str.split("\n"):
        if "->" in line:
            parts = line.split("->", maxsplit=2)
            if len(parts) != 3:
                continue

            cause, relation, effect = map(str.strip, parts)
            edges.append((cause, relation, effect))
    return Graph(edges)


GRAPH_FORMAT_PROMPT = """\
The graph should be represented using the following format:
- Represent the relations as a sequence of (cause, relation, effect) triplets.
- Use the format 'cause -> relation -> effect' for each relationship, with one relationship per line.
- If there are no clear causal relationships, return nothing.
- Each line must be only "cause -> relation -> effect" without any additional text or formatting.
- Only proper nouns should be capitalised; anything else should be lowercase.
"""


def build_causal_graph(client: GPTClient, item_id: str, text: str) -> Graph:
    """Build a causal graph from the given text using the OpenAI API."""
    prompt = f"""Extract causal relationships from the following text.
{GRAPH_FORMAT_PROMPT}

Text:
{text}

Causal relationships:"""

    response = client.run(item_id, prompt)
    response = remove_prefix(response, "Causal relationships:")
    return parse_graph(response)


def combine_graphs(client: GPTClient, item_id: str, graphs: Iterable[Graph]) -> Graph:
    """Combine multiple graphs into a single graph using the LLM."""
    if not graphs:
        raise ValueError("At least one graph must be provided.")

    graph_representations: list[str] = []
    for i, graph in enumerate(graphs, 1):
        edges = (f"{src} -> {rel} -> {dst}" for src, rel, dst in graph.edges)
        graph_representations.append(f"Graph {i}:\n" + "\n".join(edges))

    prompt = f"""Given the following causal graphs, combine them into a single coherent graph.
Merge similar nodes, remove redundancies and merge edges with the same nodes and
similar relations.
{GRAPH_FORMAT_PROMPT}

{"\n\n".join(graph_representations)}

Combined graph:"""

    response = client.run(item_id, prompt)
    response = remove_prefix(response, "Combined graph:")
    return parse_graph(response)


def answer_question(
    client: GPTClient, item_id: str, graph: Graph, question: str
) -> str:
    """Generate an answer to the question using the causal graph."""
    graph_repr = "\n".join(f"{src} {rel} {dst}" for src, rel, dst in graph.edges)
    prompt = f"""Using the following causal graph, answer the question. Be succint in \
your response.

Graph:
{graph_repr}

Question:
{question}

Answer:"""
    response = client.run(item_id, prompt)
    answer = remove_prefix(response, "Answer:")

    if not answer:
        log.warning(f"Failed to generate answer for question: {question}")
        answer = "Unable to generate answer"

    return answer


def format_graph(graph: Graph, indent_size: int = 0) -> str:
    """Format the nodes and edges of a graph as a string."""
    indent = " " * indent_size
    lines = [f"{indent}Nodes: {len(graph.nodes)}"]
    lines.extend(f"{indent}  {node}" for node in graph.nodes)
    lines.append(f"{indent}Edges: {len(graph.edges)}")
    lines.extend(f"{indent}  {src} -> {rel} -> {dst}" for src, rel, dst in graph.edges)
    return "\n" + "\n".join(lines) + "\n"


def log_graphs(graphs: Sequence[Graph], texts: Sequence[str]) -> None:
    # sourcery skip: merge-list-appends-into-extend
    """Log the number of graphs, nodes in each graph, and edges in each graph."""
    lines = [f"\n  Number of graphs: {len(graphs)}"]
    for i, (graph, text) in enumerate(zip(graphs, texts), 1):
        lines.append(f"    Graph {i}:")
        lines.append(f"      Text: {text}")
        lines.append(format_graph(graph, 6))
    log.debug("\n".join(lines))


def calculate_similarity(text1: str, text2: str, model: SentenceTransformer) -> float:
    """Calculate the cosine similarity between two texts using sentence embeddings."""
    embeddings = model.encode([text1, text2])
    similarity = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
    return (similarity + 1) / 2  # Normalise to [0, 1]


def parse_mapping(mapping_str: str) -> dict[str, str]:
    """Parse a mapping of "old -> new" values from a string, one per line."""
    mapping: dict[str, str] = {}
    for line in mapping_str.splitlines():
        if "->" in line:
            parts = line.split("->", maxsplit=1)
            if len(parts) != 2:
                continue

            old, new = map(str.strip, parts)
            mapping[old] = new
    return mapping


def log_mapping(mapping: dict[str, str]) -> None:
    """Log a mapping of old values to new values. Reverses the mapping for clarity."""
    reverse_mapping: dict[str, list[str]] = defaultdict(list)
    for old, new in mapping.items():
        reverse_mapping[new].append(old)

    lines = [
        f"    {" / ".join(olds)} -> {new}" for new, olds in reverse_mapping.items()
    ]
    log.debug("\n" + "\n".join(lines) + "\n")


def merge_items(
    client: GPTClient, item_id: str, items: Iterable[str], name: str
) -> dict[str, str]:
    """Merge items with similar meaning."""
    prompt = f"""Given the following list of {name}s, one per line, merge {name}s with \
similar meaning. Respond with the list of combined {name}s. The response format should \
only contain the merged items and no other text or formatting. The items should be in the \
be 'old {name} -> new {name}' for each {name} that has been merged. Only include \
{name} that have been merged.

{name.capitalize()}s:
{"\n".join(items)}

Merged {name}s:"""
    response = client.run(item_id, prompt)
    response = remove_prefix(response, f"Merged {name}s:")
    return parse_mapping(response)


def merge_relations(
    client: GPTClient, item_id: str, graphs: Sequence[Graph]
) -> Sequence[Graph]:
    """Merge nodes with similar meaning in the graphs."""
    relations = {relation for graph in graphs for _, relation, _ in graph.edges}
    relation_mapping = merge_items(client, item_id, relations, "relation")
    log_mapping(relation_mapping)

    merged_graphs: list[Graph] = []

    for graph in graphs:
        new_edges: list[Edge] = []
        for src, relation, dst in graph.edges:
            new_relation = relation_mapping.get(relation, relation)
            new_edges.append(Edge(src, new_relation, dst))
        merged_graphs.append(Graph(new_edges))

    return merged_graphs


def merge_nodes(
    client: GPTClient, item_id: str, graphs: Sequence[Graph]
) -> Sequence[Graph]:
    """Merge nodes with similar meaning in the graphs."""
    nodes = {node for graph in graphs for node in graph.nodes}
    node_mapping = merge_items(client, item_id, nodes, "node")
    log_mapping(node_mapping)

    merged_graphs: list[Graph] = []

    for graph in graphs:
        new_edges: list[Edge] = []
        for src, rel, dst in graph.edges:
            new_src = node_mapping.get(src, src)
            new_dst = node_mapping.get(dst, dst)
            new_edges.append(Edge(new_src, rel, new_dst))
        merged_graphs.append(Graph(new_edges))

    return merged_graphs


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
    log_level: str,
) -> None:
    # Set up logger
    if log_level.upper() not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raise ValueError(f"Invalid log level: {log_level}")
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.setLevel(log_level.upper())

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
        log.info(f"Item {i}/{len(dataset)}:")

        texts = item.texts[:max_texts]
        log.info(f"  Building causal graphs. ({len(texts)} texts)")
        graphs = [build_causal_graph(client, item.id, text) for text in texts]
        log_graphs(graphs, texts)

        log.info("  Merging similar nodes.")
        graphs = merge_nodes(client, item.id, graphs)

        log.info("  Merging similar relations.")
        graphs = merge_relations(client, item.id, graphs)

        log.info("  Combining causal graphs.")
        combined_graph = combine_graphs(client, item.id, graphs)
        log.debug(format_graph(combined_graph))

        log.info("  Merging similar nodes in combined graph.")
        combined_graph = merge_nodes(client, item.id, [combined_graph])[0]

        log.info("  Merging similar relations in combined graph.")
        combined_graph = merge_relations(client, item.id, [combined_graph])[0]
        log.debug(format_graph(combined_graph))

        log.info("  Answering question.")
        predicted_answer = answer_question(client, item.id, combined_graph, item.query)

        log.info("  Calculating similarity score.")
        similarity = calculate_similarity(predicted_answer, item.answer, senttf_model)

        output_item = OutputItem(
            id=item.id,
            query=item.query,
            texts=item.texts,
            expected_answer=item.answer,
            generated_answer=predicted_answer,
        )
        output_items.append(output_item)

        log.info(f"Query: {item.query}")
        log.info(f"Predicted Answer: {predicted_answer}")
        log.info(f"Expected Answer: {item.answer}")
        log.info(f"Similarity Score: {similarity:.4f}\n")

    results: list[ResultItem] = []
    for output in output_items:
        cosine = calculate_similarity(
            output.expected_answer, output.generated_answer, senttf_model
        )
        metrics = metrics_.calculate_sentence(
            output.expected_answer, output.generated_answer
        )
        results.append(
            ResultItem(
                output,
                Metrics(
                    cosine_similarity=cosine,
                    em=metrics.em,
                    f1=metrics.f1,
                    rouge_l_precision=metrics.rouge_l_precision,
                    rouge_l_recall=metrics.rouge_l_recall,
                    rouge_l_f1=metrics.rouge_l_f1,
                ),
            )
        )

    dataset_metrics = metrics_.calculate_dataset([
        metrics_.Instance(output.expected_answer, output.generated_answer)
        for output in output_items
    ])
    log.info(dataset_metrics)

    avg_similarity = sum(result.metrics.cosine_similarity for result in results) / len(
        results
    )
    log.info(f"Average Similarity Score: {avg_similarity:.4f}")
    log.info(f"Total Cost: ${client.calc_cost()}")

    output_dir = output_path / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "result.json").write_text(
        json.dumps([asdict(result) for result in results], indent=2)
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
        "-k",
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
    parser.add_argument(
        "--log-level",
        default="info",
        help="Logging level (default: %(default)s)",
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
        args.log_level,
    )
