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

import hashlib
import json
import logging
import os
import time
from collections import defaultdict
from collections.abc import Collection, Iterable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from whyqa import metrics as metrics_
from whyqa.graph.common import (
    GPTClient,
    Metrics,
    OutputItem,
    ResultItem,
    calculate_similarity,
    load_dataset,
    parse_args,
    setup_logging,
    show_elapsed_time,
)

log = logging.getLogger(__file__)


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
    prompt = f"""Using the following causal graph, answer the question.
Make sure to consider the relationships in the graph when generating the answer.
The answer must be as concise as possible and should not contain any additional text.
Ensure that the answer only contains the relevant information and no additional context.

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
    setup_logging(log, log_level)

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
        "max_samples": max_samples,
        "seed": seed,
    }
    log.info(
        "Configuration:\n" + "\n".join(f"  {k}: {v}" for k, v in config.items()) + "\n"
    )

    dataset = dataset[:max_samples]
    output_items: list[OutputItem] = []

    start_time = time.perf_counter()

    for i, item in enumerate(tqdm(dataset), 1):
        log.debug(f"Item {i}/{len(dataset)}:")

        texts = item.texts[:max_texts]
        log.debug(f"  Building causal graphs. ({len(texts)} texts)")
        graphs = [build_causal_graph(client, item.id, text) for text in texts]
        log_graphs(graphs, texts)

        log.debug("  Merging similar nodes.")
        graphs = merge_nodes(client, item.id, graphs)

        log.debug("  Merging similar relations.")
        graphs = merge_relations(client, item.id, graphs)

        log.debug("  Combining causal graphs.")
        combined_graph = combine_graphs(client, item.id, graphs)
        log.debug(format_graph(combined_graph))

        log.debug("  Merging similar nodes in combined graph.")
        combined_graph = merge_nodes(client, item.id, [combined_graph])[0]

        log.debug("  Merging similar relations in combined graph.")
        combined_graph = merge_relations(client, item.id, [combined_graph])[0]
        log.debug(format_graph(combined_graph))

        log.debug("  Answering question.")
        predicted_answer = answer_question(client, item.id, combined_graph, item.query)

        output_item = OutputItem(
            id=item.id,
            query=item.query,
            texts=item.texts,
            expected_answer=item.answer,
            generated_answer=predicted_answer,
        )
        output_items.append(output_item)

        log.debug(f"Query: {item.query}")
        log.debug(f"Predicted Answer: {predicted_answer}")
        log.debug(f"Expected Answer: {item.answer}")

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
    avg_similarity = sum(result.metrics.cosine_similarity for result in results) / len(
        results
    )
    final_metrics = Metrics(
        cosine_similarity=avg_similarity,
        em=dataset_metrics.em,
        f1=dataset_metrics.f1,
        rouge_l_precision=dataset_metrics.rouge_l_precision,
        rouge_l_recall=dataset_metrics.rouge_l_recall,
        rouge_l_f1=dataset_metrics.rouge_l_f1,
    )
    log.info(final_metrics)

    log.info(f"Total Cost: ${client.calc_cost()}")

    elapsed_time_s = time.perf_counter() - start_time
    config["elapsed_time"] = elapsed_time_s

    log.info(f"Elapsed time: {show_elapsed_time(elapsed_time_s)}")

    output_dir = output_path / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "result.json").write_text(
        json.dumps([asdict(result) for result in results], indent=2)
    )
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    (output_dir / "metrics.json").write_text(
        json.dumps(asdict(final_metrics), indent=2)
    )

    log_serialisable = {
        item_id: [asdict(interaction) for interaction in interactions]
        for item_id, interactions in client.log.items()
    }
    (output_dir / "log.json").write_text(json.dumps(log_serialisable, indent=2))


if __name__ == "__main__":
    args = parse_args(__doc__)
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
