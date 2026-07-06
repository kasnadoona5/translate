"""Lightweight graph-based state machine for the Tarjomeh translation pipeline.

Provides a simple directed-graph executor that passes a mutable state dict
through a sequence of handler nodes, with support for conditional edges,
loops with max-iteration guards, error termination, and progress callbacks.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

class HandlerFn(Protocol):
    """Callable signature for a node handler."""

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]: ...


class ConditionFn(Protocol):
    """Callable signature for a conditional edge predicate."""

    def __call__(self, state: dict[str, Any]) -> bool: ...


class ProgressCallback(Protocol):
    """Optional callback invoked after each node execution."""

    def __call__(
        self,
        node_name: str,
        state: dict[str, Any],
        elapsed: float,
    ) -> None: ...


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Node:
    """A named processing step in the state machine."""

    name: str
    handler: HandlerFn
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Edge:
    """A directed edge between two nodes, optionally guarded by a condition.

    If *condition* is ``None`` the edge is unconditional.  When multiple
    edges leave the same source, the first whose condition returns ``True``
    is followed; an unconditional edge acts as a fallback.
    """

    source: str
    target: str
    condition: ConditionFn | None = None


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class StateMachine:
    """A lightweight graph executor.

    Nodes are callables that receive and return a state dict.  Edges define
    transitions between nodes and may carry guard conditions.

    Special behaviour:
    - **Loops**: Edges that revisit previous nodes are allowed.  The executor
      enforces a global ``max_iterations`` limit to prevent infinite loops.
    - **Error termination**: If a handler raises an exception on
      ``consecutive_failure_threshold`` consecutive invocations, the machine
      transitions to a ``paused_error`` terminal node (created automatically).
    - **Progress callbacks**: An optional callback is invoked after each
      successful node execution.
    - **Serialisable state**: The state dict is plain JSON-compatible data,
      suitable for checkpoint persistence.
    """

    # Reserved terminal node name for error bail-out
    PAUSED_ERROR_NODE = "paused_error"

    def __init__(
        self,
        *,
        max_iterations: int = 500,
        consecutive_failure_threshold: int = 3,
    ) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, list[Edge]] = {}  # source -> [edges]
        self._entry: str | None = None
        self._terminals: set[str] = set()
        self._max_iterations = max_iterations
        self._consecutive_failure_threshold = consecutive_failure_threshold
        self._progress_callback: ProgressCallback | None = None

    # -- Builder API --------------------------------------------------------

    def add_node(
        self,
        name: str,
        handler: HandlerFn,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a processing node.

        Parameters
        ----------
        name:
            Unique node identifier.
        handler:
            Callable ``(state) -> state``.
        metadata:
            Optional metadata dict attached to the node for introspection.
        """
        if name in self._nodes:
            raise ValueError(f"Duplicate node name: {name!r}")
        self._nodes[name] = Node(
            name=name,
            handler=handler,
            metadata=metadata or {},
        )

    def add_edge(
        self,
        source: str,
        target: str,
        condition: ConditionFn | None = None,
    ) -> None:
        """Add a directed edge from *source* to *target*.

        Parameters
        ----------
        source:
            Name of the source node.
        target:
            Name of the target node.
        condition:
            Optional predicate ``(state) -> bool``.  If ``None`` the edge is
            unconditional (used as fallback when no conditional edge matches).
        """
        edge = Edge(source=source, target=target, condition=condition)
        self._edges.setdefault(source, []).append(edge)

    def set_entry(self, name: str) -> None:
        """Designate the entry node of the graph."""
        self._entry = name

    def set_terminal(self, names: list[str]) -> None:
        """Designate one or more terminal (end) nodes."""
        self._terminals = set(names)

    def set_progress_callback(self, callback: ProgressCallback | None) -> None:
        """Attach (or detach) a progress callback."""
        self._progress_callback = callback

    # -- Execution ----------------------------------------------------------

    def execute(self, initial_state: dict[str, Any]) -> dict[str, Any]:
        """Run the state machine from the entry node to a terminal node.

        Parameters
        ----------
        initial_state:
            The starting state dict.  A shallow copy is made internally so the
            caller's original dict is not mutated.

        Returns
        -------
        dict[str, Any]
            The final state dict after the machine reaches a terminal node.

        Raises
        ------
        RuntimeError
            If the graph is misconfigured (no entry, missing nodes/edges) or
            the iteration limit is exceeded.
        """
        self._validate_graph()

        state = dict(initial_state)
        state.setdefault("_sm_iteration", 0)
        state.setdefault("_sm_consecutive_failures", 0)
        state.setdefault("_sm_history", [])
        state.setdefault("_sm_error", None)

        current = self._entry
        assert current is not None  # guaranteed by _validate_graph

        while current not in self._terminals:
            # Iteration guard
            state["_sm_iteration"] += 1
            if state["_sm_iteration"] > self._max_iterations:
                logger.error(
                    "State machine exceeded max iterations (%d) at node %r",
                    self._max_iterations,
                    current,
                )
                state["_sm_error"] = (
                    f"Max iterations ({self._max_iterations}) exceeded at node {current!r}"
                )
                return self._enter_paused_error(state)

            node = self._nodes[current]
            t0 = time.monotonic()

            try:
                state = node.handler(state)
                elapsed = time.monotonic() - t0

                # Reset consecutive failure counter on success
                state["_sm_consecutive_failures"] = 0
                state["_sm_history"].append(
                    {"node": current, "elapsed": round(elapsed, 4), "error": None}
                )

                logger.debug(
                    "Node %r completed in %.3fs", current, elapsed
                )

                if self._progress_callback is not None:
                    self._progress_callback(current, state, elapsed)

            except Exception as exc:
                elapsed = time.monotonic() - t0
                state["_sm_consecutive_failures"] += 1
                state["_sm_history"].append(
                    {"node": current, "elapsed": round(elapsed, 4), "error": str(exc)}
                )

                logger.warning(
                    "Node %r failed (%d/%d consecutive): %s",
                    current,
                    state["_sm_consecutive_failures"],
                    self._consecutive_failure_threshold,
                    exc,
                )

                if state["_sm_consecutive_failures"] >= self._consecutive_failure_threshold:
                    state["_sm_error"] = (
                        f"Consecutive failure threshold "
                        f"({self._consecutive_failure_threshold}) reached at node "
                        f"{current!r}: {exc}"
                    )
                    return self._enter_paused_error(state)

                # On a non-fatal error, stay on the same node and retry
                continue

            # Resolve the next node via edges
            current = self._resolve_next(current, state)

        # Terminal node reached — run its handler if it has one
        if current in self._nodes and current != self.PAUSED_ERROR_NODE:
            try:
                state = self._nodes[current].handler(state)
            except Exception as exc:
                logger.error("Terminal node %r handler failed: %s", current, exc)
                state["_sm_error"] = str(exc)

        state["_sm_terminal"] = current
        return state

    # -- Internal helpers ---------------------------------------------------

    def _resolve_next(self, current: str, state: dict[str, Any]) -> str:
        """Determine the next node by evaluating outgoing edges."""
        edges = self._edges.get(current, [])
        if not edges:
            raise RuntimeError(
                f"Node {current!r} has no outgoing edges and is not terminal"
            )

        fallback: str | None = None
        for edge in edges:
            if edge.condition is None:
                fallback = edge.target
            elif edge.condition(state):
                return edge.target

        if fallback is not None:
            return fallback

        raise RuntimeError(
            f"No edge condition matched at node {current!r} and no "
            f"unconditional fallback edge exists"
        )

    def _enter_paused_error(self, state: dict[str, Any]) -> dict[str, Any]:
        """Transition into the error-terminal node."""
        # Auto-create paused_error node if not explicitly added
        if self.PAUSED_ERROR_NODE not in self._nodes:
            self.add_node(
                self.PAUSED_ERROR_NODE,
                handler=lambda s: s,
                metadata={"auto_created": True},
            )
        self._terminals.add(self.PAUSED_ERROR_NODE)

        state["_sm_terminal"] = self.PAUSED_ERROR_NODE
        logger.error("State machine entering paused_error: %s", state.get("_sm_error"))
        return state

    def _validate_graph(self) -> None:
        """Check that the graph is well-formed before execution."""
        if self._entry is None:
            raise RuntimeError("No entry node set — call set_entry()")

        if not self._terminals:
            raise RuntimeError("No terminal nodes set — call set_terminal()")

        if self._entry not in self._nodes:
            raise RuntimeError(f"Entry node {self._entry!r} has no registered handler")

        # Verify all edge targets / sources reference known nodes
        all_node_names = set(self._nodes) | {self.PAUSED_ERROR_NODE}
        for source, edges in self._edges.items():
            if source not in all_node_names:
                raise RuntimeError(f"Edge source {source!r} is not a registered node")
            for edge in edges:
                if edge.target not in all_node_names:
                    raise RuntimeError(
                        f"Edge target {edge.target!r} (from {source!r}) "
                        f"is not a registered node"
                    )

    # -- Serialisation helpers ----------------------------------------------

    def get_checkpoint_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of the state dict suitable for JSON serialisation.

        Strips internal callables and non-serialisable objects, keeping only
        JSON-compatible data for checkpoint persistence.
        """
        checkpoint: dict[str, Any] = {}
        for key, value in state.items():
            if callable(value):
                continue
            try:
                # Quick serialisability test
                import json
                json.dumps(value)
                checkpoint[key] = copy.deepcopy(value)
            except (TypeError, ValueError, OverflowError):
                logger.debug("Skipping non-serialisable state key: %s", key)
        return checkpoint
