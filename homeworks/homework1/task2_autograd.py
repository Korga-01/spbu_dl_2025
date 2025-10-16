from __future__ import annotations

from typing import Callable, Iterable, List, Set


class Node:
    def __init__(self, data: float, _children: Iterable[Node] = (), _op: str = ""):
        self.data: float = float(data)
        self.grad: float = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set[Node] = set(_children)
        self._op: str = _op

    def __repr__(self) -> str:
        return f"Element(data={self.data}, grad={self.grad})"

    # Addition
    def __add__(self, other: Node | float) -> Node:
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: float) -> Node:
        return self + other

    # Multiplication
    def __mul__(self, other: Node | float) -> Node:
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: float) -> Node:
        return self * other

    # ReLU activation
    def relu(self) -> Node:
        out = Node(self.data if self.data > 0.0 else 0.0, (self,), "relu")

        def _backward() -> None:
            self.grad += (1.0 if self.data > 0.0 else 0.0) * out.grad

        out._backward = _backward
        return out

    # Backward pass: topological order via DFS
    def backward(self) -> None:
        topo: List[Node] = []
        visited: Set[Node] = set()

        def build(v: Node) -> None:
            if v in visited:
                return
            visited.add(v)
            for child in v._prev:
                build(child)
            topo.append(v)

        build(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


if __name__ == "__main__":
    # Example from the statement
    a = Node(2)
    b = Node(-3)
    c = Node(10)
    d = a + b * c
    e = d.relu()
    e.backward()
    # Table for values and gradients of a, b, c, d, e
    header = f"{'Node':<6} | {'Value':>10} | {'Grad':>10}"
    sep = "-" * len(header)
    def fmt_row(name: str, n: Node) -> str:
        return f"{name:<6} | {n.data:>10.4f} | {n.grad:>10.4f}"
    print(sep)
    print(header)
    print(sep)
    print(fmt_row('a', a))
    print(fmt_row('b', b))
    print(fmt_row('c', c))
    print(fmt_row('d', d))
    print(fmt_row('e', e))
    print(sep)

    # Simple tests
    # 1) gradient of (x*y + x) at x=3, y=4 -> d/dx = y + 1 = 5, d/dy = x = 3
    x = Node(3)
    y = Node(4)
    z = x * y + x
    z.backward()
    # Second example table
    header2 = f"{'Expr':<10} | {'Value':>10} | {'Grad x':>10} | {'Grad y':>10}"
    sep2 = "-" * len(header2)
    print(sep2)
    print(header2)
    print(sep2)
    print(f"{'x*y + x':<10} | {z.data:>10.4f} | {x.grad:>10.4f} | {y.grad:>10.4f}")
    print(sep2)


