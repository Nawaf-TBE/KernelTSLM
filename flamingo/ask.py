"""Ask a model questions about test Snapshots: a few prepared ones, then your own.

  python ask.py                       # the base model
  python ask.py ../runs/flamingo/model  # the fine-tuned model

At the prompt: type a question, ":class softmax" or ":path L1_P23/dt-bf16_s42" to switch the Snapshot, ":task" for the task question, or ":quit".
"""

import sys

import numpy as np

from finetune_kernelsight import kernelsight_examples
from utils import OpenTSLM, _autocast, chosen_option

PREPARED = [
    "What does the first counter do over time? Answer in one sentence.",
    "Is this GPU kernel limited by compute or by memory bandwidth? Explain briefly.",
    "Which of the counters has the highest values?",
]


def main():
    model = OpenTSLM.load(sys.argv[1] if len(sys.argv) > 1 else "hf://opentime/tslm-nov26").model
    model.eval()
    test = kernelsight_examples(dict(np.load("../kernelsight.npz")))["test"]
    by_class = {}
    for _, example in test:
        by_class.setdefault(example.answer, example)

    def ask(example, question):
        with _autocast(model):
            return model.generate(example.series, question, max_new_tokens=96)

    print("\n=== The task question, with the answer options ===")
    for name in ("matmul", "attention", "softmax", "pooling"):
        example = by_class[name]
        output = ask(example, example.evaluated_question())
        print(f"\n[{name}] model says: {output!r} -> counted as {chosen_option(output, example.options)}")

    example = by_class["matmul"]
    print("\n=== Free questions, on a matmul Snapshot ===")
    for question in PREPARED:
        print(f"\nQ: {question}\nA: {ask(example, question)}")

    print(f"\nYour turn. Classes: {', '.join(sorted(by_class))}. Now on: {example.answer}")
    while (line := input("\n> ").strip()) != ":quit":
        if line.startswith(":class "):
            example = by_class.get(line.split()[1], example)
            print(f"Now on: {example.answer}")
        elif line.startswith(":path "):
            found = [e for path, e in test if line.split()[1] in path]
            example = found[0] if found else example
            print(f"Now on: {example.answer} ({len(found)} matches)")
        elif line == ":task":
            output = ask(example, example.evaluated_question())
            print(f"{output!r} -> counted as {chosen_option(output, example.options)} (right: {example.answer})")
        elif line:
            print(ask(example, line))


if __name__ == "__main__":
    main()
