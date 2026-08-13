from pathlib import Path
import json
import ollama


# ============================================================
# URBANWATCH — LLAMA AI ASSISTANT
# ============================================================

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

YOLO_RESULTS = (
    RESULTS
    / "yolo_evaluation"
    / "evaluation_summary.json"
)

LOVEDA_RESULTS = (
    RESULTS
    / "loveda_evaluation_resnet18"
    / "evaluation_results.json"
)

LEVIR_RESULTS = (
    RESULTS
    / "change_detection"
    / "test_evaluation"
    / "test_results.json"
)

FINAL_REPORT = (
    RESULTS
    / "urbanwatch_final_report.json"
)


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):

    if not path.exists():
        return {
            "status": "not found",
            "path": str(path)
        }

    try:
        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:
            return json.load(f)

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


# ============================================================
# PROJECT CONTEXT
# ============================================================

def build_context():

    return {
        "project": "UrbanWatch",

        "description": (
            "UrbanWatch is a remote sensing AI pipeline "
            "for satellite imagery analysis."
        ),

        "models": {
            "SpaceNet 2":
                "YOLO building detection",

            "LoveDA":
                "DeepLabV3 with ResNet18 "
                "semantic segmentation",

            "LEVIR-CD+":
                "Siamese deep learning "
                "change detection"
        },

        "verified_results": {

            "SpaceNet 2": load_json(
                YOLO_RESULTS
            ),

            "LoveDA": load_json(
                LOVEDA_RESULTS
            ),

            "LEVIR-CD+": load_json(
                LEVIR_RESULTS
            ),

            "Final report": load_json(
                FINAL_REPORT
            )
        },

        "known_metrics": {

            "SpaceNet 2": {
                "Precision": 0.8274,
                "Recall": 0.7516,
                "mAP@50": 0.8212,
                "mAP@50-95": 0.5004
            },

            "LoveDA": {
                "mIoU": 0.5402,
                "Dice/F1": 0.6847,
                "Pixel Accuracy": 0.6880
            },

            "LEVIR-CD+": {
                "Precision": 0.7070,
                "Recall": 0.7310,
                "Dice/F1": 0.7188,
                "IoU": 0.5610,
                "Pixel Accuracy": 0.9764
            }
        }
    }


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are the UrbanWatch AI Analyst.

UrbanWatch is a remote sensing AI project containing:

1. SpaceNet 2 building detection
2. LoveDA semantic segmentation
3. LEVIR-CD+ change detection

Answer questions about the UrbanWatch project using only
the supplied project context.

Rules:

- Do not invent experimental results.
- Use the reported metrics exactly.
- Explain metrics accurately.
- Distinguish mAP, IoU, Dice/F1, precision,
  recall and pixel accuracy.
- Pixel accuracy can be misleading when
  background pixels dominate.
- If information is unavailable, say so.
- Keep answers technically accurate but understandable.
"""


# ============================================================
# LLAMA QUERY
# ============================================================

def ask_llama(
    question,
    context
):

    prompt = f"""
{SYSTEM_PROMPT}

============================================================
URBANWATCH PROJECT CONTEXT
============================================================

{json.dumps(
    context,
    indent=2,
    default=str
)}

============================================================
USER QUESTION
============================================================

{question}

============================================================
ANSWER
============================================================
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response[
        "message"
    ][
        "content"
    ]


# ============================================================
# MAIN CHAT
# ============================================================

def main():

    print("=" * 70)
    print("URBANWATCH — LLAMA AI ASSISTANT")
    print("=" * 70)

    print()
    print(
        "Loading UrbanWatch project context..."
    )

    context = build_context()

    print(
        "Context loaded successfully."
    )

    print()
    print(
        "Ask questions about the UrbanWatch "
        "models, results, datasets or pipeline."
    )

    print(
        "Type 'exit' to quit."
    )

    print()

    while True:

        question = input(
            "You: "
        ).strip()

        if question.lower() in {
            "exit",
            "quit"
        }:

            print(
                "UrbanWatch AI: Goodbye."
            )

            break

        if not question:
            continue

        try:

            print()
            print(
                "UrbanWatch AI:"
            )

            answer = ask_llama(
                question,
                context
            )

            print(
                answer
            )

            print()

        except Exception as e:

            print()
            print(
                "ERROR:"
            )

            print(
                str(e)
            )

            print()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
    