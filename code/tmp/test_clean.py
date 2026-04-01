import json
from pathlib import Path
from code.tmp.main import MultimodalCustomerAgent

def test():
    print("Running LLM test using SiliconFlow API key...")

    agent = MultimodalCustomerAgent()
    agent.build_knowledge_base()

    # Change question to one that likely has an image associated with it
    question = "VR头显怎么佩戴和调整？"
    print(f"Sending question to agent: {question}")
    res = agent.analyze_and_answer(question)

    print("\n\n--- LLM Answer ---")
    print(res["ret"])

    print("\n--- Retrieved Images ---")
    for p in res["image_paths"]:
        print(p)

    print("\n--- References ---")
    for r in res["references"]:
        print(r)

    # Save to file
    output_path = Path(__file__).resolve().parent / "test_result_vr_llm_new.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\nResult saved to: {output_path}")

if __name__ == "__main__":
    test()