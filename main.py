import os
from agent import SofiaAgent

agent = SofiaAgent("models/model-testing.joblib", "models/vectorizer.joblib")

major = None
major = input("Before starting, which is your major career? (default: none) > ")

while True:
    # Input question
    question = input("What is your question? > ")

    if question == "exit":
        print("Bye bye!")
        break
    if question == "clear" or question == "cls":
        os.system('cls')
        continue

    response, pred_label, preprocessed, prob = agent.ask(question, major)
    print(response)
    print(f"Label: {pred_label}, processed: {preprocessed}\n")