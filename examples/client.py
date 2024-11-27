import requests

def chat_with_model(user_input, history):
    url = "http://localhost:8000/chat/"
    payload = {
        "user_input": user_input,
        "history": history
    }
    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to send request: {response.status_code} {response.text}")
        return None

def main():
    history = []
    print("Enter 'q' to quit, 'c' to clear chat history.")
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ['q', 'quit']:
            print("Exiting chat.")
            break
        if user_input.lower() == 'c':
            print("Clearing chat history.")
            history.clear()
            continue

        result = chat_with_model(user_input, history)
        if result:
            # Display the response from the model.
            print(f"Assistant: {result['response']}")
            # Update the chat history from the response.
            history = result['history']

if __name__ == "__main__":
    main()
