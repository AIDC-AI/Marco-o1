import httpx


def chat_with_model_stream(user_input, history, url="http://localhost:8000/chat/"):
    payload = {
        "user_input": user_input,
        "history": history
    }

    headers = {
        "Content-Type": "application/json"
    }

    # Use httpx to send a POST request without stream=True, handle streaming in response context
    with httpx.Client() as client:
        with client.stream("POST", url, json=payload, headers=headers) as response:
            if response.status_code == 200:
                print("Assistant:", end=" ")
                for chunk in response.iter_text():
                    print(chunk, end="", flush=True)
                print()
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

        chat_with_model_stream(user_input, history)
        # Future improvement: Update history based on API response if needed.


if __name__ == "__main__":
    main()

