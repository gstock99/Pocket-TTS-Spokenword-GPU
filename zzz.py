from huggingface_hub import HfApi
api = HfApi()

try:
    # This will raise an error if the token is invalid
    user_info = api.whoami(token="hf_YOUR_TOKEN_HERE")
    print("Success! Your token is valid and belongs to:", user_info['name'])
except Exception as e:
    print("Invalid token or authentication failed:", e)