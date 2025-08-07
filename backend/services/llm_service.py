import os
from dotenv import load_dotenv
from openai import OpenAI
import json

class LLMService:

    def __init__(self, json_file_path):
        self.model = self._load_model()
        self.json_file_path = json_file_path

    def _load_model(self):
        """
        Load the LLM model from environment variables.
        """
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key for OpenAI is not set in environment variables.")
        
        return OpenAI(api_key=api_key)

    def _load_prompt_id(self):
        """
        Load the prompt from a JSON file.
        """
        load_dotenv()
        prompt_id = os.getenv("PROMPT_ID")
        
        return prompt_id

    def _load_prompt(self):
        try:
            with open("./llm/prompts/system.txt", "r") as file:
                system_prompt = file.read()
            return system_prompt

        except FileNotFoundError:
            raise FileNotFoundError("System prompt file not found. Please ensure the file exists at './llm/prompts/system.txt'.")

    def load_json_data(self):
        """
        Load JSON data from the specified file path.
        """
        if not os.path.exists(self.json_file_path):
            raise FileNotFoundError(f"JSON file not found at {self.json_file_path}")
        
        with open(self.json_file_path, 'r') as file:
            data = json.load(file)
        
        return data

    def generate_response(self):
        try: 
            prompt_id = self._load_prompt_id()
            data_input = self.load_json_data()
            system_prompt = self._load_prompt()
            # Convert data_input to a JSON string
            data_input_str = json.dumps(data_input, indent=2)
            # print("Data input loaded:", type(data_input_str))
            response = self.model.responses.create(
                model="gpt-4o-2024-08-06",
                instructions=system_prompt,
                # prompt = prompt_id,
                input = data_input_str
            )
            # print("Response received from LLM:", response)
            return  response.output[0].content[0].text

        except Exception as e:
            print(f"An error occurred while generating response: {e}")
            return None

if __name__ == "__main__":
    json_file_path = "./dummy.json"
    llm_service = LLMService(json_file_path)
    response = llm_service.generate_response()
    if response:
        print("Generated Response:", response.output[0].content[0].text)
    else:
        print("Failed to generate response.")