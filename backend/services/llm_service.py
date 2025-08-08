import os
from dotenv import load_dotenv
from openai import OpenAI

class LLMService:

    def __init__(self, txt_file_path):
        self.model = self._load_model()
        self.txt_file_path = txt_file_path

    def _load_model(self):
        """
        Load the LLM model from environment variables.
        """
        print("Loading LLM model...")
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        print("API Key loaded:", api_key is not None)
        if not api_key:
            raise ValueError("API key for OpenAI is not set in environment variables.")
        
        return OpenAI(api_key=api_key)

    def _load_prompt_id(self):
        """
        Load the prompt from a txt file.
        """
        load_dotenv()
        prompt_id = os.getenv("PROMPT_ID")
        
        return prompt_id

    def _load_prompt(self):
        try:
            with open("./backend/llm/prompts/system.txt", "r") as file:
                system_prompt = file.read()
            return system_prompt

        except FileNotFoundError:
            raise FileNotFoundError("System prompt file not found. Please ensure the file exists at './llm/prompts/system.txt'.")

    def load_txt_data(self):
        """
        Load text data from the specified file path.
        """
        print("Loading text data from:", self.txt_file_path)
        if not os.path.exists(self.txt_file_path):
            raise FileNotFoundError(f"Text file not found at {self.txt_file_path}")
        
        with open(self.txt_file_path, 'r') as file:
            data = file.read()
        
        return data

    def generate_response(self):
        try: 
            print("Generating response using LLM...")
            prompt_id = self._load_prompt_id()
            print("Prompt ID loaded:", prompt_id)
            data_input = self.load_txt_data()
            print("Data input loaded:", type(data_input))
            system_prompt = self._load_prompt()
            print("System prompt loaded:", system_prompt)
            # Convert data_input to a txt string
            data_input_str = data_input
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
    txt_file_path = "./dummy.txt"     
    llm_service = LLMService(txt_file_path)
    response = llm_service.generate_response()
    if response:
        print("Generated Response:", response.output[0].content[0].text)
    else:
        print("Failed to generate response.")