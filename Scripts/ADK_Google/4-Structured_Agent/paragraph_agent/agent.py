from google.adk.agents import LlmAgent
from pydantic import BaseModel, Field

class output_format(BaseModel):
    para: str = Field(
        description = "Write a short 50 word paragraph on the topic provided to you. It should be well formatted, and should use good language. "
    )

root_agent = LlmAgent(
    name= "paragraph_agent",
    model= "gemini-2.5-flash-lite",
    description = "Paragraph writing Agent",
    instruction = """
    You are an AI assistant whose job is to write a ~50 word paragraph on a topic which will be provided to you.
    General instructions:
    Make use of proper grammar and writing skills for these paragraphs.
    Do not exceed 65 words in any paragraph, and do not go below 45 words.
    
    IMPORTANT: You will output the result in a valid JSON format matching the structure given below
    {
        "para": "the short descriptive paragraph here"
    }

    DO NOT OUTPUT ANYTHING OTHER the specified JSON response.
    """,
    output_schema=output_format,
    output_key="para"
)