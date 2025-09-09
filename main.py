from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent,AgentExecutor
from tools import search_tool,wiki_tool
import re

load_dotenv()
# llm= ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")  # or another Gemini model

class ResearchResponse(BaseModel):
    summary: str
    topic:str
    source: list[str]
    tools_used: list[str]
parser= PydanticOutputParser(pydantic_object=ResearchResponse)

prompt= ChatPromptTemplate.from_messages(
    [
       (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())
tools=[search_tool,wiki_tool]
agent=create_tool_calling_agent(llm=llm,prompt=prompt,tools=tools)
agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)
query=input("How can i help you in the research: ")
response = agent_executor.invoke({"query": query})
print(response)
try:
    output_str = response.get("output")
    # Remove code block markers if present
    json_str = re.sub(r"^```json|```$", "", output_str, flags=re.MULTILINE).strip()
    structured_response = parser.parse(json_str)
except Exception as e:
    print(f"Error parsing response: {e}")
    structured_response = None
print(structured_response)