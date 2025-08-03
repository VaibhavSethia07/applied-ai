import os

import dotenv
from smolagents import (DuckDuckGoSearchTool, OpenAIServerModel,
                        PromptTemplates, ToolCallingAgent, VisitWebpageTool)

project_dir = os.path.join(os.path.dirname(__file__))
dotenv_path = os.path.join(project_dir, '.env')
dotenv.load_dotenv(dotenv_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")


tools = [DuckDuckGoSearchTool(max_results=5),  # internet search through a free search engine
         VisitWebpageTool(),  # reading text from a web page
         ]


model = OpenAIServerModel(model_id="openai/gpt-4o-mini", api_base=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
prompt_template = PromptTemplates(system_prompt="""
        You play the role of Web Searcher.
        You must answer the user's query by searching for answer on web.
        You have a DuckDuckGoSearchTool that searches the internet for web pages.
        You have a VisitWebpageTool that reads text from a web page.
        When you find the answer, return the answer
        """,
                                  final_answer="""
        Return the answer in the following format:
        <answer>
        """)

agent = ToolCallingAgent(tools=tools,
                         model=model,
                         prompt_templates=prompt_template,
                         max_steps=6  # limit against looping
                         )

query1 = "In what year did Cassius Clay change his name?"
output1 = agent.run(query1)

print("Executor rersult: ", output1)

query2 = "How much do 1000 tokens of YandexGPT Pro in Yandex Cloud cost for two modes?"
output2 = agent.run(query2)

print("Executor rersult: ", output2)

query3 = "Find the name of the team that took 3rd place in TI10. In which tournament did this team debut?"
output3 = agent.run(query3)

print("Executor rersult: ", output3)
