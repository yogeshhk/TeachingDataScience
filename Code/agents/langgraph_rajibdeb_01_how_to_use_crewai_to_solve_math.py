import os

from crewai import Agent, Task, Crew
from dotenv import load_dotenv
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain_community.tools.wolfram_alpha import WolframAlphaQueryRun

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
WOLFRAM_ALPHA_APPID = os.environ.get('WOLFRAM_ALPHA_APPID')

math_tool = WolframAlphaQueryRun(api_wrapper =WolframAlphaAPIWrapper())
# search_tool = DuckDuckGoSearchRun()
math_student = Agent(
    role = "Math Student at Georgia Tech University",
    goal = "You are an expert in solving mathematics problems related to differential and integral calculus",
    backstory = """You are one of the best students in the Maths department of the university. You can solve 
    complex mathematics problem related to differential and integral calculus.
    """,
    verbose=True,
    allow_delegation=False,
    tools=[math_tool]

)

math_professor = Agent(
    role = "Math professor at Georgia Tech University",
    goal = "You correct and provide the final right answers of math problems submitted by your students",
    backstory = """You are a professor at the university and have a PH.D in Mathematics. You are expert in grading and correcting 
    solutions to math problems related to Calculus.
    """,
    verbose=True,
    allow_delegation=True

)

solving_task = Task(description="""Approximate the integral of function 2 + square(x) using five equal intervals between -1 and +1. 
Please provide detailed steps of the solution.
""",agent=math_student)

grading_task = Task(description="""For the solution provided, review the solution and correct if required. Please provide detailed steps of the solution.
""",agent=math_professor)

crew = Crew(
  agents=[math_student, math_professor],
  tasks=[solving_task, grading_task],
  verbose=2, # You can set it to 1 or 2 to different logging levels
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)