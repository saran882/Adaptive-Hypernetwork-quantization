from crewai import Agent, Task, Crew, LLM

try:
    llm = LLM(model='ollama/llama3.1', base_url='http://localhost:11434')
    
    agent = Agent(
        role="Assistant",
        goal="Say hello",
        backstory="Friendly AI",
        llm=llm
    )
    
    task = Task(description="Say hi to the user", expected_output="A greeting", agent=agent)
    crew = Crew(agents=[agent], tasks=[task])
    print(crew.kickoff())
except Exception as e:
    print("Error:", e)
