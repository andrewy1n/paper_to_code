from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
import os

@CrewBase
class PaperToCode():
    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self):
        self.llm = self._configure_llm()

    def _configure_llm(self):
        return {
            "config": {
                "base_url": "https://api.deepseek.com",
                "api_key": os.getenv("DEEPSEEK_API_KEY")
            },
            "model": "deepseek-reasoner"
        }
    
    @agent
    def code_architect(self) -> Agent:
        return Agent(
            config=self.agents_config['code_architect'],
            llm=self.llm,
            verbose=True
        )
    
    @agent
    def developer(self):
        return Agent(
            config=self.agents_config['developer'],
            llm=self.llm,
            verbose=True
        )
    
    @agent
    def code_review(self):
        return Agent(
            config=self.agents_config['code_review'],
            llm=self.llm,
            verbose=True
        )
    
    @agent
    def qa_engineer(self):
        return Agent(
            config=self.agents_config['qa_engineer'],
            llm=self.llm,
            verbose=True
        )
    
    def architecture_task(self, agent):
        return Task(
            config=self.tasks_config['architecture_task'],
            output_file="requirements.md",
            agent=agent
        )
    
    def development_task(self, agent, context):
        return Task(
            config=self.tasks_config['development_task'],
            agent=agent,
            context=context
        )
    
    def review_task(self, agent, context):
        return Task(
            config=self.tasks_config['review_task'],
            agent=agent,
            context=context
        )
    
    def qa_task(self, agent, context):
        return Task(
            config=self.tasks_config['qa_task'],
            agent=agent,
            context=context
        )

    @crew
    def crew(self) -> Crew:
        """Creates the PaperToCode crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
