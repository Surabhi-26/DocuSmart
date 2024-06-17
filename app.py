import os
import tkinter as tk
from tkinter import filedialog
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI, OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from crewai.process import Process
import gradio as gr
import numpy as np


os.environ["GOOGLE_API_KEY"] = "YOUR-API-KEY"


from crewai_tools import PDFSearchTool
from crewai_tools import FileReadTool
from crewai_tools import DOCXSearchTool
from crewai_tools import TXTSearchTool
from crewai_tools import CSVSearchTool



llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    verbose=True,
    temperature=0.6,
    google_api_key=os.environ["GOOGLE_API_KEY"]
)


#--------------------------------------------Class for choosing agent---------------------------------------#
class agentCollection:

    def agentPDF(filepath):
        agentpdf = Agent(
            role="PDF Content Searcher and Writer",
            goal="Generate a detailed description of relevant content from a PDF provided by the user",
            backstory="You are an expert in navigating and extracting detailed information from PDF documents. Your task is to find the most relevant and accurate content within the PDF and provide a detailed description that addresses the user's query.",
            verbose=True,
            tools=[toolsCollection.toolPDF(filepath)],
            llm=llm,
            allow_delegation=False
           
        )
        return agentpdf

    def agentFile(filepath):
        agentfile = Agent(
            role="General File Content Searcher and Writer",
            goal="Generate a detailed description of relevant content from various file formats provided by the user",
            backstory="You have extensive experience in handling different types of files, like JSON and CSV. Your role is to expertly extract and describe the most pertinent information from any file format to meet the user's needs.",
            verbose=True,
            tools=[toolsCollection.toolFile(filepath)],
            llm=llm,
            allow_delegation=False
           
        )
        return agentfile

    def agentTXT(filepath):
        agenttxt = Agent(
            role="Text File Content Searcher and Writer",
            goal="Generate a detailed description of relevant content from text files provided by the user",
            backstory="You specialize in working with plain text files. Your job is to sift through the text and identify the most relevant information, providing a detailed description that fulfills the user's query.",
            verbose=True,
            tools=[toolsCollection.toolTXT(filepath)],
            llm=llm,
            allow_delegation=False
          
        )
        return agenttxt

    def agentDOCX(filepath):
        agentdoc = Agent(
            role="DOCX Content Searcher and Writer",
            goal="Generate a detailed description of relevant content from DOCX files provided by the user",
            backstory="You are proficient in reading and extracting detailed information from DOCX documents. Your expertise allows you to locate and describe the most relevant content within a DOCX file, ensuring the user's query is answered thoroughly and accurately.",
            verbose=True,
            tools=[toolsCollection.toolDOCX(filepath)],
            llm=llm,
            allow_delegation=False
           
        )
        return agentdoc
    
    def agentCSV(filepath):
        agentcsv = create_csv_agent(
            ChatGoogleGenerativeAI(temperature=0.6, model="gemini-1.5-flash-latest"),
            filepath,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
        )
        return agentcsv

    def agentContentWriter():
        agentwriter = Agent(
            role="Content Writer",
            goal="Summarize the data received from other agents into a comprehensive report or blog",
            backstory="""You are a skilled content writer with expertise in synthesizing information from various sources. Your task is to use the detailed descriptions provided by other agents to create a well-structured and coherent summary that addresses the user's query in detail.""",
            verbose=True,
            llm=llm,
            
        )
        return agentwriter

#--------------------------------------------Class for choosing tool---------------------------------------#

class toolsCollection:
 
    def toolPDF(filepath):
        if filepath == "":
            print("FILE NOT FOUND")
            return
        pdftool = PDFSearchTool(
            config=dict(
                llm=dict(
                    provider="google",
                    config=dict(
                        model="gemini-1.5-flash-latest",
                    ),
                ),
                embedder=dict(
                    provider="huggingface",
                    config=dict(
                        model="sentence-transformers/msmarco-distilbert-base-v4"
                    ),
                ),
            ),
            pdf=filepath
        )
        return pdftool

    def toolFile(filepath):
        filetool = FileReadTool(
            config=dict(
                llm=dict(
                    provider="google",
                    config=dict(
                        model="gemini-1.5-flash-latest",
                    ),
                ),
                embedder=dict(
                    provider="huggingface",
                    config=dict(
                        model="sentence-transformers/msmarco-distilbert-base-v4"
                    ),
                ),
            ),
            file_path=filepath
        )
        return filetool

    def toolTXT(filepath):
        txttool = TXTSearchTool(
            config=dict(
                llm=dict(
                    provider="google",
                    config=dict(
                        model="gemini-1.5-flash-latest",
                    ),
                ),
                embedder=dict(
                    provider="huggingface",
                    config=dict(
                        model="sentence-transformers/msmarco-distilbert-base-v4"
                    ),
                ),
            ),
            txt=filepath
        )
        return txttool

    def toolDOCX(filepath):
        if filepath == "":
            print("FILE NOT FOUND")
            return
        docxtool = DOCXSearchTool(
            config=dict(
                llm=dict(
                    provider="google",
                    config=dict(
                        model="gemini-1.5-flash-latest",
                    ),
                ),
                embedder=dict(
                    provider="huggingface",
                    config=dict(
                        model="sentence-transformers/msmarco-distilbert-base-v4"
                    ),
                ),
            ),
            docx=filepath
        )
        return docxtool

    def toolCSV(filepath):
        csvtool = CSVSearchTool(
            config=dict(
                llm=dict(
                    provider="google",
                    config=dict(
                        model="gemini-1.5-flash-latest",
                    ),
                ),
                embedder=dict(
                    provider="huggingface",
                    config=dict(
                        model="sentence-transformers/msmarco-distilbert-base-v4"
                    ),
                ),
            ),
            csv=filepath
        )
        return csvtool

def run_ai(file, query, required_ans_format):
    filepath = file.name

    if filepath.endswith(".pdf"):
        myagent = agentCollection.agentPDF(filepath)
    elif filepath.endswith(".json"):
        myagent = agentCollection.agentFile(filepath)
    elif filepath.endswith(".docx"):
        myagent = agentCollection.agentDOCX(filepath)
    elif filepath.endswith(".txt"):
        myagent = agentCollection.agentTXT(filepath)
    elif filepath.endswith(".csv"):
        myagent = agentCollection.agentCSV(filepath)
        return myagent.run(query)

    task = Task(
        description=f"{query}",
        expected_output=f"detailed description on {query}",
        agent=myagent,
    )

    content_writer_agent = agentCollection.agentContentWriter()
    content_writer_task = Task(
        description=f"{query}",
        expected_output=f'{required_ans_format}',
        agent=content_writer_agent,
    )

    crew = Crew(
        agents=[myagent, content_writer_agent],
        tasks=[task, content_writer_task],
        process=Process.sequential,
        verbose=2
    )

    result = crew.kickoff()
    return result

testing_folder = os.path.join(os.getcwd(), "Testing Folder")
interface = gr.Interface(
    fn=run_ai,
    inputs=[
        gr.File(label="Upload File"),
        gr.Textbox(label="Query"),
        gr.Textbox(label="Expected Output")
    ],
    outputs="text",
    title="DocuSmart",
    description=(
        "Upload a file (CSV, PDF, DOCX, TXT, JSON) and enter your query to get detailed information.\n\n"
        "### Instructions:\n"
        "1. Upload the file you want to talk to.\n"
        "2. Enter your question in the Query field.\n"
        "3. Specify the desired output format, e.g., one line answer.\n"
        "4. Press 'Submit' and wait for the response.\n\n"
    ),
    examples=[
        [(os.path.join(testing_folder, "LabManual.pdf")), "What is RIP?", "detailed description"],
        [(os.path.join(testing_folder, "ElectricCarData_Clean.csv")), "Which Brand has most vehicles?", "one line answer"]
    ],
    theme=gr.themes.Soft()
)

interface.launch()
