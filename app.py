#import dependencies
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

#load environment variables
load_dotenv()

# A Simple UI
st.title("Startup GPT 🚀")
text = st.text_input("Write the industry for your startup")

#setup llm
llm = OpenAI()

#setup prompts
idea_template = "List 5 startup ideas in the {industry} industry without any explanation to it."
idea_prompt = PromptTemplate(input_variables=["industry"],template=idea_template)

explanation_template = "Explain this startup idea in more than 8 words and less than 15 words : {idea}"
explanation_prompt = PromptTemplate(input_variables=["idea"], template = explanation_template)

startupName_template = "Suggest a name of a startup for the idea: {idea}"
startupName_prompt = PromptTemplate(input_variables=["idea"], template = startupName_template)

userPersona_template = "Define the Target User Persona for the startup idea: {idea}"
userPersona_prompt = PromptTemplate(input_variables=["idea"], template = userPersona_template)

userPainPoints_template = "Identify and explain the main problems the startup will solve for users for the startup idea: {idea}"
userPainPoints_prompt = PromptTemplate(input_variables=["idea"], template = userPainPoints_template)

USP_template = "Explain the USP for the startup idea: {idea}"
USP_prompt = PromptTemplate(input_variables=["idea"], template = USP_template)

#setup llm chains
idea_chain = LLMChain(llm = llm, prompt = idea_prompt)
explanation_chain = LLMChain(llm=llm, prompt = explanation_prompt)
startupName_chain = LLMChain(llm=llm, prompt = startupName_prompt)
userPersona_chain = LLMChain(llm=llm, prompt = userPersona_prompt)
userPainPoints_chain = LLMChain(llm=llm, prompt = userPainPoints_prompt)
USP_chain = LLMChain(llm=llm,prompt = USP_prompt)

#response from gpt
if text:
    #st.write(prompt.format(industry = text))
    response = idea_chain.run(industry = text)
    #st.write(response)
    response_list = response.split('\n')
    del response_list[:2]
    
    for idea in response_list:
        with st.expander(idea):
            explanation = explanation_chain.run(idea = idea)
            name=startupName_chain.run(idea=idea)
            userPersona = userPersona_chain.run(idea = idea)
            userPainPoints = userPainPoints_chain.run(idea = idea)
            USP = USP_chain.run(idea = idea)
            st.info(explanation)
            st.info(f"**Startup Name** \n {name}")
            st.info(f"**USP** \n {USP}")
            st.info(f"**User Persona** \n {userPersona}")

