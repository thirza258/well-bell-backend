import json
import os
import pandas as pd
import itertools
import numpy as np
import joblib

from django.shortcuts import render
from django.conf import settings
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import FormData
from langchain.schema import HumanMessage
from .serializers import FormDataSerializer

# Create your views here.
# Define prompt for question-answering
filepath = os.path.join(settings.BASE_DIR, 'handler', 'assets', 'FAQ_data.csv')
impact_filepath = os.path.join(settings.BASE_DIR, 'handler', 'assets', 'impact.csv')
dataset = pd.read_csv(filepath)
json_data = dataset.to_json(orient='records')
json_obj = json.loads(json_data)

dataset_impact = pd.read_csv(impact_filepath)
json_data_impact = dataset_impact.to_json(orient='records')
json_obj_impact = json.loads(json_data_impact)

llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)
vector_store_impact = InMemoryVectorStore(embeddings)

vector_store_impact.add_documents([
    Document(page_content=json.dumps(chunk)) for chunk in json_obj_impact
])

vector_store.add_documents([
    Document(page_content=json.dumps(chunk)) for chunk in json_obj
])
prompt = hub.pull("rlm/rag-prompt")
prompt_impact = hub.pull("rlm/rag-prompt")
# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"], k=5)
    if not retrieved_docs:
        return {"context": []}  # Ensure the context is at least an empty list
    return {"context": retrieved_docs}

def retrieve_impact(state: State):
    retrieved_docs = vector_store_impact.similarity_search(state["question"], k=3)
    if not retrieved_docs:
        return {"context": []}  # Ensure the context is at least an empty list
    return {"context": retrieved_docs}

def generate(state: State):
    if not state["context"]:
        return {"answer": "No relevant context found to answer the question."}

    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    custom_prompt = f"Based on the following context, answer the question:\n\nContext:\n{docs_content}\n\nQuestion: {state['question']}"
    messages = [HumanMessage(content=custom_prompt)]
    response = llm.invoke(messages)
    return {"answer": response.content}

def generate_impact(state: State):
    if not state["context"]:
        return {"answer": "No relevant context found to determine the health rating."}

    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    custom_prompt = f"Based on the following context, what is the health rating of the question: {state['question']}?\n\nGiven the Context:\n{docs_content}"
    messages = [HumanMessage(content=custom_prompt)]
    response = llm.invoke(messages)
    return {"answer": response.content}

# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

graph_builder_impact = StateGraph(State).add_sequence([retrieve_impact, generate_impact])
graph_builder_impact.add_edge(START, "retrieve_impact")
graph_impact = graph_builder_impact.compile()

model = joblib.load(os.path.join(settings.BASE_DIR, 'handler', 'assets', 'random_forest_model.pkl'))

class AskRAG(APIView):
    def post(self, request):
        question = request.data["question"]
        state = graph.invoke({"question": question})
        return Response({
            "status": 200,
            "message": "success",
            "data": state["answer"]
        }, status=status.HTTP_200_OK)

class FormDataAPIView(APIView):
    def post(self, request):
        serializer = FormDataSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            
            features = [
                "educationUse", "activities", "educationalApps", "dailyUsage",
                "performanceImpact", "usageDistraction", "beneficialSubjects",
                "usageSymptoms", "symptomFrequency", "healthPrecautions"
            ]
            
            print(serializer.data)

            # Prepare input data for the model
            input_data = {key: serializer.data[key] for key in features if key in serializer.data}

            # Label Encoding mappings
            label_encoders = {
                "educationUse": {"Never": 0, "Rarely": 1, "Sometimes": 2, "Frequently": 3, "Neutral": 2},
                "activities": {"Social Media": 0, "Web Browsing": 1, "Messaging": 2, "All of these": 3},
                "educationalApps": {"Educational Videos": 0, "Study Planner": 1, "Productivity Tools": 2, "Language": 3},
                "dailyUsage": {"< 2 hours": 0, "2-4 hours": 1, "4-6 hours": 2, "> 6 hours": 3},
                "performanceImpact": {"Strongly disagree": 0, "Disagree": 1, "Neutral": 2, "Agree": 3, "Strongly agree": 4},
                "usageDistraction": {"Not Distracting": 0, "While Studying": 1, "During Class Lectures": 2, "During Exams": 3},
                "beneficialSubjects": {"Accounting": 0, "Browsing Material": 1, "Research": 2},
                "usageSymptoms": {"Headache": 0, "Sleep disturbance": 1, "Anxiety or Stress": 2, "All of these": 3},
                "symptomFrequency": {"Never": 0, "Rarely": 1, "Sometimes": 2, "Frequently": 3},
                "healthPrecautions": {"Using Blue Light filter": 0, "Taking Break during prolonged use": 1, "Limiting Screen Time": 2, "None of Above": 3},
            }

            # Convert all list fields into individual rows (explode lists)
            list_fields = ["activities", "educationalApps", "beneficialSubjects", "usageSymptoms", "healthPrecautions"]
            exploded_rows = []

            # Generate all possible combinations of list elements
            list_combinations = list(itertools.product(
                input_data.get("activities", [None]),
                input_data.get("educationalApps", [None]),
                input_data.get("beneficialSubjects", [None]),
                input_data.get("usageSymptoms", [None]),
                input_data.get("healthPrecautions", [None]),
            ))

            for combo in list_combinations:
                row = input_data.copy()
                row["activities"], row["educationalApps"], row["beneficialSubjects"], row["usageSymptoms"], row["healthPrecautions"] = combo
                exploded_rows.append(row)

            # Convert to DataFrame
            input_df = pd.DataFrame(exploded_rows)

            # Encode categorical values
            for col, mapping in label_encoders.items():
                if col in input_df.columns:
                    input_df[col] = input_df[col].map(mapping).fillna(-1).astype(int)

            # Convert DataFrame to JSON string format
            input_df_string = input_df.to_json(orient='records')

            # Perform predictions
            predictions = model.predict(input_df)

            # Calculate final prediction (floor of mean)
                # Define health rating mapping
            health_rating_mapping = {0: "Excellent", 1: "Good", 2: "Fair", 3: "Poor", -1: "Unknown"}

            # Compute final prediction
            final_prediction = int(np.floor(np.mean(predictions)))

            # Encode final prediction into a health rating
            health_rating = health_rating_mapping.get(final_prediction, "Unknown")

            data = {
                "prediction": health_rating,  # Now returns "Excellent", "Good", etc.
                "description": graph_impact.invoke({"question": input_df_string})["answer"]
            }

            print(data)

            return Response({"status": 200, "message": "Data saved successfully", "data": data}, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


