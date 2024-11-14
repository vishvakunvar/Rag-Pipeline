import streamlit as st
import pandas as pd
import os
import plotly.express as px
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import logging
import re
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.pydantic_v1 import BaseModel, Field
import shutil
import chromadb


def main_spreadsheet():
    # Setup logging and load environment variables
    logging.basicConfig(level=logging.INFO)
    load_dotenv()

    # Set up API keys and configurations
    google_api_key = os.getenv("GOOGLE_API_KEY")
    favicon_path = "YellowFavicon.png"

    # Initialize the embedding model
    embeddings = HuggingFaceEmbeddings()

    # Initialize the LLM with Google Gemini Pro API
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=google_api_key)


    class RephrasedQuery(BaseModel):
        rephrased_query: str = Field(description="The rephrased version of the original query")


    def Rephrase_query(input_string: str) -> dict:
        model = llm
        output_parser = JsonOutputParser(pydantic_object=RephrasedQuery)
        prompt = PromptTemplate(
            template="spellcheck the user query, while maintaining its original meaning:\n\nQuery: {query}\n\n{format_instructions}\nRephrased query:",
            input_variables=["query"],
            partial_variables={"format_instructions": output_parser.get_format_instructions()}
        )
        chain = prompt | model | output_parser
        result = chain.invoke({"query": input_string})
        return result


    # Function to process Excel or CSV file and store in CHROMA.
    def process_file(file):
        try:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
        except ValueError:
            df = pd.read_excel(file, engine='xlrd')

        texts = df.astype(str).apply(lambda x: ' '.join(x), axis=1).tolist()
        metadatas = df.to_dict('records')
        metadatas = [{str(key): value for key, value in metadata.items()} for metadata in metadatas]

        # Set up a persistent directory for Chroma
        persist_directory = "./chroma_db"

        # Clear existing Chroma database if it exists
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)

        try:
            # Clear system cache of chromadb before creating vectore store instance
            # Add check to avoid ValueError: Could not connect to tenant default_tenant
            chromadb.api.client.SharedSystemClient.clear_system_cache()
            # Create Chroma vector store
            vectorstore = Chroma.from_texts(
                collection_name="vishva",
                texts=texts,
                embedding=embeddings,
                metadatas=metadatas,
                persist_directory=persist_directory
            )
            vectorstore.persist()
        except Exception as e:
            st.error(f"Error creating vectorstore: {str(e)}")
            return None, None

        return vectorstore, df
    
    st.title("Ask a question or request a chart! ")


    # File uploader for Excel and CSV files
    uploaded_file = st.sidebar.file_uploader("Choose an Excel or CSV file", type=["xlsx", "xls", "csv"])

    # Global DataFrame to store the uploaded Excel or CSV data
    if "df" not in st.session_state:
        st.session_state.df = None

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "file_processed" not in st.session_state:
        st.session_state.file_processed = False


    def reset_state():
        st.session_state.df = None
        st.session_state.vectorstore = None
        st.session_state.file_processed = False


    if not uploaded_file and st.session_state.file_processed:
        reset_state()

    # Process the uploaded file if it exists and hasn't been processed
    if uploaded_file and not st.session_state.file_processed:
        upld_btn = st.sidebar.button("Upload & Process File")
        if upld_btn:
            try:
                with st.spinner("Processing file...Please wait"):
                    vectorstore, df = process_file(uploaded_file)
                    if vectorstore is not None and df is not None:
                        st.sidebar.success("File processed and stored in CHROMA.")
                        st.session_state.vectorstore = vectorstore
                        st.session_state.df = df
                        st.session_state.file_processed = True
                    else:
                        st.sidebar.error("Error processing file. Please try again.")
            except Exception as e:
                st.sidebar.error(f"Error processing file: {str(e)}")
                logging.error(f"File processing error: {str(e)}")

    # CSS for light blue box
    instructions_style = """
        <style>
        .blue-box {
            background-color: #0e1117;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #91d5ff;
        }
        [data-testid="stHeadingWithActionElements"] a {
            display : none;
        }
        </style>
    """

    # Apply the CSS
    st.sidebar.markdown(instructions_style, unsafe_allow_html=True)

    # Instructions inside a light blue box
    st.sidebar.markdown("""
        <div class="blue-box">
            <h3>Welcome to Brainwave AI!</h3>
            <p>This application allows you to upload and analyze Excel or CSV files for quick information retrieval and dynamic chart generation.</p>
            <h4>How to use:</h4>
            <ol>
                <li>Click 'Browse File' to select your Excel or CSV file.</li>
                <li>Click 'Upload & Process File' to process your file.</li>
                <li>Once processed, you can enter a query or ask for a chart (e.g., "Show me a bar chart of Sales_Q1 vs Category").</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    st.sidebar.markdown(" ")
    st.sidebar.markdown("© 2024 Brainwave AI")

    # Initialize chat history and memory
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "charts" not in st.session_state:
        st.session_state.charts = []


    # Function to handle both conversation and chart generation
    def handle_conversation(user_input):
        if not hasattr(st.session_state, 'df') or st.session_state.df is None:
            return None, AIMessage(content="Please upload and process an Excel or CSV file first.")

        df = st.session_state.df

        chart_types = [
            "bar", "line", "scatter", "pie", "donut", "area", "treemap", "bar polar", "funnel", "scatter 3d",
            "scatter polar",
            "scatter ternary", "scatter map", "scatter mapbox", "scatter geo", "line 3d", "line polar", "line ternary",
            "line map", "line mapbox", "line geo", "timeline", "violin", "box", "strip", "histogram", "ecdf",
            "scatter matrix", "parallel coordinates", "parallel categories", "choropleth", "density contour",
            "density heatmap",
            "sunburst", "icicle", "funnel area", "choropleth map", "choropleth mapbox", "density map", "density mapbox",
            "heatmap", "stacked column"
        ]

        chart_type = next((ct for ct in chart_types if re.search(ct, user_input, re.IGNORECASE)), None)
        columns_in_query = [col for col in df.columns if re.search(re.escape(str(col)), user_input, re.IGNORECASE)]

        # Handle data-related queries
        if any(keyword in user_input.lower() for keyword in ["total", "sum", "average", "mean", "max", "min", "count"]):
            if not columns_in_query:
                return None, AIMessage(content="Please specify a column for the calculation.")

            column = columns_in_query[0]
            if "total" in user_input.lower() or "sum" in user_input.lower():
                result = df[column].sum()
                return None, AIMessage(content=f"The total {column} is {result}")
            elif "average" in user_input.lower() or "mean" in user_input.lower():
                result = df[column].mean()
                return None, AIMessage(content=f"The average {column} is {result}")
            elif "max" in user_input.lower():
                result = df[column].max()
                return None, AIMessage(content=f"The maximum {column} is {result}")
            elif "min" in user_input.lower():
                result = df[column].min()
                return None, AIMessage(content=f"The minimum {column} is {result}")
            elif "count" in user_input.lower():
                result = df[column].count()
                return None, AIMessage(content=f"The count of {column} is {result}")

        # Handle chart generation
        if chart_type and len(columns_in_query) >= 2:
            x_axis = columns_in_query[1] if len(columns_in_query) > 1 else None
            y_axis = columns_in_query[0]

        fig = None
        if chart_type == "bar":
            fig = px.bar(df, x=x_axis, y=y_axis)
        elif chart_type == "line":
            fig = px.line(df, x=x_axis, y=y_axis)
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x_axis, y=y_axis)
        elif chart_type == "pie":
            fig = px.pie(df, names=y_axis, values=x_axis)
        elif chart_type == "donut":
            fig = px.pie(df, names=y_axis, values=x_axis, hole=0.3)
        elif chart_type == "area":
            fig = px.area(df, x=x_axis, y=y_axis)
        elif chart_type == "treemap":
            fig = px.treemap(df, path=[x_axis], values=y_axis)
        elif chart_type == "bar polar":
            fig = px.bar_polar(df, r=y_axis, theta=x_axis)
        elif chart_type == "funnel":
            fig = px.funnel(df, x=x_axis, y=y_axis)
        elif chart_type == "scatter 3d":
            fig = px.scatter_3d(df, x=x_axis, y=y_axis, z=y_axis)
        elif chart_type == "scatter polar":
            fig = px.scatter_polar(df, r=y_axis, theta=x_axis)
        elif chart_type == "scatter ternary":
            fig = px.scatter_ternary(df, a=x_axis, b=y_axis, c=x_axis)
        elif chart_type == "scatter map":
            fig = px.scatter_map(df, lat=x_axis, lon=y_axis)
        elif chart_type == "scatter mapbox":
            fig = px.scatter_mapbox(df, lat=x_axis, lon=y_axis)
        elif chart_type == "scatter geo":
            fig = px.scatter_geo(df, lat=x_axis, lon=y_axis)
        elif chart_type == "line 3d":
            fig = px.line_3d(df, x=x_axis, y=y_axis, z=y_axis)
        elif chart_type == "line polar":
            fig = px.line_polar(df, r=y_axis, theta=x_axis)
        elif chart_type == "line ternary":
            fig = px.line_ternary(df, a=x_axis, b=y_axis, c=x_axis)
        elif chart_type == "line map":
            fig = px.line_map(df, lat=x_axis, lon=y_axis)
        elif chart_type == "line mapbox":
            fig = px.line_mapbox(df, lat=x_axis, lon=y_axis)
        elif chart_type == "line geo":
            fig = px.line_geo(df, lat=x_axis, lon=y_axis)
        elif chart_type == "timeline":
            fig = px.timeline(df, x_start=x_axis, x_end=y_axis, y=x_axis)
        elif chart_type == "violin":
            fig = px.violin(df, x=x_axis, y=y_axis)
        elif chart_type == "box":
            fig = px.box(df, x=x_axis, y=y_axis)
        elif chart_type == "strip":
            fig = px.strip(df, x=x_axis, y=y_axis)
        elif chart_type == "histogram":
            fig = px.histogram(df, x=x_axis, y=y_axis)
        elif chart_type == "ecdf":
            fig = px.ecdf(df, x=x_axis, y=y_axis)
        elif chart_type == "scatter matrix":
            fig = px.scatter_matrix(df, dimensions=[x_axis, y_axis])
        elif chart_type == "parallel coordinates":
            fig = px.parallel_coordinates(df, dimensions=[x_axis, y_axis])
        elif chart_type == "parallel categories":
            fig = px.parallel_categories(df, dimensions=[x_axis, y_axis])
        elif chart_type == "choropleth":
            fig = px.choropleth(df, locations=x_axis, color=y_axis)
        elif chart_type == "density contour":
            fig = px.density_contour(df, x=x_axis, y=y_axis)
        elif chart_type == "density heatmap":
            fig = px.density_heatmap(df, x=x_axis, y=y_axis)
        elif chart_type == "sunburst":
            fig = px.sunburst(df, path=[x_axis], values=y_axis)
        elif chart_type == "icicle":
            fig = px.icicle(df, path=[x_axis], values=y_axis)
        elif chart_type == "funnel area":
            fig = px.funnel_area(df, x=x_axis, y=y_axis)
        elif chart_type == "choropleth map":
            fig = px.choropleth_map(df, locations=x_axis, color=y_axis)
        elif chart_type == "choropleth mapbox":
            fig = px.choropleth_mapbox(df, locations=x_axis, color=y_axis)
        elif chart_type == "density map":
            fig = px.density_map(df, lat=x_axis, lon=y_axis)
        elif chart_type == "density mapbox":
            fig = px.density_mapbox(df, lat=x_axis, lon=y_axis)
            # Add more chart types here...

        if fig:
            return fig, AIMessage(content=f"Here is your {chart_type} chart for {y_axis} vs {x_axis}.")

        # Handle conversational query
        if "vectorstore" in st.session_state and st.session_state.vectorstore:
            try:
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5}),
                    memory=st.session_state.chat_history
                )
                result = qa_chain({"question": user_input})
                return None, AIMessage(content=result["answer"])
            except Exception as e:
                logging.error(f"Error in conversational query: {str(e)}")
                return None, AIMessage(
                    content="I'm sorry, I encountered an error while processing your query. Please try again.")

        return None, AIMessage(
            content="Sorry, I couldn't interpret your request. Please mention the chart type and column names correctly.")


    # Display chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "chart" in message:
                st.plotly_chart(message["chart"])

    # Handle user input
    if prompt := st.chat_input("Enter a Prompt Here"):
        prompt = Rephrase_query(prompt)["rephrased_query"]

        with st.container():
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                with st.spinner("Generating response..."):
                    chart, response = handle_conversation(prompt)
                    st.markdown(response.content)
                    if chart:
                        st.plotly_chart(chart)
                        st.session_state.messages.append({"role": "assistant", "content": response.content, "chart": chart})
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": response.content})

    hide_default_loader = f"""
    <style>
    div[data-testid="stToolbar"] {{    
        visibility: hidden;    
        height: 50%;    
        position: fixed;
    }}
    div[data-testid="stDecoration"]{{    
        visibility: hidden;    
        height: 0%;    
        position: fixed;
    }}
    .stFileUploader{{    
        margin-top : 4%
    }}
    div[data-testid="stStatusWidget"]{{    
        visibility: hidden;    
        height: 50%;    
        position: fixed;
    }}
    #MainMenu{{    
        visibility: hidden;    
        height: 0%;
    }}
    header{{    
        visibility: hidden;    
        height: 0%;
    }}
    footer{{   
        visibility: hidden;
        height: 0%;
    }}
    .st-emotion-cache-qcqlej{{
        display:{"block" if(prompt) else "none" };
        flex-grow:{"1" if(prompt) else "" }; 
    }}
    .st-emotion-cache-bm2z3a{{
        justify-content:{"" if(prompt) else "center" }; 
    }}
    .st-emotion-cache-1eo1tir{{
        padding:1rem
    }}
       </style>
       """
    st.markdown(hide_default_loader, unsafe_allow_html=True)

if __name__ == "__main__":
    main_spreadsheet()
