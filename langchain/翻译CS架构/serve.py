from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langserve import add_routes

sys_template = "Translate the following text to French: {text}"
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", sys_template),
        ("user", "{text}"),
    ]
)
model = ChatOpenAI(
    model= MODEL,
    api_key = API_KEY,
    base_url = BASE_URL,
)

chain = prompt_template | model | StrOutputParser()
app = FastAPI(
    title="LangServe Example",
    description="An example of using LangServe with a translation model.",
    version="1.0.0",
)
add_routes(
    app,
    chain,
    path="/translate"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)