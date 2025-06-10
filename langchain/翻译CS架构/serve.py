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
    base_url="https://openrouter.ai/api/v1",
    api_key='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',
    model="thudm/glm-4-32b:free",
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