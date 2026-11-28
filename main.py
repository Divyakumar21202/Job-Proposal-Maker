from proposal_agent import app
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("DivyaKumar_Resume_IIIT.pdf")

docs = loader.load()

user_profile = "\n".join(doc.page_content for doc in docs)

job_description = input("Enter job details: ")

user_input = "Please help me to make proposal for the given details."

while user_input != "exit":
    result = app.invoke({
        "job_description":job_description, 
        "user_input":user_input,
        "user_profile":user_profile,
    })
    print("ai: ",result["result"])
    user_input = input("you: ")

