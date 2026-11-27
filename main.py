from proposal_agent import app

job_description = input("Enter job details: ")

user_input = "Please help me to make proposal for the given details."

while user_input != "exit":
    result = app.invoke({
        "job_description":job_description, 
        "user_input":user_input,
    })
    print("ai: ",result["result"])
    user_input = input("you: ")

