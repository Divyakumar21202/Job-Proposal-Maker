from proposal_agent import app

user_input = input("Enter your query: ")


while user_input != "exit":
    result = app.invoke({"query":user_input})
    print("ai: ",result["result"])
    user_input = input("Enter your query: ")

