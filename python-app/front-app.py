from flask import Flask, request, jsonify, render_template
from langchain_core.messages import HumanMessage
from app import app_instance
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rag_system")

# Initialize the state
state = {
    "messages": []
}

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Main route for the application.
    Handles user input and displays responses.
    """
    global state

    if request.method == "POST":
        user_input = request.form.get("query")
        
        if user_input:
            # Update the state with the user's input
            state["messages"].append(HumanMessage(content=user_input))
            
            try:
                # Invoke the application
                new_state = app_instance.invoke(state, config={"configurable": {"thread_id": 1}})
                final_response = new_state["messages"][-1].content
                
                # Update the state for the next iteration
                state["messages"] = [HumanMessage(content=final_response)]
                
                return jsonify({"response": final_response})
            except Exception as e:
                logger.error(f"Error during invocation: {e}")
                return jsonify({"error": "An error occurred during processing. Please try again."})
    
    return render_template("index.html")


@app.route("/reset", methods=["POST"])
def reset():
    """
    Reset the conversation state.
    """
    global state
    state = {"messages": []}
    return jsonify({"message": "Conversation reset successfully."})


if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=True)
