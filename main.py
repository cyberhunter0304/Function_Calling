from flask import Flask, request, jsonify
from llm_functions import ask_llm_which_function_to_call
import os

app = Flask(__name__)

reservations = []
reservation_counter = 1

#Functions to manage reservations
def make_reservation(name, date, time, party_size):
    global reservation_counter
    
    reservation = {
        "id": reservation_counter,
        "name": name,
        "date": date,
        "time": time,
        "party_size": int(party_size),
        "status": "confirmed"
    }
    
    reservations.append(reservation)
    reservation_counter += 1
    
    return {
        "success": True,
        "message": f"✅ Reservation created successfully for {name}",
        "reservation": reservation
    }
#Function to list all reservations
def list_reservations():
    if not reservations:
        return {
            "success": True,
            "message": "No reservations found",
            "reservations": []
        }
    
    return {
        "success": True,
        "message": f"Found {len(reservations)} reservation(s)",
        "reservations": reservations
    }
#Function to cancel a reservation by ID
def cancel_reservation(reservation_id):
    global reservations
    
    reservation_id = int(reservation_id)
    
    for reservation in reservations:
        if reservation["id"] == reservation_id:
            reservations.remove(reservation)
            return {
                "success": True,
                "message": f"✅ Reservation #{reservation_id} cancelled successfully",
                "cancelled_reservation": reservation
            }
    
    return {
        "success": False,
        "message": f"❌ Reservation #{reservation_id} not found"
    }

#Tools available for LLM to call
AVAILABLE_TOOLS = {
    "make_reservation": make_reservation,
    "list_reservations": list_reservations,
    "cancel_reservation": cancel_reservation
}

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                "error": "Missing 'message' field in request body" #Message is Missing
            }), 400
        
        user_message = data['message'] #User Message is passed as input to LLM
        llm_decision = ask_llm_which_function_to_call(user_message) 
        #llm_functions.py is called here passing user message as input  
        
        if llm_decision.get('function_name') == 'none':
            return jsonify({
                "type": "direct_response",
                "response": llm_decision.get('response'),
                "function_called": None
            })
        
        #Returned JSON from llm_functions.py is processed here (line 103)
        function_name = llm_decision['function_name']
        parameters = llm_decision.get('parameters', {})
        
        #Validate function name
        #Prevent invalid function calls
        if function_name not in AVAILABLE_TOOLS:
            return jsonify({
                "error": f"Unknown function: {function_name}"
            }), 400
        
        function_to_call = AVAILABLE_TOOLS[function_name]
        result = function_to_call(**parameters) #unpack parameters dictionary to keyword arguments
        
        #RUNS the appropriate function
        #RETURNS the result

        return jsonify({
            "type": "function_execution",
            "llm_reasoning": llm_decision.get('reasoning'),
            "function_called": function_name,
            "parameters_used": parameters,
            "result": result
        })
    
        #Sends the final response back to the user (POSTMAN)
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "type": "error"
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)