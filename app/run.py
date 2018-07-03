# Import application from the app folder
from app import app

# Running the server on localhost with port 8080 
app.run(host='0.0.0.0', port=8080, debug=True)
