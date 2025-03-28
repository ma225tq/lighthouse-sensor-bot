from flask_socketio import SocketIO, emit

# Initialize SocketIO without attaching it to an app yet
socketio = SocketIO()

# This will be called after the app is created
def init_socketio(app, cors_origins):
    socketio.init_app(app, cors_allowed_origins=cors_origins)
    return socketio

def setup_websocket_routes(socketio):
    @socketio.on('connect', namespace='/query')
    def handle_connect():
        """Handle client connection"""
        emit('connection_response', {'data': 'Connected'})
        
    @socketio.on('disconnect', namespace='/query')
    def handle_disconnect():
        """Handle client disconnection"""
        print('Client disconnected')