from flask_socketio import emit

def setup_websocket_routes(socketio):
    @socketio.on('connect', namespace='/query')
    def handle_connect():
        """Handle client connection"""
        emit('connection_response', {'data': 'Connected'})