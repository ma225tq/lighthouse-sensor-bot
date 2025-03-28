import logging
from flask_socketio import emit

class WebSocketLogHandler(logging.Handler):
    """Custom logging handler that emits logs to WebSocket clients"""
    
    def __init__(self, socket_event='log_message'):
        super().__init__()
        self.socket_event = socket_event
        self.current_query = ""
        
    def emit(self, record):
        try:
            log_entry = self.format(record)
            
            # Extract SQL query if present
            if "Running:" in log_entry:
                # If we have a query in progress, emit it before starting a new one
                if self.current_query:
                    try:
                        emit(self.socket_event, {
                            'type': 'sql_query',
                            'content': self.current_query.strip()
                        }, namespace='/query', broadcast=True)
                    except Exception as e:
                        print(f"Error emitting previous query: {e}")
                
                # Start a new query
                self.current_query = log_entry.split("Running:", 1)[1].strip()
                
                # Emit the new query immediately
                try:
                    emit(self.socket_event, {
                        'type': 'sql_query',
                        'content': self.current_query.strip()
                    }, namespace='/query', broadcast=True)
                except Exception as e:
                    print(f"Error emitting websocket message: {e}")
            
            # If this is a continuation of the current query, update it
            elif self.current_query and log_entry.strip() and not log_entry.strip().startswith("INFO"):
                self.current_query += " " + log_entry.strip()
                
        except Exception as e:
            print(f"Error in WebSocketLogHandler: {e}")
            self.handleError(record)
    
    def flush(self):
        """Ensure any pending query is emitted when the handler is flushed"""
        if self.current_query:
            try:
                emit(self.socket_event, {
                    'type': 'sql_query',
                    'content': self.current_query.strip()
                }, namespace='/query', broadcast=True)
                self.current_query = ""
            except Exception as e:
                print(f"Error emitting final query during flush: {e}")