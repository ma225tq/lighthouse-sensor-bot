import React, { createContext, useContext, useEffect, useState } from 'react';
import { io } from 'socket.io-client';

const WebSocketContext = createContext(null);

export const WebSocketProvider = ({ children }) => {
  const [socket, setSocket] = useState(null);
  const [sqlQueries, setSqlQueries] = useState([]);
  const [queryStatus, setQueryStatus] = useState('idle');
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    let SERVER_URL;
    if (process.env.DEV_MODE) {
      SERVER_URL = process.env.SERVER_URL;
    } else {
      SERVER_URL = process.env.NEXT_PUBLIC_SERVER_URL || 'http://localhost:5000';
    }
    
    console.log('Connecting to WebSocket server at:', SERVER_URL);
    
    // Initialize socket connection
    const socketInstance = io(`${SERVER_URL}/query`, {
      transports: ['websocket'],
      autoConnect: true,
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000
    });

    // Set up event listeners
    socketInstance.on('connect', () => {
      console.log('WebSocket connected');
      setConnected(true);
    });

    socketInstance.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      setConnected(false);
    });

    socketInstance.on('log_message', (data) => {
      if (data.type === 'sql_query') {
        setSqlQueries(prev => [...prev, data.content]);
      }
    });

    socketInstance.on('query_status', (data) => {
      setQueryStatus(data.status);
    });

    socketInstance.on('disconnect', () => {
      console.log('WebSocket disconnected');
      setConnected(false);
    });

    // Save socket instance
    setSocket(socketInstance);

    // Clean up on unmount
    return () => {
      socketInstance.disconnect();
    };
  }, []);

  // Reset queries when starting a new query
  const resetQueries = () => {
    setSqlQueries([]);
    setQueryStatus('idle');
  };

  return (
    <WebSocketContext.Provider value={{ 
      socket, 
      sqlQueries, 
      queryStatus, 
      resetQueries,
      connected 
    }}>
      {children}
    </WebSocketContext.Provider>
  );
};

export const useWebSocket = () => useContext(WebSocketContext); 