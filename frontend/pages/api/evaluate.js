import dotenv from 'dotenv';
import process from 'process';

const SERVER_URL = process.env.NEXT_PUBLIC_SERVER_URL;

export default async function handler(req, res) {
  if (req.method === "POST") {
    try {
      const { model_id } = req.body;
      
      if (!model_id) {
        return res.status(400).json({ error: "Model ID is required" });
      }
      
      console.log("Sending evaluation request to backend:", { model_id });
      
      const response = await fetch(`${SERVER_URL}/api/evaluate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ model_id }),
      });

      if (!response.ok) {
        console.error("Backend error:", response.status, response.statusText);
        throw new Error(`Error in Flask endpoint response: ${response.statusText}`);
      }

      const data = await response.json();
      console.log("Received evaluation results from backend:", data);
      
      return res.status(200).json(data);
    } catch (error) {
      console.error("Error during evaluation:", error);
      return res.status(500).json({ 
        error: "Failed to evaluate model", 
        message: error.message 
      });
    }
  } else {
    res.setHeader('Allow', ['POST']);
    res.status(405).end(`Method ${req.method} Not Allowed`);
  }
} 