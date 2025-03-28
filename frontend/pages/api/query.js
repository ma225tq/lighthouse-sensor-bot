import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';
import process from 'process';


const SERVER_URL = process.env.NEXT_PUBLIC_SERVER_URL;

export default async function handler(req, res) {
  if (req.method === "POST") {
    try {
      const { question, source_file, prompt_file, llm_model_id } = req.body;
      
      let questionText = question;
      
      // If prompt_file is provided, read the content from the file.
      if (prompt_file) {
        const promptPath = path.join(process.cwd(), 'public', 'prompts', prompt_file);
        
        if (fs.existsSync(promptPath)) {
          questionText = fs.readFileSync(promptPath, 'utf8');
        } else {
          return res.status(400).json({ content: `Error: Prompt file '${prompt_file}' not found` });
        }
      }
      
      if (!questionText) {
        return res.status(400).json({ content: "Error: No question provided" });
      }
      
      console.log("Sending to backend:", { question: questionText, source_file, llm_model_id });
      
      const response = await fetch(`${SERVER_URL}/api/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ 
          question: questionText,
          source_file,
          llm_model_id: llm_model_id
        }),
      });

      if (!response.ok) {
        console.error("Backend error:", response.status, response.statusText);
        throw new Error("Error in Flask endpoint response");
      }

      const data = await response.json();
      console.log("Received from backend:", data);
      
      // Normalize the response format.
      if (data.response) {
        // If backend returns 'response' key, convert it to 'content'.
        res.status(200).json({ 
          content: data.response,
          full_response: data.full_response || data.response 
        });
      } else if (data.content) {
        // If backend already returns 'content' key, use it directly.
        res.status(200).json({ 
          content: data.content,
          full_response: data.full_response || data.content 
        });
      } else {
        res.status(200).json({ 
          content: "Response received but no content found",
          full_response: data.full_response || "No full response available" 
        });
      }

    } catch (error) {
      console.error(error);
      res.status(500).json({ content: `Internal Server Error: ${error.message}` });
    }
  } else {
    res.status(405).json({ content: "Method Not Allowed" });
  }
}