import process from 'process';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  const SERVER_URL = process.env.NEXT_PUBLIC_SERVER_URL;
  
  try {
    const response = await fetch(`${SERVER_URL}/api/export-chart`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(req.body),
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(`Error exporting chart: ${errorData.error || response.statusText}`);
    }
    
    const data = await response.json();
    res.status(200).json(data);
  } catch (error) {
    console.error('Error exporting chart:', error);
    res.status(500).json({ error: 'Failed to export chart' });
  }
} 