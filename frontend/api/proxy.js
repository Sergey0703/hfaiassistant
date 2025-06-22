// File: frontend/api/proxy.js
const HF_BASE_URL = 'https://sergey070373-aiassistant.hf.space';

export default async function handler(req, res) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  // Handle preflight OPTIONS request
  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  try {
    // Extract the path from the request URL
    const targetPath = req.query.path || '';
    const targetUrl = `${HF_BASE_URL}${targetPath}`;

    console.log(`Proxying ${req.method} ${targetPath} -> ${targetUrl}`);

    // Prepare headers
    const headers = {
      'Accept': 'application/json',
      'User-Agent': 'Legal-Assistant-Proxy/1.0',
    };

    // Add Content-Type for POST/PUT requests
    if (req.method === 'POST' || req.method === 'PUT') {
      headers['Content-Type'] = 'application/json';
    }

    // Prepare request body
    let body = undefined;
    if (req.method === 'POST' || req.method === 'PUT') {
      body = JSON.stringify(req.body);
    }

    // Make request to HuggingFace using native fetch (Node 18+)
    const response = await fetch(targetUrl, {
      method: req.method,
      headers: headers,
      body: body,
    });

    console.log(`Response: ${response.status} ${response.statusText}`);

    // Get response text first
    const responseText = await response.text();
    
    // Try to parse as JSON, fallback to text
    let responseData;
    try {
      responseData = JSON.parse(responseText);
    } catch (e) {
      responseData = { message: responseText };
    }

    // Return response with same status code
    res.status(response.status).json(responseData);

  } catch (error) {
    console.error('Proxy error:', error);
    res.status(500).json({ 
      error: 'Proxy error', 
      message: error.message,
      details: error.toString()
    });
  }
}