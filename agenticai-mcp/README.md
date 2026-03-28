# MCP Context Demo (Model Context Protocol)

This project demonstrates a **simple educational implementation inspired by the Model Context Protocol (MCP)**.

MCP allows AI systems to connect to external tools and services in a structured and secure way.

## Project Structure

```
mcp-context-demo
│
├── src
│   ├── server.py
│   └── client.py
│
├── docs
│   └── architecture.md
│
├── presentation
│   └── mcp_presentation.pptx
│
└── README.md
```

## What This Project Demonstrates

- Tool registration
- Sending structured context
- Returning structured responses
- Simple client/server interaction

## Run the Demo

Navigate to the src folder:

```bash
cd src
python server.py
python client.py
```

## Example Concept

Client sends:

```
{
  "tool": "word_count",
  "context": {
      "text": "..."
  }
}
```

Server processes the request and returns structured output.

## Learning Goals

Understand:

- What MCP is
- How tools connect to AI
- Context passing
- Structured outputs

## Real MCP Resources

Official MCP implementations include:

- AI assistants
- tool servers
- data connectors
